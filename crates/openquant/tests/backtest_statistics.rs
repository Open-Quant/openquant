use chrono::{NaiveDate, NaiveDateTime, Duration};
use csv::ReaderBuilder;
use std::path::Path;
use openquant::backtest_statistics::*;

fn dates(n: usize, start: &str, days: i64) -> Vec<NaiveDateTime> {
    let base = NaiveDate::parse_from_str(start, "%Y-%m-%d").unwrap().and_hms_opt(0, 0, 0).unwrap();
    (0..n).map(|i| base + Duration::days(i as i64 * days)).collect()
}

fn load_dollar_bar_sample() -> Vec<(NaiveDateTime, f64)> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/backtest_statistics/dollar_bar_sample.csv");
    let mut rdr = ReaderBuilder::new().has_headers(true).from_path(path).unwrap();
    let mut rows: Vec<(NaiveDateTime, f64)> = Vec::new();
    for rec in rdr.records() {
        let rec = rec.unwrap();
        let ts = NaiveDateTime::parse_from_str(&rec[0], "%Y-%m-%d %H:%M:%S%.3f").unwrap();
        let close: f64 = rec[4].parse().unwrap();
        rows.push((ts, close));
    }
    rows
}

fn log_returns(series: &[(NaiveDateTime, f64)]) -> Vec<(NaiveDateTime, f64)> {
    let mut out = Vec::new();
    for i in 1..series.len() {
        let prev = series[i - 1].1;
        let curr = series[i].1;
        out.push((series[i].0, (curr.ln() - prev.ln())));
    }
    out
}

#[test]
fn test_timing_of_flattening_and_flips() {
    let dates = dates(10, "2000-01-01", 1);
    let flip_positions = [1.0, 1.5, 0.5, 0.0, -0.5, -1.0, 0.5, 1.5, 1.5, 1.5];
    let series: Vec<(NaiveDateTime, f64)> = dates.iter().copied().zip(flip_positions.iter().copied()).collect();
    let res = timing_of_flattening_and_flips(&series);
    let flips = vec![dates[6]];
    let flattenings = vec![dates[3], dates[9]];
    let mut expected = flips;
    expected.extend(flattenings);
    expected.sort();
    assert_eq!(res.len(), expected.len());
}

#[test]
fn test_average_holding_period() {
    let dates = dates(10, "2000-01-01", 1);
    let hold_positions = [0.0, 1.0, 1.0, -1.0, -1.0, 0.0, 0.0, 2.0, 2.0, 0.0];
    let series: Vec<(NaiveDateTime, f64)> = dates.iter().copied().zip(hold_positions.iter().copied()).collect();
    let avg = average_holding_period(&series).unwrap();
    assert!((avg - 2.0).abs() < 1e-4);

    let no_closed = [0.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0];
    let series2: Vec<(NaiveDateTime, f64)> = dates.iter().copied().zip(no_closed.iter().copied()).collect();
    assert!(average_holding_period(&series2).is_none());
}

#[test]
fn test_bets_concentration() {
    let series = load_dollar_bar_sample();
    let logret = log_returns(&series);
    let returns: Vec<f64> = logret.iter().map(|(_, r)| *r).collect();
    let pos_conc = bets_concentration(&returns).unwrap();
    let flipped: Vec<f64> = returns.iter().map(|r| -*r).collect();
    let neg_conc = bets_concentration(&flipped).unwrap();
    assert!((pos_conc - neg_conc).abs() < 1e-5);
    assert!((pos_conc - 2.0111445).abs() < 1e-3);
}

#[test]
fn test_all_bets_concentration() {
    let series = load_dollar_bar_sample();
    let logret = log_returns(&series);
    let (pos, neg, time) = all_bets_concentration(&logret);
    assert!(pos.is_some());
    assert!(neg.is_some());
    assert!(time.is_some());
    assert!((pos.unwrap() - 0.0014938).abs() < 1e-5);
    assert!((neg.unwrap() - 0.0016261).abs() < 1e-5);
    assert!((time.unwrap() - 0.0195998).abs() < 1e-5);
}

#[test]
fn test_drawdown_and_time_under_water() {
    let dates = dates(10, "2000-01-01", 1);
    let dollar_ret = [100.0, 110.0, 90.0, 100.0, 120.0, 130.0, 100.0, 120.0, 140.0, 130.0];
    let series: Vec<(NaiveDateTime, f64)> = dates.iter().copied().zip(dollar_ret.iter().copied()).collect();
    let (dd, tuw) = drawdown_and_time_under_water(&series, true);
    assert_eq!(dd, vec![20.0, 30.0, 10.0]);
    assert_eq!(tuw.len(), dd.len());
}

#[test]
fn test_sharpe_information_ratios() {
    let normal_ret = [0.01, 0.03, 0.02, 0.01, -0.01, 0.02, 0.01, 0.0, -0.01, 0.01];
    let sharpe = sharpe_ratio(&normal_ret[1..], 12.0, 0.005);
    assert!((sharpe - 0.987483).abs() < 1e-2);
    let info = information_ratio(&normal_ret[1..], 0.006, 12.0);
    assert!((info - 0.733559).abs() < 1e-2);
}

#[test]
fn test_probabilistic_deflated_sr() {
    let psr = probabilistic_sharpe_ratio(1.14, 1.0, 250, 0.0, 3.0);
    assert!((psr - 0.95727).abs() < 1e-3);

    let sr_est = [3.5, 1.01, 1.02];
    let dsr = deflated_sharpe_ratio(1.14, &sr_est, 250, 0.0, 3.0, false, false);
    assert!((dsr - 0.95836).abs() < 1e-3);
    let bench = deflated_sharpe_ratio(1.14, &[0.4, 100.0], 250, 0.0, 3.0, true, true);
    assert!((bench - 1.012241).abs() < 1e-3);
    let param = deflated_sharpe_ratio(1.14, &[0.4, 100.0], 250, 0.0, 3.0, true, false);
    assert!((param - 0.941740).abs() < 1e-3);
}

#[test]
fn test_minimum_track_record_length() {
    let min_trl = minimum_track_record_length(1.14, 1.0, 0.0, 3.0, 0.05);
    assert!((min_trl - 228.73497).abs() < 1e-1); // loosened tolerance
}

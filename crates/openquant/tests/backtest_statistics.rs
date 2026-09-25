use chrono::{Duration, NaiveDate, NaiveDateTime};
use csv::ReaderBuilder;
use openquant::backtest_statistics::*;
use serde_json::Value;
use std::path::Path;

/// Expected values from tests/fixtures/backtest_statistics/generate.py (AFML ch. 14 and
/// Bailey & Lopez de Prado, computed in numpy/scipy/pandas independently of this crate).
fn reference() -> Value {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/backtest_statistics/reference.json");
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

fn expected(reference: &Value, pointer: &str) -> f64 {
    reference
        .pointer(pointer)
        .and_then(Value::as_f64)
        .unwrap_or_else(|| panic!("missing {pointer}"))
}

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
    let series: Vec<(NaiveDateTime, f64)> =
        dates.iter().copied().zip(flip_positions.iter().copied()).collect();
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
    let series: Vec<(NaiveDateTime, f64)> =
        dates.iter().copied().zip(hold_positions.iter().copied()).collect();
    // Hand-derived: long 1 from day 1 flips at day 3 (held 2 days, weight 1), short 1 from
    // day 3 is flattened at day 5 (2 days, weight 1), long 2 from day 7 is flattened at day 9
    // (2 days, weight 2). Weighted mean (2*1 + 2*1 + 2*2) / 4 = 2.
    let avg = average_holding_period(&series).unwrap();
    assert!((avg - 2.0).abs() < 1e-12);

    let no_closed = [0.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0];
    let series2: Vec<(NaiveDateTime, f64)> =
        dates.iter().copied().zip(no_closed.iter().copied()).collect();
    assert!(average_holding_period(&series2).is_none());
}

#[test]
fn test_bets_concentration() {
    let series = load_dollar_bar_sample();
    let logret = log_returns(&series);
    let returns: Vec<f64> = logret.iter().map(|(_, r)| *r).collect();
    let reference = reference();
    let pos_conc = bets_concentration(&returns).unwrap();
    let flipped: Vec<f64> = returns.iter().map(|r| -*r).collect();
    let neg_conc = bets_concentration(&flipped).unwrap();
    // Weights r / sum(r) are unchanged by negating every return.
    assert!((pos_conc - neg_conc).abs() < 1e-12);
    assert!((pos_conc - expected(&reference, "/bets_concentration")).abs() < 1e-9);
}

#[test]
fn test_all_bets_concentration() {
    let series = load_dollar_bar_sample();
    let logret = log_returns(&series);
    let reference = reference();
    let (pos, neg, time) = all_bets_concentration(&logret);
    assert!(pos.is_some());
    assert!(neg.is_some());
    assert!(time.is_some());
    let want = |key: &str| expected(&reference, &format!("/all_bets_concentration/{key}"));
    assert!((pos.unwrap() - want("positive")).abs() < 1e-12);
    assert!((neg.unwrap() - want("negative")).abs() < 1e-12);
    assert!((time.unwrap() - want("time")).abs() < 1e-12);
}

#[test]
fn test_drawdown_and_time_under_water() {
    let dates = dates(10, "2000-01-01", 1);
    let dollar_ret = [100.0, 110.0, 90.0, 100.0, 120.0, 130.0, 100.0, 120.0, 140.0, 130.0];
    let series: Vec<(NaiveDateTime, f64)> =
        dates.iter().copied().zip(dollar_ret.iter().copied()).collect();
    // Hand-derived (AFML snippet 14.4): high-water marks 100, 110, 120, 130, 140. The 110
    // mark falls to 90 (20), 120 is never under water, 130 falls to 100 (30), 140 to 130 (10).
    let (dd, tuw) = drawdown_and_time_under_water(&series, true);
    assert_eq!(dd, vec![20.0, 30.0, 10.0]);
    assert_eq!(tuw.len(), dd.len());
}

#[test]
fn test_sharpe_information_ratios() {
    let reference = reference();
    let normal_ret = [0.01, 0.03, 0.02, 0.01, -0.01, 0.02, 0.01, 0.0, -0.01, 0.01];
    let sharpe = sharpe_ratio(&normal_ret[1..], 12.0, 0.005);
    assert!((sharpe - expected(&reference, "/sharpe_ratio")).abs() < 1e-12);
    let info = information_ratio(&normal_ret[1..], 0.006, 12.0);
    assert!((info - expected(&reference, "/information_ratio")).abs() < 1e-12);
}

#[test]
fn test_probabilistic_deflated_sr() {
    let reference = reference();
    let psr = probabilistic_sharpe_ratio(1.14, 1.0, 250, 0.0, 3.0);
    assert!((psr - expected(&reference, "/psr")).abs() < 1e-9);

    let sr_est = [3.5, 1.01, 1.02];
    let dsr = deflated_sharpe_ratio(1.14, &sr_est, 250, 0.0, 3.0, false, false).unwrap();
    assert!((dsr - expected(&reference, "/dsr")).abs() < 1e-9);
    let bench = deflated_sharpe_ratio(1.14, &[0.4, 100.0], 250, 0.0, 3.0, true, true).unwrap();
    assert!((bench - expected(&reference, "/dsr_benchmark_from_params")).abs() < 1e-9);
    let param = deflated_sharpe_ratio(1.14, &[0.4, 100.0], 250, 0.0, 3.0, true, false).unwrap();
    assert!((param - expected(&reference, "/dsr_from_params")).abs() < 1e-9);
}

#[test]
fn test_minimum_track_record_length() {
    let min_trl = minimum_track_record_length(1.14, 1.0, 0.0, 3.0, 0.05).unwrap();
    assert!((min_trl - expected(&reference(), "/min_trl")).abs() < 1e-7);
}

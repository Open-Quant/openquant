use chrono::NaiveDate;
use csv::ReaderBuilder;
use openquant::etf_trick::{get_futures_roll_series, FuturesRollRow};
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct OpenRow {
    date: String,
    spx: f64,
}

#[derive(Debug, Deserialize)]
struct CloseRow {
    date: String,
    spx: f64,
}

fn fixture_dir() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/etf_trick")
}

fn load_rows() -> Vec<FuturesRollRow> {
    let open_path = fixture_dir().join("open_df.csv");
    let close_path = fixture_dir().join("close_df.csv");

    let mut open_rdr = ReaderBuilder::new().has_headers(true).from_path(open_path).unwrap();
    let mut close_rdr = ReaderBuilder::new().has_headers(true).from_path(close_path).unwrap();

    let opens: Vec<OpenRow> = open_rdr.deserialize().map(|r| r.unwrap()).collect();
    let closes: Vec<CloseRow> = close_rdr.deserialize().map(|r| r.unwrap()).collect();
    assert_eq!(opens.len(), closes.len());

    let roll_1 = NaiveDate::from_ymd_opt(2017, 3, 20).unwrap();
    let roll_2 = NaiveDate::from_ymd_opt(2018, 1, 17).unwrap();

    let mut rows = Vec::with_capacity(opens.len());
    for (o, c) in opens.into_iter().zip(closes.into_iter()) {
        assert_eq!(o.date, c.date);
        let date = NaiveDate::parse_from_str(&o.date, "%Y-%m-%d").unwrap();
        let current = if date <= roll_1 {
            "futures_1"
        } else if date <= roll_2 {
            "futures_2"
        } else {
            "futures_3"
        };
        rows.push(FuturesRollRow {
            date,
            open: o.spx,
            close: c.spx,
            security: current.to_string(),
            current_security: current.to_string(),
        });
    }
    rows
}

fn unique_count(v: &[f64]) -> usize {
    let mut uniq: Vec<f64> = Vec::new();
    for &x in v {
        if !uniq.iter().any(|u| (*u - x).abs() < 1e-12) {
            uniq.push(x);
        }
    }
    uniq.len()
}

#[test]
fn test_futures_roll() {
    let rows = load_rows();

    let gaps_diff_no_backward = get_futures_roll_series(&rows, "absolute", false).unwrap();
    let gaps_rel_no_backward = get_futures_roll_series(&rows, "relative", false).unwrap();
    let gaps_diff_with_backward = get_futures_roll_series(&rows, "absolute", true).unwrap();
    let gaps_rel_with_backward = get_futures_roll_series(&rows, "relative", true).unwrap();
    assert!(get_futures_roll_series(&rows, "unknown", true).is_err());

    assert_eq!(unique_count(&gaps_diff_no_backward), 3);
    assert_eq!(unique_count(&gaps_rel_no_backward), 3);

    assert_eq!(gaps_diff_no_backward[0], 0.0);
    assert_eq!(*gaps_diff_no_backward.last().unwrap(), -1.75);
    assert_eq!(gaps_diff_with_backward[0], 1.75);
    assert_eq!(*gaps_diff_with_backward.last().unwrap(), 0.0);

    assert_eq!(gaps_rel_no_backward[0], 1.0);
    assert!((gaps_rel_no_backward.last().unwrap() - 0.999294).abs() < 1e-6);
    assert!((gaps_rel_with_backward[0] - (1.0 / 0.999294)).abs() < 1e-6);
    assert_eq!(*gaps_rel_with_backward.last().unwrap(), 1.0);
}

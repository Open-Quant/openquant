use csv::ReaderBuilder;
use openquant::util::volatility::{get_garman_class_vol, get_parkinson_vol, get_yang_zhang_vol};
use serde::Deserialize;
use serde_json::Value;

#[derive(Debug, Deserialize)]
struct OhlcRow {
    open: f64,
    high: f64,
    low: f64,
    close: f64,
}

fn load_ohlc() -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/backtest_statistics/dollar_bar_sample.csv");
    let mut rdr = ReaderBuilder::new().has_headers(true).from_path(path).unwrap();

    let mut open = Vec::new();
    let mut high = Vec::new();
    let mut low = Vec::new();
    let mut close = Vec::new();

    for result in rdr.deserialize::<OhlcRow>() {
        let row = result.unwrap();
        open.push(row.open);
        high.push(row.high);
        low.push(row.low);
        close.push(row.close);
    }
    (open, high, low, close)
}

/// Expected values from tests/fixtures/volatility/generate_range.py (Parkinson 1980,
/// Garman & Klass 1980, Yang & Zhang 2000, computed in pandas independently of this crate).
fn reference() -> Value {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/volatility/range_reference.json");
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

/// Compare an estimator's output with one entry of the reference: the NaN warm-up count,
/// the mean of the non-NaN values and the sampled positions.
fn assert_matches(actual: &[f64], expected: &Value) {
    let n_nan = actual.iter().filter(|v| v.is_nan()).count() as u64;
    assert_eq!(n_nan, expected["n_nan"].as_u64().unwrap());
    let mean = expected["mean_excluding_nan"].as_f64().unwrap();
    assert!((nanmean(actual) - mean).abs() < 1e-12, "{} vs {mean}", nanmean(actual));
    for sample in expected["samples"].as_array().unwrap() {
        let i = sample["position"].as_u64().unwrap() as usize;
        match sample["value"].as_f64() {
            None => assert!(actual[i].is_nan(), "position {i}: {}", actual[i]),
            Some(v) => assert!((actual[i] - v).abs() < 1e-12, "position {i}: {} vs {v}", actual[i]),
        }
    }
}

fn nanmean(values: &[f64]) -> f64 {
    let mut sum = 0.0;
    let mut n = 0usize;
    for &v in values {
        if !v.is_nan() {
            sum += v;
            n += 1;
        }
    }
    sum / n as f64
}

#[test]
fn test_volatility_estimators_match_reference() {
    let reference = reference();
    let (open, high, low, close) = load_ohlc();
    let gm_vol = get_garman_class_vol(&open, &high, &low, &close, 20).unwrap();
    let yz_vol = get_yang_zhang_vol(&open, &high, &low, &close, 20).unwrap();
    let park_vol = get_parkinson_vol(&high, &low, 20).unwrap();

    assert_eq!(close.len(), gm_vol.len());
    assert_eq!(close.len(), yz_vol.len());
    assert_eq!(close.len(), park_vol.len());

    assert_matches(&gm_vol, &reference["garman_klass"]);
    assert_matches(&yz_vol, &reference["yang_zhang_paper"]);
    assert_matches(&park_vol, &reference["parkinson"]);
}

/// Issue #165: Yang & Zhang (2000) with the open-to-close term ln(C_t/O_t) and demeaned
/// overnight and open-to-close variances, from tests/fixtures/volatility/generate_range.py.
#[test]
fn test_yang_zhang_matches_paper() {
    let (open, high, low, close) = load_ohlc();
    let yz_vol = get_yang_zhang_vol(&open, &high, &low, &close, 20).unwrap();
    assert_matches(&yz_vol, &reference()["yang_zhang_paper"]);
}

/// Yang-Zhang is drift-independent: a series that gaps up 0.2% every night and rises 1% from
/// open to close every day, with the high at the close and the low at the open, has constant
/// overnight and open-to-close returns (zero variance once demeaned) and a zero
/// Rogers-Satchell term, so its volatility is zero.
#[test]
fn test_yang_zhang_is_zero_for_pure_drift() {
    let (mut open, mut close) = (vec![100.0], vec![101.0]);
    for i in 1..40 {
        open.push(close[i - 1] * 1.002);
        close.push(open[i] * 1.01);
    }
    let yz_vol = get_yang_zhang_vol(&open, &close, &open, &close, 10).unwrap();
    assert!(yz_vol[..10].iter().all(|v| v.is_nan()));
    for (i, v) in yz_vol.iter().enumerate().skip(10) {
        assert!(v.abs() < 1e-12, "bar {i}: {v}");
    }
}

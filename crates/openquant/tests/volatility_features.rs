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
    // `yang_zhang` is the library's form of the estimator, which departs from the paper; see
    // generate_range.py and `test_yang_zhang_matches_paper` below.
    assert_matches(&yz_vol, &reference["yang_zhang"]);
    assert_matches(&park_vol, &reference["parkinson"]);
}

#[test]
#[ignore = "FINDING: get_yang_zhang_vol uses ln(C_t/O_{t-1}) for the close term instead of \
            Yang & Zhang's open-to-close ln(C_t/O_t), and undemeaned moments; see \
            tests/fixtures/volatility/generate_range.py"]
fn test_yang_zhang_matches_paper() {
    let (open, high, low, close) = load_ohlc();
    let yz_vol = get_yang_zhang_vol(&open, &high, &low, &close, 20).unwrap();
    assert_matches(&yz_vol, &reference()["yang_zhang_paper"]);
}

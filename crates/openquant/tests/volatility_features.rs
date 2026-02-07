use csv::ReaderBuilder;
use openquant::util::volatility::{get_garman_class_vol, get_parksinson_vol, get_yang_zhang_vol};
use serde::Deserialize;

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
fn test_volatility_estimators_match_mlfinlab_baseline() {
    let (open, high, low, close) = load_ohlc();
    let gm_vol = get_garman_class_vol(&open, &high, &low, &close, 20);
    let yz_vol = get_yang_zhang_vol(&open, &high, &low, &close, 20);
    let park_vol = get_parksinson_vol(&high, &low, 20);

    assert_eq!(close.len(), gm_vol.len());
    assert_eq!(close.len(), yz_vol.len());
    assert_eq!(close.len(), park_vol.len());

    assert!((nanmean(&gm_vol) - 0.001482).abs() < 1e-6);
    assert!((nanmean(&yz_vol) - 0.00162001).abs() < 1e-6);
    assert!((nanmean(&park_vol) - 0.00149997).abs() < 1e-6);
}

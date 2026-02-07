use csv::ReaderBuilder;
use openquant::fracdiff::{frac_diff, frac_diff_ffd, get_weights, get_weights_ffd};

fn load_close_series() -> Vec<f64> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/backtest_statistics/dollar_bar_sample.csv");
    let mut rdr = ReaderBuilder::new().has_headers(true).from_path(path).unwrap();
    let mut close = Vec::new();
    for rec in rdr.records() {
        let rec = rec.unwrap();
        close.push(rec[4].parse::<f64>().unwrap());
    }
    close
}

#[test]
fn test_get_weights() {
    let weights = get_weights(0.9, 100);
    assert_eq!(weights.len(), 100);
    assert_eq!(*weights.last().unwrap(), 1.0);
}

#[test]
fn test_get_weights_ffd() {
    let weights = get_weights_ffd(0.9, 1e-3, 100);
    assert_eq!(weights.len(), 12);
    assert_eq!(*weights.last().unwrap(), 1.0);
}

#[test]
fn test_frac_diff() {
    let data = load_close_series();
    for i in 1..10 {
        let diff_amt = i as f64 / 10.0;
        let fd = frac_diff(&data, diff_amt, 0.01);
        assert_eq!(fd.len(), data.len());
        assert!(fd[0].is_nan());
    }
}

#[test]
fn test_frac_diff_ffd() {
    let data = load_close_series();
    for i in 1..10 {
        let diff_amt = i as f64 / 10.0;
        let fd = frac_diff_ffd(&data, diff_amt, 1e-5);
        assert_eq!(fd.len(), data.len());
        assert!(fd[0].is_nan());
    }
}

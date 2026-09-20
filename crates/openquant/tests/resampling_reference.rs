//! `resample_by = "W"` must mean "use rows 4, 9, 14, ...", for every module that accepts it.
//!
//! hrp and hcaa check the same property in their own reference files. These two modules shared
//! the defect (#93) and had no test that resampled more than one asset.

use std::path::Path;

use csv::ReaderBuilder;
use nalgebra::DMatrix;
use openquant::cla::ReturnsEstimation;
use openquant::portfolio_optimization::{allocate_inverse_variance_with, AllocationOptions};

fn load_prices() -> DMatrix<f64> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/portfolio_optimization/stock_prices.csv");
    let mut rdr = ReaderBuilder::new().has_headers(true).from_path(path).unwrap();
    let cols = rdr.headers().unwrap().len() - 1;
    let mut flat = Vec::new();
    for rec in rdr.records() {
        flat.extend(rec.unwrap().iter().skip(1).map(|x| x.parse::<f64>().unwrap()));
    }
    DMatrix::from_row_slice(flat.len() / cols, cols, &flat)
}

fn every_fifth_row(prices: &DMatrix<f64>) -> DMatrix<f64> {
    let kept: Vec<usize> = (4..prices.nrows()).step_by(5).collect();
    DMatrix::from_fn(kept.len(), prices.ncols(), |r, c| prices[(kept[r], c)])
}

#[test]
fn cla_weekly_returns_equal_returns_of_every_fifth_row() {
    let prices = load_prices();
    let resampled = ReturnsEstimation::calculate_returns(&prices, Some("W")).unwrap();
    let direct = ReturnsEstimation::calculate_returns(&every_fifth_row(&prices), None).unwrap();
    assert_eq!(resampled, direct);
}

#[test]
fn inverse_variance_weekly_weights_equal_weights_on_every_fifth_row() {
    let prices = load_prices();
    let weekly = AllocationOptions { resample_by: Some("W"), ..AllocationOptions::default() };
    let resampled = allocate_inverse_variance_with(&prices, &weekly).unwrap();
    let direct =
        allocate_inverse_variance_with(&every_fifth_row(&prices), &AllocationOptions::default())
            .unwrap();
    // Inverse-variance weights are scale-free, so the two annualisation factors cancel.
    for (a, b) in resampled.weights.iter().zip(&direct.weights) {
        assert!((a - b).abs() < 1e-12, "{a} vs {b}");
    }
}

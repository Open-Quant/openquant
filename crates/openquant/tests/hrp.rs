use csv::ReaderBuilder;
use nalgebra::DMatrix;
use openquant::hrp::{HierarchicalRiskParity, HrpError};
use std::path::Path;

fn load_prices_and_names() -> (DMatrix<f64>, Vec<String>) {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/portfolio_optimization/stock_prices.csv");
    let mut rdr = ReaderBuilder::new().has_headers(true).from_path(path).unwrap();
    let headers = rdr.headers().unwrap().clone();
    let names: Vec<String> = headers.iter().skip(1).map(|s| s.to_string()).collect();
    let mut rows: Vec<Vec<f64>> = Vec::new();
    for rec in rdr.records() {
        let row = rec.unwrap();
        rows.push(row.iter().skip(1).map(|x| x.parse::<f64>().unwrap()).collect::<Vec<f64>>());
    }
    let nrows = rows.len();
    let ncols = rows[0].len();
    let flat = rows.into_iter().flat_map(|r| r.into_iter()).collect::<Vec<f64>>();
    (DMatrix::from_vec(nrows, ncols, flat), names)
}

fn returns_from_prices(prices: &DMatrix<f64>) -> DMatrix<f64> {
    let mut out = DMatrix::zeros(prices.nrows() - 1, prices.ncols());
    for r in 1..prices.nrows() {
        for c in 0..prices.ncols() {
            out[(r - 1, c)] = prices[(r, c)] / prices[(r - 1, c)] - 1.0;
        }
    }
    out
}

fn covariance(returns: &DMatrix<f64>) -> DMatrix<f64> {
    let rows = returns.nrows();
    let cols = returns.ncols();
    let means: Vec<f64> = (0..cols).map(|c| returns.column(c).sum() / rows as f64).collect();
    let mut cov = DMatrix::zeros(cols, cols);
    for i in 0..cols {
        for j in i..cols {
            let mut s = 0.0;
            for r in 0..rows {
                s += (returns[(r, i)] - means[i]) * (returns[(r, j)] - means[j]);
            }
            s /= (rows - 1) as f64;
            cov[(i, j)] = s;
            cov[(j, i)] = s;
        }
    }
    cov
}

fn assert_weights(weights: &[f64], n_assets: usize) {
    assert_eq!(weights.len(), n_assets);
    assert!(weights.iter().all(|w| *w >= 0.0));
    let sum: f64 = weights.iter().sum();
    assert!((sum - 1.0).abs() < 1e-6, "sum={sum}");
}

#[test]
fn test_hrp() {
    let (prices, names) = load_prices_and_names();
    let mut hrp = HierarchicalRiskParity::new();
    hrp.allocate(&names, Some(&prices), None, None, None, false).unwrap();
    assert_weights(&hrp.weights, names.len());
}

#[test]
fn test_hrp_with_shrinkage() {
    let (prices, names) = load_prices_and_names();
    let mut hrp = HierarchicalRiskParity::new();
    hrp.allocate(&names, Some(&prices), None, None, None, true).unwrap();
    assert_weights(&hrp.weights, names.len());
}

#[test]
fn test_dendrogram_plot() {
    let (prices, names) = load_prices_and_names();
    let mut hrp = HierarchicalRiskParity::new();
    hrp.allocate(&names, Some(&prices), None, None, None, true).unwrap();
    let dendrogram = hrp.plot_clusters(&names).unwrap();
    assert!(!dendrogram.icoord.is_empty());
    assert!(!dendrogram.dcoord.is_empty());
    assert!(!dendrogram.ivl.is_empty());
    assert!(!dendrogram.leaves.is_empty());
    assert!(!dendrogram.color_list.is_empty());
}

#[test]
fn test_quasi_diagonalization() {
    let (prices, names) = load_prices_and_names();
    let mut hrp = HierarchicalRiskParity::new();
    hrp.allocate(&names, Some(&prices), None, None, None, false).unwrap();
    assert_eq!(
        hrp.ordered_indices,
        vec![13, 9, 10, 8, 14, 7, 1, 6, 4, 16, 3, 17, 12, 18, 22, 0, 15, 21, 11, 2, 20, 5, 19]
    );
}

#[test]
fn test_resampling_asset_prices() {
    let (prices, names) = load_prices_and_names();
    let mut hrp = HierarchicalRiskParity::new();
    hrp.allocate(&names, Some(&prices), None, None, Some("B"), false).unwrap();
    assert_weights(&hrp.weights, names.len());
}

#[test]
fn test_all_inputs_none() {
    let (_, names) = load_prices_and_names();
    let mut hrp = HierarchicalRiskParity::new();
    let err = hrp.allocate(&names, None, None, None, None, false).unwrap_err();
    assert_eq!(err, HrpError::NoData);
}

#[test]
fn test_hrp_with_input_as_returns() {
    let (prices, names) = load_prices_and_names();
    let returns = returns_from_prices(&prices);
    let mut hrp = HierarchicalRiskParity::new();
    hrp.allocate(&names, None, Some(&returns), None, None, false).unwrap();
    assert_weights(&hrp.weights, names.len());
}

#[test]
fn test_hrp_with_input_as_covariance_matrix() {
    let (prices, names) = load_prices_and_names();
    let returns = returns_from_prices(&prices);
    let cov = covariance(&returns);
    let mut hrp = HierarchicalRiskParity::new();
    hrp.allocate(&names, None, None, Some(&cov), None, false).unwrap();
    assert_weights(&hrp.weights, names.len());
}

#[test]
fn test_value_error_for_incorrect_dimensions() {
    let (prices, names) = load_prices_and_names();
    let mut hrp = HierarchicalRiskParity::new();
    let bad_names = names[0..(names.len() - 1)].to_vec();
    let err = hrp.allocate(&bad_names, Some(&prices), None, None, None, false).unwrap_err();
    assert!(matches!(err, HrpError::DimensionMismatch(_)));
}

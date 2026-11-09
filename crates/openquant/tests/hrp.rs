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
    (DMatrix::from_row_slice(nrows, ncols, &flat), names)
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
    assert_eq!(hrp.ordered_indices, reference_leaf_order());
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

/// HRP weights do not depend on the scale of the covariance. The bisection split used to add
/// `f64::EPSILON` to the sum of the two cluster variances, which dominates when variances are
/// tiny: `diag(1e-18, 4e-18)` gave about `[0.9955, 0.0045]` instead of `[0.8, 0.2]`.
#[test]
fn test_weights_invariant_to_covariance_scale() {
    let names: Vec<String> = ["a", "b"].map(String::from).to_vec();
    for scale in [1.0, 1e-4, 1e-12, 1e-18, 1e-300] {
        let cov = DMatrix::from_row_slice(2, 2, &[scale, 0.0, 0.0, 4.0 * scale]);
        let mut hrp = HierarchicalRiskParity::new();
        hrp.allocate(&names, None, None, Some(&cov), None, false).unwrap();
        assert!((hrp.weights[0] - 0.8).abs() < 1e-12, "scale {scale}: {:?}", hrp.weights);
        assert!((hrp.weights[1] - 0.2).abs() < 1e-12, "scale {scale}: {:?}", hrp.weights);
    }
}

/// A `NaN` in the covariance reaches the weights instead of being read as zero risk. The
/// cluster variance was floored with `f64::max(v, 0.0)`, which returns 0 for a `NaN` `v`, so the
/// pair with the unknown covariance looked riskless and took nearly all the weight.
#[test]
fn test_nan_covariance_propagates_to_weights() {
    let names: Vec<String> = ["a", "b", "c"].map(String::from).to_vec();
    let nan = f64::NAN;
    let cov = DMatrix::from_row_slice(3, 3, &[0.04, nan, 0.0, nan, 0.04, 0.0, 0.0, 0.0, 0.04]);
    let mut hrp = HierarchicalRiskParity::new();
    hrp.allocate(&names, None, None, Some(&cov), None, false).unwrap();
    assert!(hrp.weights.iter().all(|w| w.is_nan()), "{:?}", hrp.weights);
}

/// A constant-return asset has zero variance, which `allocate` rejects. The sample covariance
/// used to compute each column's mean as `sum / rows`, which rounds for a constant `0.1`
/// (`0.30000000000000004 / 3`), leaving a variance of about `1e-34`: the asset passed the
/// positive-variance check and took practically all the weight.
#[test]
fn test_constant_return_asset_is_rejected() {
    let names: Vec<String> = ["flat", "risky"].map(String::from).to_vec();
    let returns = DMatrix::from_row_slice(3, 2, &[0.1, 0.01, 0.1, -0.02, 0.1, 0.03]);
    let mut hrp = HierarchicalRiskParity::new();
    let result = hrp.allocate(&names, None, Some(&returns), None, None, false);
    assert_eq!(result, Err(HrpError::NoData), "weights {:?}", hrp.weights);
}

/// Leaf order of scipy's single-linkage dendrogram on the same prices, clustered on the distance
/// of distances as Snippet 16.4 does (tests/fixtures/hrp/generate.py).
fn reference_leaf_order() -> Vec<usize> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/hrp/reference.json");
    let reference: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    reference["stock_prices"]["distance_of_distances"]["order"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect()
}

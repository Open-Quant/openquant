use csv::ReaderBuilder;
use nalgebra::DMatrix;
use openquant::hcaa::{HcaaError, HierarchicalClusteringAssetAllocation};
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
    let flat: Vec<f64> = rows.into_iter().flat_map(|r| r.into_iter()).collect();
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

fn assert_basic_weights(weights: &[f64], n_assets: usize) {
    assert_eq!(weights.len(), n_assets);
    assert!(weights.iter().all(|w| *w >= 0.0));
    let sum: f64 = weights.iter().sum();
    assert!((sum - 1.0).abs() < 1e-6, "sum={sum}");
}

#[test]
fn test_hcaa_equal_weight() {
    let (prices, names) = load_prices_and_names();
    let mut hcaa = HierarchicalClusteringAssetAllocation::default();
    hcaa.allocate(&names, Some(&prices), None, None, None, "equal_weighting", 0.05, Some(5), None)
        .unwrap();
    assert_basic_weights(&hcaa.weights, names.len());
}

#[test]
fn test_hcaa_min_variance() {
    let (prices, names) = load_prices_and_names();
    let mut hcaa = HierarchicalClusteringAssetAllocation::default();
    hcaa.allocate(&names, Some(&prices), None, None, None, "minimum_variance", 0.05, Some(5), None)
        .unwrap();
    assert_basic_weights(&hcaa.weights, names.len());
}

#[test]
fn test_hcaa_min_standard_deviation() {
    let (prices, names) = load_prices_and_names();
    let mut hcaa = HierarchicalClusteringAssetAllocation::default();
    hcaa.allocate(
        &names,
        Some(&prices),
        None,
        None,
        None,
        "minimum_standard_deviation",
        0.05,
        Some(5),
        None,
    )
    .unwrap();
    assert_basic_weights(&hcaa.weights, names.len());
}

#[test]
fn test_hcaa_sharpe_ratio_mean() {
    let (prices, names) = load_prices_and_names();
    let mut hcaa = HierarchicalClusteringAssetAllocation::new("mean");
    hcaa.allocate(&names, Some(&prices), None, None, None, "sharpe_ratio", 0.05, Some(5), None)
        .unwrap();
    assert_basic_weights(&hcaa.weights, names.len());
}

#[test]
fn test_hcaa_sharpe_ratio_exponential() {
    let (prices, names) = load_prices_and_names();
    let mut hcaa = HierarchicalClusteringAssetAllocation::new("exponential");
    hcaa.allocate(&names, Some(&prices), None, None, None, "sharpe_ratio", 0.05, Some(5), None)
        .unwrap();
    assert_basic_weights(&hcaa.weights, names.len());
}

#[test]
fn test_value_error_for_unknown_returns() {
    let (prices, names) = load_prices_and_names();
    let mut hcaa = HierarchicalClusteringAssetAllocation::new("unknown_returns");
    let err = hcaa
        .allocate(&names, Some(&prices), None, None, None, "sharpe_ratio", 0.05, Some(5), None)
        .unwrap_err();
    assert!(matches!(err, HcaaError::UnknownReturns(_)));
}

#[test]
fn test_value_error_for_sharpe_ratio_without_prices_or_expected() {
    let (prices, names) = load_prices_and_names();
    let returns = returns_from_prices(&prices);
    let mut hcaa = HierarchicalClusteringAssetAllocation::default();
    let err = hcaa
        .allocate(&names, None, Some(&returns), None, None, "sharpe_ratio", 0.05, Some(5), None)
        .unwrap_err();
    assert_eq!(err, HcaaError::MissingExpectedReturnsForSharpe);
}

#[test]
fn test_hcaa_expected_shortfall() {
    let (prices, names) = load_prices_and_names();
    let mut hcaa = HierarchicalClusteringAssetAllocation::default();
    hcaa.allocate(
        &names,
        Some(&prices),
        None,
        None,
        None,
        "expected_shortfall",
        0.05,
        Some(5),
        None,
    )
    .unwrap();
    assert_basic_weights(&hcaa.weights, names.len());
}

#[test]
fn test_hcaa_conditional_drawdown_risk() {
    let (prices, names) = load_prices_and_names();
    let mut hcaa = HierarchicalClusteringAssetAllocation::default();
    hcaa.allocate(
        &names,
        Some(&prices),
        None,
        None,
        None,
        "conditional_drawdown_risk",
        0.05,
        Some(5),
        None,
    )
    .unwrap();
    assert_basic_weights(&hcaa.weights, names.len());
}

#[test]
fn test_quasi_diagonalization() {
    let (prices, names) = load_prices_and_names();
    let mut hcaa = HierarchicalClusteringAssetAllocation::default();
    hcaa.allocate(&names, Some(&prices), None, None, None, "equal_weighting", 0.05, Some(5), None)
        .unwrap();
    assert_eq!(
        hcaa.ordered_indices,
        vec![13, 9, 10, 8, 14, 7, 1, 6, 4, 16, 3, 17, 12, 18, 22, 0, 15, 21, 11, 2, 20, 5, 19]
    );
}

#[test]
fn test_all_inputs_none() {
    let (_, names) = load_prices_and_names();
    let mut hcaa = HierarchicalClusteringAssetAllocation::default();
    let err = hcaa
        .allocate(&names, None, None, None, None, "equal_weighting", 0.05, Some(5), None)
        .unwrap_err();
    assert_eq!(err, HcaaError::NoData);
}

#[test]
fn test_hcaa_with_input_as_returns() {
    let (prices, names) = load_prices_and_names();
    let returns = returns_from_prices(&prices);
    let mut hcaa = HierarchicalClusteringAssetAllocation::default();
    hcaa.allocate(&names, None, Some(&returns), None, None, "equal_weighting", 0.05, None, None)
        .unwrap();
    assert_basic_weights(&hcaa.weights, names.len());
}

#[test]
fn test_hcaa_with_input_as_covariance_matrix() {
    let (prices, names) = load_prices_and_names();
    let returns = returns_from_prices(&prices);
    let cov = covariance(&returns);
    let mut hcaa = HierarchicalClusteringAssetAllocation::default();
    hcaa.allocate(
        &names,
        None,
        Some(&returns),
        Some(&cov),
        None,
        "equal_weighting",
        0.05,
        Some(6),
        None,
    )
    .unwrap();
    assert_basic_weights(&hcaa.weights, names.len());
}

#[test]
fn test_value_error_for_allocation_metric() {
    let (prices, names) = load_prices_and_names();
    let mut hcaa = HierarchicalClusteringAssetAllocation::default();
    let err = hcaa
        .allocate(&names, Some(&prices), None, None, None, "random_metric", 0.05, Some(5), None)
        .unwrap_err();
    assert!(matches!(err, HcaaError::UnknownAllocationMetric(_)));
}

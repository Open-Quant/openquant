use csv::ReaderBuilder;
use nalgebra::DMatrix;
use openquant::portfolio_optimization::{
    allocate_efficient_risk, allocate_from_inputs, allocate_inverse_variance, allocate_max_sharpe, allocate_min_vol,
    allocate_with_solution, compute_expected_and_covariance, returns_method_from_str, AllocationOptions, AllocError,
    ReturnsMethod,
};
use serde_json::Value;
use std::collections::HashMap;
use std::path::Path;

fn load_prices() -> DMatrix<f64> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/portfolio_optimization/stock_prices.csv");
    let mut rdr = ReaderBuilder::new().has_headers(true).from_path(path).unwrap();
    let mut data: Vec<Vec<f64>> = Vec::new();
    for result in rdr.records() {
        let rec = result.unwrap();
        let mut row = Vec::new();
        for field in rec.iter().skip(1) {
            row.push(field.parse::<f64>().unwrap());
        }
        data.push(row);
    }
    let rows = data.len();
    let cols = data[0].len();
    let flat: Vec<f64> = data.into_iter().flat_map(|r| r.into_iter()).collect();
    DMatrix::from_vec(rows, cols, flat)
}

fn load_fixture() -> Value {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/portfolio_optimization/mean_variance_fixture.json");
    let file = std::fs::File::open(path).unwrap();
    serde_json::from_reader(file).unwrap()
}

#[test]
fn test_inverse_variance_weights() {
    let prices = load_prices();
    let res = allocate_inverse_variance(&prices).expect("solution");
    assert_eq!(res.weights.len(), prices.ncols());
    let sum: f64 = res.weights.iter().sum();
    assert!((sum - 1.0).abs() < 1e-6);
    assert!(res.weights.iter().all(|w| *w >= 0.0));
}

#[test]
fn test_min_volatility_basic() {
    let prices = load_prices();
    let res = allocate_min_vol(&prices, None, None).expect("solution");
    let sum: f64 = res.weights.iter().sum();
    assert!((sum - 1.0).abs() < 1e-3); // allow small drift from gradient loop
    assert!(res.weights.iter().all(|w| *w >= -1e-6));
}

#[test]
fn test_against_python_fixture_weights() {
    let prices = load_prices();
    let fixture = load_fixture();
    let weights = fixture["weights"]["inverse_variance"].as_array().unwrap();
    let res = allocate_inverse_variance(&prices).unwrap();
    let max_diff = res
        .weights
        .iter()
        .zip(weights.iter())
        .map(|(r, e)| (r - e.as_f64().unwrap()).abs())
        .fold(0.0_f64, f64::max);
    assert!(max_diff < 1.0, "inverse variance max diff {}", max_diff);

    let w_min = fixture["weights"]["min_volatility"].as_array().unwrap();
    let res_min = allocate_min_vol(&prices, None, None).unwrap();
    let max_diff = res_min
        .weights
        .iter()
        .zip(w_min.iter())
        .map(|(r, e)| (r - e.as_f64().unwrap()).abs())
        .fold(0.0_f64, f64::max);
    assert!(max_diff < 1.0, "min vol bound diff {}", max_diff);

    let w_max = fixture["weights"]["max_sharpe"].as_array().unwrap();
    let res_max = allocate_max_sharpe(&prices, 0.0, None, None).unwrap();
    let max_diff = res_max
        .weights
        .iter()
        .zip(w_max.iter())
        .map(|(r, e)| (r - e.as_f64().unwrap()).abs())
        .fold(0.0_f64, f64::max);
    assert!(max_diff < 1.0, "max sharpe diff {}", max_diff);
}

#[test]
fn test_max_sharpe_basic() {
    let prices = load_prices();
    let res = allocate_max_sharpe(&prices, 0.0, None, None).expect("solution");
    let sum: f64 = res.weights.iter().sum();
    assert!((sum - 1.0).abs() < 1e-2);
    assert!(res.weights.iter().all(|w| *w >= -1e-6));
}

#[test]
fn test_efficient_risk_basic() {
    let prices = load_prices();
    let res = allocate_efficient_risk(&prices, 0.001, None, None).expect("solution");
    let sum: f64 = res.weights.iter().sum();
    assert!((sum - 1.0).abs() < 1e-2);
    assert!(res.weights.iter().all(|w| *w >= -1e-6));
}

#[test]
fn test_specific_weight_bounds() {
    let prices = load_prices();
    let mut bounds = HashMap::new();
    bounds.insert(0, (0.3, 1.0));
    let opts = AllocationOptions {
        bounds: Some(bounds),
        ..Default::default()
    };
    let res = allocate_min_vol_with_opts(&prices, &opts).expect("solution");
    assert!(res.weights[0] >= 0.3 - 1e-6);
    let sum: f64 = res.weights.iter().sum();
    assert!((sum - 1.0).abs() < 1e-3);
}

#[test]
fn test_infeasible_bounds_error() {
    let prices = load_prices();
    let err = allocate_min_vol(&prices, None, Some((0.9, 1.0))).unwrap_err();
    assert!(matches!(err, AllocError::InfeasibleBounds { .. }));
    let err = allocate_max_sharpe(&prices, 0.0, None, Some((0.9, 1.0))).unwrap_err();
    assert!(matches!(err, AllocError::InfeasibleBounds { .. }));
}

#[test]
fn test_unknown_solution_string() {
    let prices = load_prices();
    let err = allocate_with_solution(&prices, "ivp", &AllocationOptions::default()).unwrap_err();
    assert!(matches!(err, AllocError::UnknownSolution(_)));
}

#[test]
fn test_unknown_returns_method() {
    assert!(returns_method_from_str("unknown").is_err());
}

#[test]
fn test_allocation_with_supplied_inputs() {
    let prices = load_prices();
    let (expected, cov) = compute_expected_and_covariance(&prices, ReturnsMethod::Mean, None).unwrap();
    let res = allocate_from_inputs(&expected, &cov, "inverse_variance", &AllocationOptions::default()).unwrap();
    assert_eq!(res.weights.len(), prices.ncols());
    assert!((res.weights.iter().sum::<f64>() - 1.0).abs() < 1e-6);
}

#[test]
fn test_bound_and_infeasible_behavior_against_fixture() {
    let prices = load_prices();
    let fixture = load_fixture();
    let mut bounds = HashMap::new();
    bounds.insert(0, (0.3, 1.0));
    let opts = AllocationOptions {
        bounds: Some(bounds),
        ..Default::default()
    };
    let res_min = openquant::portfolio_optimization::allocate_min_vol_with(&prices, &opts).unwrap();
    let exp_min = fixture["weights"]["min_volatility_bound0"].as_array().unwrap();
    let max_diff = res_min
        .weights
        .iter()
        .zip(exp_min.iter())
        .map(|(r, e)| (r - e.as_f64().unwrap()).abs())
        .fold(0.0_f64, f64::max);
    assert!(max_diff < 0.25, "min vol bound diff {}", max_diff);

    let res_max = openquant::portfolio_optimization::allocate_max_sharpe_with(&prices, &opts).unwrap();
    let exp_max = fixture["weights"]["max_sharpe_bound0"].as_array().unwrap();
    let max_diff = res_max
        .weights
        .iter()
        .zip(exp_max.iter())
        .map(|(r, e)| (r - e.as_f64().unwrap()).abs())
        .fold(0.0_f64, f64::max);
    assert!(max_diff < 1.0, "max sharpe bound diff {}", max_diff);

    let res_eff = openquant::portfolio_optimization::allocate_efficient_risk_with(&prices, &AllocationOptions { target_return: 0.01, ..opts.clone() }).unwrap();
    let exp_eff = fixture["weights"]["efficient_risk_bound0"].as_array().unwrap();
    let max_diff = res_eff
        .weights
        .iter()
        .zip(exp_eff.iter())
        .map(|(r, e)| (r - e.as_f64().unwrap()).abs())
        .fold(0.0_f64, f64::max);
    assert!(max_diff < 1.0, "efficient risk bound diff {}", max_diff);

    let err = allocate_min_vol(&prices, None, Some((0.9, 1.0))).unwrap_err();
    assert!(matches!(err, AllocError::InfeasibleBounds { .. }));
}

#[test]
fn test_exponential_returns_method() {
    let prices = load_prices();
    let opts = AllocationOptions {
        returns_method: ReturnsMethod::Exponential { span: 50 },
        ..Default::default()
    };
    let res = allocate_inverse_variance_with_opts(&prices, &opts).unwrap();
    assert_eq!(res.weights.len(), prices.ncols());
    assert!((res.weights.iter().sum::<f64>() - 1.0).abs() < 1e-6);
}

#[test]
fn test_resample_weekly() {
    let prices = load_prices();
    let opts = AllocationOptions {
        resample_by: Some("W"),
        target_return: 0.001,
        ..Default::default()
    };
    let res = allocate_efficient_risk_with_opts(&prices, &opts).unwrap();
    assert!((res.weights.iter().sum::<f64>() - 1.0).abs() < 1e-2);
}

// Helper wrappers to keep test calls succinct
fn allocate_inverse_variance_with_opts(prices: &DMatrix<f64>, opts: &AllocationOptions) -> Result<openquant::portfolio_optimization::MeanVariance, AllocError> {
    openquant::portfolio_optimization::allocate_inverse_variance_with(prices, opts)
}

fn allocate_min_vol_with_opts(prices: &DMatrix<f64>, opts: &AllocationOptions) -> Result<openquant::portfolio_optimization::MeanVariance, AllocError> {
    openquant::portfolio_optimization::allocate_min_vol_with(prices, opts)
}

fn allocate_efficient_risk_with_opts(prices: &DMatrix<f64>, opts: &AllocationOptions) -> Result<openquant::portfolio_optimization::MeanVariance, AllocError> {
    openquant::portfolio_optimization::allocate_efficient_risk_with(prices, opts)
}

use csv::ReaderBuilder;
use nalgebra::DMatrix;
use openquant::portfolio_optimization::{
    allocate_efficient_risk, allocate_from_inputs, allocate_inverse_variance, allocate_max_sharpe,
    allocate_min_vol, allocate_with_solution, compute_expected_and_covariance,
    returns_method_from_str, AllocError, AllocationOptions, ReturnsMethod,
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
    DMatrix::from_row_slice(rows, cols, &flat)
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
    // Sensitive to matrix orientation: 5.6e-4 with the prices read correctly, 0.22 when the
    // loader interleaved rows and columns (#74). The remaining comparisons below still use
    // tolerances too loose to mean anything; min-vol is 0.17 from the reference (#76).
    assert!(max_diff < 1e-3, "inverse variance max diff {max_diff}");

    let w_min = fixture["weights"]["min_volatility"].as_array().unwrap();
    let res_min = allocate_min_vol(&prices, None, None).unwrap();
    let max_diff = res_min
        .weights
        .iter()
        .zip(w_min.iter())
        .map(|(r, e)| (r - e.as_f64().unwrap()).abs())
        .fold(0.0_f64, f64::max);
    // 0.0028 now that the long-only problem is actually solved (it was 0.17 when the
    // unconstrained optimum was clamped). What is left is the returns convention: log returns
    // here, simple returns in the fixture, the same gap inverse-variance shows above.
    assert!(max_diff < 5e-3, "min vol diff {max_diff}");

    // The fixture's max_sharpe weights are not compared: they depend on a risk-free rate and an
    // annualisation this file does not know, and the old `< 1.0` tolerance could not fail anyway.
    // The optimisers are checked against scipy on fixed inputs in portfolio_qp_reference.rs.
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
    let opts = AllocationOptions { bounds: Some(bounds), ..Default::default() };
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
    let (expected, cov) =
        compute_expected_and_covariance(&prices, ReturnsMethod::Mean, None).unwrap();
    let res =
        allocate_from_inputs(&expected, &cov, "inverse_variance", &AllocationOptions::default())
            .unwrap();
    assert_eq!(res.weights.len(), prices.ncols());
    assert!((res.weights.iter().sum::<f64>() - 1.0).abs() < 1e-6);
}

#[test]
fn test_bound_and_infeasible_behavior() {
    // This used to compare against the fixture's `*_bound0` weights at tolerances of 0.25 and
    // 1.0. Those weights were generated under a different constraint (all 23 are non-zero and
    // capped at 0.3), so the comparison was never like for like. What is checked here is the
    // bound this test applies; bounded optima are checked against scipy in
    // portfolio_qp_reference.rs.
    let prices = load_prices();
    let mut bounds = HashMap::new();
    bounds.insert(0, (0.3, 1.0));
    let opts = AllocationOptions { bounds: Some(bounds), ..Default::default() };

    let unbounded = allocate_min_vol(&prices, None, None).unwrap();
    assert!(unbounded.weights[0] < 0.3, "the floor must bind for this test to mean anything");

    let solutions = [
        openquant::portfolio_optimization::allocate_min_vol_with(&prices, &opts).unwrap(),
        openquant::portfolio_optimization::allocate_max_sharpe_with(&prices, &opts).unwrap(),
        openquant::portfolio_optimization::allocate_efficient_risk_with(
            &prices,
            &AllocationOptions { target_return: 0.0, ..opts.clone() },
        )
        .unwrap(),
    ];
    for res in &solutions {
        assert!(res.weights[0] >= 0.3 - 1e-9, "floor violated: {}", res.weights[0]);
        assert!((res.weights.iter().sum::<f64>() - 1.0).abs() < 1e-9);
        assert!(res.weights.iter().all(|w| *w >= -1e-9 && *w <= 1.0 + 1e-9));
    }
    // A floor that binds is met exactly by the minimum-variance portfolio, and costs variance.
    assert!((solutions[0].weights[0] - 0.3).abs() < 1e-9);
    assert!(solutions[0].portfolio_risk > unbounded.portfolio_risk);

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
    let opts =
        AllocationOptions { resample_by: Some("W"), target_return: 0.001, ..Default::default() };
    let res = allocate_efficient_risk_with_opts(&prices, &opts).unwrap();
    assert!((res.weights.iter().sum::<f64>() - 1.0).abs() < 1e-2);
}

// Helper wrappers to keep test calls succinct
fn allocate_inverse_variance_with_opts(
    prices: &DMatrix<f64>,
    opts: &AllocationOptions,
) -> Result<openquant::portfolio_optimization::MeanVariance, AllocError> {
    openquant::portfolio_optimization::allocate_inverse_variance_with(prices, opts)
}

fn allocate_min_vol_with_opts(
    prices: &DMatrix<f64>,
    opts: &AllocationOptions,
) -> Result<openquant::portfolio_optimization::MeanVariance, AllocError> {
    openquant::portfolio_optimization::allocate_min_vol_with(prices, opts)
}

fn allocate_efficient_risk_with_opts(
    prices: &DMatrix<f64>,
    opts: &AllocationOptions,
) -> Result<openquant::portfolio_optimization::MeanVariance, AllocError> {
    openquant::portfolio_optimization::allocate_efficient_risk_with(prices, opts)
}

/// `efficient_risk` used to fail with "no portfolio satisfies the constraints" for feasible
/// targets whenever returns were in decimal units: the return constraint's coefficients were
/// twenty times smaller than the budget row's, ADMM stalled, and hitting the iteration cap was
/// reported as infeasibility. The same problem in percent units solved. Rows are now scaled.
#[test]
fn efficient_risk_does_not_depend_on_the_units_of_the_return_constraint() {
    use nalgebra::DMatrix;
    use openquant::portfolio_optimization::{allocate_from_inputs, AllocationOptions};

    let mu = [0.03, 0.07, 0.09, 0.04];
    let vol = [0.05, 0.16, 0.22, 0.15];
    let rho =
        [[1.0, 0.1, 0.1, 0.1], [0.1, 1.0, 0.8, 0.0], [0.1, 0.8, 1.0, 0.0], [0.1, 0.0, 0.0, 1.0]];
    let cov = DMatrix::from_fn(4, 4, |i, j| rho[i][j] * vol[i] * vol[j]);

    for target in [0.04, 0.05, 0.06, 0.07, 0.08] {
        let opts = AllocationOptions { target_return: target, ..AllocationOptions::default() };
        let decimal = allocate_from_inputs(&mu, &cov, "efficient_risk", &opts)
            .unwrap_or_else(|e| panic!("target {target}: {e}"));
        // The return constraint binds above the minimum-variance return of about 3.3%.
        assert!((decimal.portfolio_return - target).abs() < 1e-7, "target {target}");
        assert!((decimal.weights.iter().sum::<f64>() - 1.0).abs() < 1e-9);
        assert!(decimal.weights.iter().all(|w| *w > -1e-9));

        // The same problem stated in percent must give the same weights.
        let mu_pct: Vec<f64> = mu.iter().map(|m| m * 100.0).collect();
        let cov_pct = &cov * 1e4;
        let opts_pct =
            AllocationOptions { target_return: target * 100.0, ..AllocationOptions::default() };
        let percent = allocate_from_inputs(&mu_pct, &cov_pct, "efficient_risk", &opts_pct).unwrap();
        for (a, b) in decimal.weights.iter().zip(&percent.weights) {
            assert!((a - b).abs() < 1e-6, "target {target}: {a} vs {b}");
        }
    }
}

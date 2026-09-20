//! The mean-variance optimisers against scipy on a fixed (mu, C).
//!
//! `tests/fixtures/portfolio_optimization/generate_qp_reference.py` produced the reference. It
//! takes expected returns and covariance as given, so no returns convention is involved, and it
//! maximises the Sharpe ratio directly rather than through the substitution the library uses.
use nalgebra::DMatrix;
use openquant::portfolio_optimization::{allocate_from_inputs, AllocationOptions};
use serde_json::Value;
use std::collections::HashMap;
use std::path::Path;

fn read(name: &str) -> Value {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/portfolio_optimization")
        .join(name);
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

fn floats(value: &Value) -> Vec<f64> {
    value
        .as_array()
        .unwrap()
        .iter()
        .flat_map(|v| match v {
            Value::Array(inner) => inner.iter().map(|x| x.as_f64().unwrap()).collect::<Vec<_>>(),
            other => vec![other.as_f64().unwrap()],
        })
        .collect()
}

fn inputs() -> (Vec<f64>, DMatrix<f64>) {
    let fixture = read("mean_variance_fixture.json");
    let mu = floats(&fixture["expected_returns_weekly"]);
    let n = mu.len();
    (mu, DMatrix::from_row_slice(n, n, &floats(&fixture["covariance_weekly"])))
}

fn options(bounds: &str) -> AllocationOptions<'static> {
    let mut opts = AllocationOptions::default();
    match bounds {
        "long_only" => {}
        "asset0>=0.3" => opts.bounds = Some(HashMap::from([(0, (0.3, 1.0))])),
        "all in [0.01, 0.15]" => opts.tuple_bounds = Some((0.01, 0.15)),
        other => panic!("unknown bounds {other}"),
    }
    opts
}

fn variance(w: &[f64], cov: &DMatrix<f64>) -> f64 {
    let n = w.len();
    (0..n).map(|i| (0..n).map(|j| w[i] * cov[(i, j)] * w[j]).sum::<f64>()).sum()
}

/// scipy's SLSQP is good to about 1e-7 in the weights here; the library polishes its answer on
/// the active set and is tighter than that. 1e-5 is comfortably between a solver disagreement
/// (the previous implementation was out by 0.2 to 0.7) and SLSQP's own noise.
const WEIGHT_TOLERANCE: f64 = 1e-5;

fn check(case: &str, solution: &str) {
    let (mu, cov) = inputs();
    let reference = &read("qp_reference.json")["cases"][case];
    let mut opts = options(reference["bounds"].as_str().unwrap());
    if let Some(target) = reference["target_return"].as_f64() {
        opts.target_return = target;
    }
    if let Some(rf) = reference["risk_free"].as_f64() {
        opts.risk_free_rate = rf;
    }

    let got = allocate_from_inputs(&mu, &cov, solution, &opts).unwrap().weights;
    let want = floats(&reference["weights"]);

    assert!((got.iter().sum::<f64>() - 1.0).abs() < 1e-9, "{case}: weights do not sum to 1");
    for (i, (g, w)) in got.iter().zip(&want).enumerate() {
        assert!((g - w).abs() < WEIGHT_TOLERANCE, "{case}: asset {i} got {g}, scipy {w}");
    }

    // Agreement is not enough: the library's answer must be at least as good as scipy's.
    let (got_var, want_var) = (variance(&got, &cov), reference["variance"].as_f64().unwrap());
    if solution == "max_sharpe" {
        let sharpe =
            |w: &[f64], var: f64| w.iter().zip(&mu).map(|(a, b)| a * b).sum::<f64>() / var.sqrt();
        assert!(
            sharpe(&got, got_var) >= sharpe(&want, want_var) * (1.0 - 1e-9),
            "{case}: worse Sharpe"
        );
    } else {
        assert!(
            got_var <= want_var * (1.0 + 1e-9),
            "{case}: variance {got_var} > scipy {want_var}"
        );
    }
}

#[test]
fn min_vol_long_only() {
    check("min_vol", "min_volatility");
}

#[test]
fn min_vol_with_a_floor_on_one_asset() {
    check("min_vol_asset0_floor", "min_volatility");
}

#[test]
fn min_vol_with_every_weight_boxed() {
    check("min_vol_capped", "min_volatility");
}

#[test]
fn max_sharpe_long_only() {
    check("max_sharpe", "max_sharpe");
}

#[test]
fn max_sharpe_with_a_floor_on_one_asset() {
    check("max_sharpe_asset0_floor", "max_sharpe");
}

#[test]
fn efficient_risk_at_a_binding_target() {
    check("efficient_risk", "efficient_risk");
}

#[test]
fn efficient_risk_below_the_min_variance_return_is_the_min_variance_portfolio() {
    check("efficient_risk_below_min_var", "efficient_risk");
}

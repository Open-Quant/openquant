//! The Critical Line Algorithm against scipy, on the fixed (mu, C) of
//! `tests/fixtures/portfolio_optimization/` (see generate_qp_reference.py there).
//!
//! CLA and a QP solver are different algorithms for the same problem, so agreement between
//! them, and with scipy, is evidence about both.
use nalgebra::DMatrix;
use openquant::cla::{WeightBounds, CLA};
use openquant::portfolio_optimization::{allocate_from_inputs, AllocationOptions};
use serde_json::Value;
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

fn run(bounds: WeightBounds, solution: &str) -> CLA {
    let (mu, cov) = inputs();
    let mut cla = CLA::new(bounds, "mean");
    let mu_col = DMatrix::from_column_slice(mu.len(), 1, &mu);
    cla.allocate(None, Some(&mu_col), Some(&cov), None, Some(solution)).unwrap();
    cla
}

fn ret(w: &[f64], mu: &[f64]) -> f64 {
    w.iter().zip(mu).map(|(a, b)| a * b).sum()
}

fn variance(w: &[f64], cov: &DMatrix<f64>) -> f64 {
    let n = w.len();
    (0..n).map(|i| (0..n).map(|j| w[i] * cov[(i, j)] * w[j]).sum::<f64>()).sum()
}

/// Variance of the CLA frontier at return `r`: weights are linear between turning points and
/// so is the return, so the portfolio at `r` is found by interpolating within its segment.
fn frontier_variance(points: &[Vec<f64>], mu: &[f64], cov: &DMatrix<f64>, r: f64) -> f64 {
    for pair in points.windows(2) {
        let (r0, r1) = (ret(&pair[0], mu), ret(&pair[1], mu));
        // scipy locates the ends of the frontier to about 1e-9 in return, so a target may fall
        // that far outside the segment it belongs to.
        if r <= r0 + 1e-7 && r >= r1 - 1e-7 && (r0 - r1).abs() > 1e-15 {
            let a = ((r - r1) / (r0 - r1)).clamp(0.0, 1.0);
            let w: Vec<f64> =
                pair[0].iter().zip(&pair[1]).map(|(x, y)| a * x + (1.0 - a) * y).collect();
            return variance(&w, cov);
        }
    }
    panic!("return {r} is outside the frontier");
}

fn check_turning_points(bounds: WeightBounds, lower: f64, upper: f64, frontier: &str) {
    let (mu, cov) = inputs();
    let cla = run(bounds, "cla_turning_points");
    let points = &cla.weights;
    assert!(points.len() >= 3, "only {} turning point(s)", points.len());
    assert_eq!(cla.lambdas.len(), points.len());

    for w in points {
        assert!((w.iter().sum::<f64>() - 1.0).abs() < 1e-9);
        assert!(w.iter().all(|x| *x >= lower - 1e-9 && *x <= upper + 1e-9));
    }
    // From the maximum-return corner down to minimum variance, return and risk never rise. They
    // are not strictly falling: the corner appears twice, once at lambda = infinity and once at
    // the lambda where the second asset enters with a weight of zero, as in the paper's output.
    for pair in points.windows(2) {
        assert!(ret(&pair[0], &mu) >= ret(&pair[1], &mu) - 1e-12);
        assert!(variance(&pair[0], &cov) >= variance(&pair[1], &cov) - 1e-18);
    }
    assert!(ret(&points[0], &mu) > ret(points.last().unwrap(), &mu));
    assert!(cla.lambdas[0].is_infinite());
    // Never rising; equal when several assets change status at the same lambda.
    assert!(cla.lambdas.windows(2).all(|l| l[0] >= l[1]));
    assert!(cla.lambdas[1] > 0.0);
    assert_eq!(*cla.lambdas.last().unwrap(), 0.0);

    let reference = &read("qp_reference.json")["frontiers"][frontier];
    let top = ret(&points[0], &mu);
    assert!((top - reference["max_return"].as_f64().unwrap()).abs() < 1e-9, "max return {top}");

    // scipy's minimum variance at each target return. SLSQP resolves the variance to about
    // 1e-9 relative here; 1e-6 leaves room for that and nothing else.
    for point in reference["points"].as_array().unwrap() {
        let r = point["target_return"].as_f64().unwrap();
        let want = point["variance"].as_f64().unwrap();
        let got = frontier_variance(points, &mu, &cov, r);
        assert!((got - want).abs() <= 1e-6 * want, "at return {r}: CLA {got}, scipy {want}");
    }
}

#[test]
fn long_only_frontier_matches_scipy() {
    check_turning_points(WeightBounds::Tuple(0.0, 1.0), 0.0, 1.0, "long_only");
}

#[test]
fn boxed_frontier_matches_scipy() {
    check_turning_points(WeightBounds::Tuple(0.01, 0.15), 0.01, 0.15, "capped");
}

#[test]
fn the_first_turning_point_is_all_in_the_best_asset() {
    let (mu, _) = inputs();
    let cla = run(WeightBounds::Tuple(0.0, 1.0), "cla_turning_points");
    let best = (0..mu.len()).max_by(|&a, &b| mu[a].total_cmp(&mu[b])).unwrap();
    assert!((cla.weights[0][best] - 1.0).abs() < 1e-12);
}

/// CLA and the QP solver are independent routes to the same two portfolios.
#[test]
fn min_volatility_and_max_sharpe_agree_with_the_qp_solver() {
    let (mu, cov) = inputs();
    for (solution, tolerance) in [("min_volatility", 1e-8), ("max_sharpe", 1e-6)] {
        let cla = run(WeightBounds::Tuple(0.0, 1.0), solution);
        assert_eq!(cla.weights.len(), 1);
        let qp = allocate_from_inputs(&mu, &cov, solution, &AllocationOptions::default()).unwrap();
        for (i, (a, b)) in cla.weights[0].iter().zip(&qp.weights).enumerate() {
            assert!((a - b).abs() < tolerance, "{solution}: asset {i} CLA {a}, QP {b}");
        }
    }
}

#[test]
fn the_efficient_frontier_is_a_curve_not_one_point() {
    let cla = run(WeightBounds::Tuple(0.0, 1.0), "efficient_frontier");
    assert!(cla.weights.len() >= 50, "{} points", cla.weights.len());
    assert_eq!(cla.efficient_frontier_means.len(), cla.weights.len());
    assert!(cla.efficient_frontier_means.windows(2).all(|m| m[0] >= m[1]));
    assert!(cla.efficient_frontier_sigma.windows(2).all(|s| s[0] >= s[1] - 1e-15));
    assert!(cla.efficient_frontier_sigma[0] > 2.0 * cla.efficient_frontier_sigma.last().unwrap());
}

#[test]
fn bounds_are_honoured_when_returns_and_covariance_are_supplied() {
    // This path used to overwrite the caller's bounds with [0, 1].
    let cla = run(WeightBounds::Tuple(0.01, 0.15), "min_volatility");
    assert!(cla.weights[0].iter().all(|w| *w >= 0.01 - 1e-9 && *w <= 0.15 + 1e-9));
}

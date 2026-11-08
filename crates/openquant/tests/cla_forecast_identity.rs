use nalgebra::DMatrix;
use openquant::cla::{WeightBounds, CLA};

fn run(mu: &[f64], variances: &[f64], bounds: WeightBounds, solution: &str) -> CLA {
    let mut cla = CLA::new(bounds, "mean");
    let cov = DMatrix::from_diagonal(&nalgebra::DVector::from_column_slice(variances));
    cla.allocate(
        None,
        Some(&DMatrix::from_column_slice(mu.len(), 1, mu)),
        Some(&cov),
        None,
        Some(solution),
    )
    .unwrap();
    cla
}

#[test]
fn equal_forecasts_keep_true_frontier_and_positive_tangency() {
    for mu in [1e-8, 0.1] {
        for solution in ["max_sharpe", "efficient_frontier", "min_volatility"] {
            let cla = run(&[mu, mu], &[1., 1.], WeightBounds::Tuple(0., 1.), solution);
            for w in &cla.weights {
                assert!((w[0] - 0.5).abs() < 1e-7, "{solution}: {w:?}");
            }
            for mean in &cla.efficient_frontier_means {
                assert!((mean - mu).abs() < mu.abs() * 1e-12 + 1e-20, "{mean} != {mu}");
            }
            assert_eq!(cla.expected_returns.as_slice(), &[mu, mu]);
        }
    }
}

#[test]
fn nearly_equal_forecasts_keep_order_and_tangency() {
    for mu in [[1e-8 + 5e-13, 1e-8], [1e-8, 1e-8 + 5e-13]] {
        let cla = run(&mu, &[1., 1.], WeightBounds::Tuple(0., 1.), "cla_turning_points");
        let best = if mu[0] > mu[1] { 0 } else { 1 };
        assert!((cla.weights[0][best] - 1.).abs() < 1e-8);
        let cla = run(&mu, &[1., 1.], WeightBounds::Tuple(0., 1.), "max_sharpe");
        assert!((cla.weights[0][0] - mu[0] / (mu[0] + mu[1])).abs() < 1e-7, "{:?}", cla.weights);
    }
}

#[test]
fn equal_forecasts_respect_active_bounds_and_diagonal_risk() {
    // Unbounded GMV weights proportional to inverse variances: [4/7,2/7,1/7].
    // At upper bound .5 on asset0, remaining .5 divides 2:1.
    let cla = run(&[1e-8; 3], &[1., 2., 4.], WeightBounds::Tuple(0., 0.5), "min_volatility");
    for (got, want) in cla.weights[0].iter().zip([0.5, 1. / 3., 1. / 6.]) {
        assert!((got - want).abs() < 1e-7, "{:?}", cla.weights);
    }
}

#[test]
fn single_asset_forecast_is_not_adjusted() {
    let cla = run(&[0.1], &[1.], WeightBounds::Tuple(0., 1.), "efficient_frontier");
    assert_eq!(cla.expected_returns[(0, 0)], 0.1);
    assert!(cla.efficient_frontier_means.iter().all(|v| *v == 0.1));
}

#[test]
fn zero_and_negative_ties_search_only_the_efficient_frontier() {
    // Constant returns leave one nondominated point, GMV. For negative returns this
    // deliberately differs from globally maximizing Sharpe over dominated portfolios.
    for mu in [0., -0.1] {
        let cla = run(&[mu, mu], &[4., 1.], WeightBounds::Tuple(0., 1.), "max_sharpe");
        assert!((cla.weights[0][0] - 0.2).abs() < 1e-7, "{:?}", cla.weights);
        assert_eq!(cla.expected_returns.as_slice(), &[mu, mu]);
    }
}

#[test]
fn exact_tie_fixed_bounds_remain_feasible() {
    let cla = run(
        &[0.1; 2],
        &[1., 2.],
        WeightBounds::Lists(vec![0.3, 0.7], vec![0.3, 0.7]),
        "efficient_frontier",
    );
    for w in &cla.weights {
        assert!((w[0] - 0.3).abs() < 1e-9);
        assert!((w[1] - 0.7).abs() < 1e-9);
    }
    for mean in &cla.efficient_frontier_means {
        assert!((mean - 0.1).abs() < 1e-12);
    }
}

#[test]
fn equal_forecasts_correlated_covariance_matches_closed_form() {
    let mut cla = CLA::default();
    let mu = DMatrix::from_column_slice(2, 1, &[0.1, 0.1]);
    let cov = DMatrix::from_row_slice(2, 2, &[0.04, 0.006, 0.006, 0.09]);
    cla.allocate(None, Some(&mu), Some(&cov), None, Some("max_sharpe")).unwrap();
    assert!((cla.weights[0][0] - 84. / 118.).abs() < 1e-8, "{:?}", cla.weights);
}

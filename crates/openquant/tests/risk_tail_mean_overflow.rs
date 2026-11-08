use openquant::risk_metrics::RiskMetrics;
use openquant::strategy_risk::{estimate_strategy_failure_probability, StrategyRiskConfig};

#[test]
fn expected_shortfall_recovers_finite_tail_sum_overflow() {
    assert_eq!(RiskMetrics.calculate_expected_shortfall(&[-1e308, -1e308, 0.0], 1.0), Ok(-1e308));
    assert_eq!(
        RiskMetrics.calculate_expected_shortfall(&[-1e308, -1e308, 5e307, 1e308], 1.0),
        Ok(-5e307)
    );
}

#[test]
fn cdar_recovers_finite_tail_sum_overflow() {
    assert_eq!(RiskMetrics.calculate_conditional_drawdown_risk(&[1e308, 0.0, 0.0], 0.5), Ok(1e308));
}

#[test]
fn strategy_failure_uses_finite_average_payouts() {
    let cfg = StrategyRiskConfig {
        years_elapsed: 1.0,
        target_sharpe: 1.0,
        investor_horizon_years: 1.0,
        bootstrap_iterations: 20,
        seed: 42,
        kde_bandwidth: None,
    };
    let baseline = estimate_strategy_failure_probability(&[1.0, 0.9, -1.0, -0.9], cfg).unwrap();
    let report =
        estimate_strategy_failure_probability(&[1e308, 9e307, -1e308, -9e307], cfg).unwrap();
    assert!(report.payout.pi_plus.is_finite());
    assert!(
        (report.implied_precision_threshold - baseline.implied_precision_threshold).abs() < 1e-15
    );
    assert_eq!(report.bootstrap_precision_samples, baseline.bootstrap_precision_samples);
}

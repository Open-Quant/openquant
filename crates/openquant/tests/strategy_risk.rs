use openquant::strategy_risk::{
    estimate_strategy_failure_probability, implied_frequency_asymmetric,
    implied_frequency_symmetric, implied_precision_asymmetric, implied_precision_symmetric,
    sharpe_asymmetric, sharpe_symmetric, AsymmetricPayout, StrategyRiskConfig,
};

#[test]
fn test_symmetric_inverse_consistency() {
    let target = 2.0;
    let n = 260.0;
    let p = implied_precision_symmetric(target, n).unwrap();
    let sr = sharpe_symmetric(p, n).unwrap();
    let implied_n = implied_frequency_symmetric(p, target).unwrap();

    assert!((sr - target).abs() < 1e-8);
    assert!((implied_n - n).abs() < 1e-8);
    assert!(p > 0.5);
}

#[test]
fn test_asymmetric_inverse_consistency() {
    let payout = AsymmetricPayout { pi_plus: 0.005, pi_minus: -0.01 };
    let target = 1.5;
    let n = 260.0;

    let p = implied_precision_asymmetric(target, n, payout).unwrap();
    let sr = sharpe_asymmetric(p, n, payout).unwrap();
    let implied_n = implied_frequency_asymmetric(p, target, payout).unwrap();

    assert!((sr - target).abs() < 1e-7);
    assert!((implied_n - n).abs() < 1e-6);
    assert!((0.5..1.0).contains(&p));
}

#[test]
fn test_sensitivity_to_small_parameter_changes() {
    let payout = AsymmetricPayout { pi_plus: 0.005, pi_minus: -0.01 };
    let base_sr = sharpe_asymmetric(0.70, 260.0, payout).unwrap();
    let higher_p_sr = sharpe_asymmetric(0.71, 260.0, payout).unwrap();
    let lower_n_sr = sharpe_asymmetric(0.70, 240.0, payout).unwrap();
    let worse_loss_sr =
        sharpe_asymmetric(0.70, 260.0, AsymmetricPayout { pi_plus: 0.005, pi_minus: -0.011 })
            .unwrap();

    assert!(higher_p_sr > base_sr);
    assert!(lower_n_sr < base_sr);
    assert!(worse_loss_sr < base_sr);
}

#[test]
fn test_strategy_failure_probability_workflow() {
    let mut outcomes = Vec::new();
    for i in 0..1200 {
        if i % 10 < 7 {
            outcomes.push(0.005);
        } else {
            outcomes.push(-0.01);
        }
    }

    let report = estimate_strategy_failure_probability(
        &outcomes,
        StrategyRiskConfig {
            years_elapsed: 5.0,
            target_sharpe: 2.0,
            investor_horizon_years: 2.0,
            bootstrap_iterations: 2_000,
            seed: 17,
            kde_bandwidth: None,
        },
    )
    .unwrap();

    assert!(report.annual_bet_frequency > 200.0);
    assert!((0.0..=1.0).contains(&report.implied_precision_threshold));
    assert!((0.0..=1.0).contains(&report.empirical_failure_probability));
    assert!((0.0..=1.0).contains(&report.kde_failure_probability));
    assert!(report.bootstrap_precision_std > 0.0);
    assert_eq!(report.bootstrap_precision_samples.len(), 2_000);
}

#[test]
fn test_failure_probability_rises_with_higher_target_sharpe() {
    let mut outcomes = Vec::new();
    for i in 0..1400 {
        if i % 10 < 7 {
            outcomes.push(0.006);
        } else {
            outcomes.push(-0.009);
        }
    }

    let cfg = StrategyRiskConfig {
        years_elapsed: 5.0,
        target_sharpe: 1.0,
        investor_horizon_years: 2.0,
        bootstrap_iterations: 1_200,
        seed: 33,
        kde_bandwidth: None,
    };
    let low_target = estimate_strategy_failure_probability(&outcomes, cfg).unwrap();
    let high_target = estimate_strategy_failure_probability(
        &outcomes,
        StrategyRiskConfig { target_sharpe: 2.0, ..cfg },
    )
    .unwrap();

    assert!(high_target.implied_precision_threshold > low_target.implied_precision_threshold);
    assert!(high_target.kde_failure_probability >= low_target.kde_failure_probability);
}

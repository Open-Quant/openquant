use openquant::strategy_risk::{
    estimate_strategy_failure_probability, implied_frequency_asymmetric,
    implied_frequency_symmetric, implied_precision_asymmetric, implied_precision_symmetric,
    sharpe_asymmetric, sharpe_symmetric, AsymmetricPayout, StrategyRiskConfig, StrategyRiskError,
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

/// #168: the implied-frequency formulas square the Sharpe ratio, so a negative mean payoff
/// used to get the frequency of the opposite edge. No frequency reaches a positive target
/// when the Sharpe ratio is negative at every frequency.
#[test]
fn test_implied_frequency_rejects_a_negative_mean_payoff() {
    // Symmetric: precision 0.45 is the mirror image of 0.55, which needs 396 bets a year.
    assert!((implied_frequency_symmetric(0.55, 2.0).unwrap() - 396.0).abs() < 1e-9);
    assert!(sharpe_symmetric(0.45, 396.0).unwrap() < 0.0);
    assert!(matches!(
        implied_frequency_symmetric(0.45, 2.0),
        Err(StrategyRiskError::NoValidRoot(_))
    ));

    // Asymmetric: win 1%, lose 2%, precision 0.6 -> mean payoff 0.03 * 0.6 - 0.02 < 0.
    let payout = AsymmetricPayout { pi_plus: 0.01, pi_minus: -0.02 };
    assert!(sharpe_asymmetric(0.6, 260.0, payout).unwrap() < 0.0);
    assert!(matches!(
        implied_frequency_asymmetric(0.6, 2.0, payout),
        Err(StrategyRiskError::NoValidRoot(_))
    ));
    // Above break-even (2/3) it still inverts sharpe_asymmetric.
    let n = implied_frequency_asymmetric(0.8, 2.0, payout).unwrap();
    assert!((sharpe_asymmetric(0.8, n, payout).unwrap() - 2.0).abs() < 1e-9);
}

/// #168: the formulas need only pi_plus > pi_minus; neither payout has to be negative.
#[test]
fn test_payouts_need_only_be_ordered() {
    let both_positive = AsymmetricPayout { pi_plus: 0.02, pi_minus: 0.01 };
    let sr = sharpe_asymmetric(0.5, 260.0, both_positive).unwrap();
    // mean 0.015, sd 0.005: Sharpe 3 per bet.
    assert!((sr - 3.0 * 260f64.sqrt()).abs() < 1e-9);

    let reversed = AsymmetricPayout { pi_plus: -0.01, pi_minus: 0.01 };
    assert!(matches!(
        sharpe_asymmetric(0.5, 260.0, reversed),
        Err(StrategyRiskError::InvalidInput(_))
    ));
}

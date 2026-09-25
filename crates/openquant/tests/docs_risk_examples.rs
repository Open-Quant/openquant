//! The Rust examples on the risk_metrics, strategy_risk and codependence docs pages, run as
//! tests. `check:examples` only compiles page snippets; these execute them.
//!
//! Each body is the page's snippet, reformatted by rustfmt. If you change one, change the other.

#[test]
fn risk_metrics_page() -> Result<(), Box<dyn std::error::Error>> {
    use nalgebra::DMatrix;
    use openquant::risk_metrics::{RiskMetrics, RiskMetricsError};

    let risk = RiskMetrics;
    let returns = [-0.08, -0.03, -0.01, 0.0, 0.01, 0.01, 0.02, 0.02, 0.03, 0.04];

    // ceil(0.25 * 9) = 3: the fourth-smallest return. The three below it average -0.04.
    assert_eq!(risk.calculate_value_at_risk(&returns, 0.25)?, 0.0);
    assert!((risk.calculate_expected_shortfall(&returns, 0.25)? + 0.04).abs() < 1e-12);

    // Nothing lies strictly below the minimum, so the tail is empty.
    assert!(risk.calculate_expected_shortfall(&returns, 0.0)?.is_nan());

    // Conditional drawdown at risk takes a cumulative series, and 0.9 means the worst 10%.
    // Drawdowns 0 0 1 0 4 1. At 0.6 the threshold is the fourth-smallest, 1; (1, 1, 4) average 2.
    let equity = [1.0, 3.0, 2.0, 5.0, 1.0, 4.0];
    assert_eq!(risk.calculate_conditional_drawdown_risk(&equity, 0.6)?, 2.0);
    assert_eq!(risk.calculate_conditional_drawdown_risk(&equity, 0.9)?, 4.0);

    let covariance = DMatrix::from_row_slice(2, 2, &[0.04, 0.01, 0.01, 0.09]);
    assert!((risk.calculate_variance(&covariance, &[0.6, 0.4])? - 0.0336).abs() < 1e-12);
    assert_eq!(
        risk.calculate_variance(&covariance, &[1.0]),
        Err(RiskMetricsError::DimensionMismatch)
    );
    assert_eq!(
        risk.calculate_value_at_risk(&returns, 1.5),
        Err(RiskMetricsError::InvalidConfidenceLevel)
    );
    Ok(())
}

#[test]
fn strategy_risk_page() -> Result<(), Box<dyn std::error::Error>> {
    use openquant::strategy_risk::{
        implied_frequency_symmetric, implied_precision_asymmetric, implied_precision_symmetric,
        sharpe_asymmetric, sharpe_symmetric, AsymmetricPayout, StrategyRiskError,
    };

    // 55% precision, daily bets.
    assert!((sharpe_symmetric(0.55, 260.0)? - 1.6206).abs() < 1e-4);
    // Reaching a Sharpe ratio of 2 at that precision takes 396 bets a year.
    assert!((implied_frequency_symmetric(0.55, 2.0)? - 396.0).abs() < 1e-6);
    // The inverse functions agree with the forward one.
    let p = implied_precision_symmetric(2.0, 396.0)?;
    assert!((p - 0.55).abs() < 1e-9);

    let payout = AsymmetricPayout { pi_plus: 0.01, pi_minus: -0.02 };
    let needed = implied_precision_asymmetric(2.0, 260.0, payout)?;
    assert!((needed - 0.7222).abs() < 1e-4);
    assert!((sharpe_asymmetric(needed, 260.0, payout)? - 2.0).abs() < 1e-6);

    // Precision of exactly one half never reaches a positive target, at any frequency,
    // and below one half the Sharpe ratio is negative, so no frequency reaches one.
    assert!(matches!(
        implied_frequency_symmetric(0.5, 1.0),
        Err(StrategyRiskError::InvalidInput(_))
    ));
    assert!(matches!(
        implied_frequency_symmetric(0.45, 2.0),
        Err(StrategyRiskError::NoValidRoot(_))
    ));
    Ok(())
}

#[test]
fn codependence_page() -> Result<(), Box<dyn std::error::Error>> {
    use openquant::codependence::{
        absolute_angular_distance, angular_distance, distance_correlation,
        get_optimal_number_of_bins, variation_of_information_score, CodependenceError,
    };

    let x: Vec<f64> = (0..=200).map(|i| f64::from(i) / 100.0 - 1.0).collect();
    let mirrored: Vec<f64> = x.iter().map(|v| -v).collect();
    let squared: Vec<f64> = x.iter().map(|v| v * v).collect();

    // rho = -1: maximal angular distance, zero absolute angular distance.
    assert!((angular_distance(&x, &mirrored)? - 1.0).abs() < 1e-12);
    assert!(absolute_angular_distance(&x, &mirrored)?.abs() < 1e-7);

    // y = x^2 on a symmetric range is uncorrelated with x, and clearly dependent on it.
    assert!((angular_distance(&x, &squared)? - 0.5f64.sqrt()).abs() < 1e-3);
    assert!(distance_correlation(&x, &squared)? > 0.4);

    // A one-to-one relationship leaves no uncertainty either way.
    assert!(variation_of_information_score(&x, &mirrored, None, true)?.abs() < 1e-12);

    assert_eq!(get_optimal_number_of_bins(1_000, None)?, 15);
    assert_eq!(get_optimal_number_of_bins(1_000, Some(0.9))?, 13);
    assert!(matches!(angular_distance(&x, &[1.0]), Err(CodependenceError::InputLengthMismatch)));
    Ok(())
}

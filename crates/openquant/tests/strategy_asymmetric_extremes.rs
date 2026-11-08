use openquant::strategy_risk::{
    implied_frequency_asymmetric, implied_precision_asymmetric, sharpe_asymmetric,
    AsymmetricPayout, StrategyRiskError,
};

#[test]
fn sharpe_asymmetric_is_invariant_to_payout_scale() {
    let expected = 4.0 / 3.0_f64.sqrt();
    for scale in [1.0, 1e200, 1e308, 1e-200, 1e-308, f64::from_bits(1)] {
        let payout = AsymmetricPayout { pi_plus: scale, pi_minus: -scale };
        let result = sharpe_asymmetric(0.75, 16.0, payout).unwrap();
        assert!((result - expected).abs() < 2e-14, "scale={scale}, result={result}");
    }
}

#[test]
fn implied_precision_is_invariant_to_payout_scale() {
    // Independent Decimal80 quadratic reference for target1.5,n260,plus1,minus-2.
    let expected = 0.7089245166113844;
    for scale in [1.0, 1e100, 1e200, 1e-100, 1e-200, 1e-300] {
        let payout = AsymmetricPayout { pi_plus: scale, pi_minus: -2.0 * scale };
        let result = implied_precision_asymmetric(1.5, 260.0, payout).unwrap();
        assert!((result - expected).abs() < 1e-12, "scale={scale}, result={result}");
    }
}

#[test]
fn implied_precision_handles_large_target_and_frequency_together() {
    let result = implied_precision_asymmetric(
        1e154,
        1e308,
        AsymmetricPayout { pi_plus: 1.0, pi_minus: -1.0 },
    )
    .unwrap();
    assert!((result - (1.0 + 1.0 / 2.0_f64.sqrt()) / 2.0).abs() < 1e-14);
}

#[test]
fn implied_frequency_recovers_finite_result_for_large_payouts() {
    for scale in [1.0, 1e100, 1e200, 1e308] {
        let result = implied_frequency_asymmetric(
            0.75,
            2.0,
            AsymmetricPayout { pi_plus: scale, pi_minus: -scale },
        )
        .unwrap();
        assert!((result - 12.0).abs() < 1e-12, "scale={scale}, result={result}");
    }
}

#[test]
fn implied_frequency_preserves_documented_absolute_mean_guard() {
    assert!(matches!(
        implied_frequency_asymmetric(
            0.75,
            2.0,
            AsymmetricPayout { pi_plus: 1e-20, pi_minus: -1e-20 }
        ),
        Err(StrategyRiskError::NoValidRoot(_))
    ));
}

#[test]
fn small_positive_target_does_not_select_negative_sharpe_root() {
    let payout = AsymmetricPayout { pi_plus: 1.0, pi_minus: -1.0 };
    let p = implied_precision_asymmetric(1e-8, 1.0, payout).unwrap();
    assert!(p > 0.5);
    assert!((p - 0.500000005).abs() < 1e-15);
}

#[test]
fn two_positive_payouts_keep_the_smaller_stable_root() {
    let payout = AsymmetricPayout { pi_plus: 2.0, pi_minus: 1.0 };
    let p = implied_precision_asymmetric(2e7, 1.0, payout).unwrap();
    // Decimal180 reference, including exact finite-target correction.
    assert!((p / 2.500_000_000_000_019e-15 - 1.0).abs() < 1e-14);
    let sr = sharpe_asymmetric(p, 1.0, payout).unwrap();
    assert!((sr / 2e7 - 1.0).abs() < 1e-14);
}

#[test]
fn subnormal_raw_discriminant_uses_scaled_coefficients() {
    let p = implied_precision_asymmetric(
        1.0882170540592825e51,
        4.464105562080811e101,
        AsymmetricPayout { pi_plus: 6.512040548456934e-131, pi_minus: 2.0300908238330063e-131 },
    )
    .unwrap();
    // Independent Decimal350 reference using the exact binary64 inputs.
    assert!((p - 0.20734158271672556).abs() < 1e-13);
}

#[test]
fn frequency_recovers_partial_intermediate_underflow() {
    // n=3*theta² for precision.75 and payouts±1; do not quantize theta²
    // before multiplying by3 when its rounded square is subnormal.
    let actual = implied_frequency_asymmetric(
        0.75,
        1e-161,
        AsymmetricPayout { pi_plus: 1.0, pi_minus: -1.0 },
    )
    .unwrap();
    let expected = 3e-322;
    assert_eq!(actual, expected);
    // A subnormal numerator divided by a small (normal) mean square gives a
    // normal frequency. The high-precision result must retain its precision.
    let actual = implied_frequency_asymmetric(
        0.75,
        1e-148,
        AsymmetricPayout { pi_plus: 4e-12, pi_minus: -4e-12 },
    )
    .unwrap();
    assert!((actual / 3e-296 - 1.0).abs() < 1e-14);
}

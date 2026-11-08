use openquant::strategy_risk::{implied_frequency_symmetric, implied_precision_symmetric};

#[test]
fn implied_precision_does_not_square_extreme_target() {
    let expected = (1.0 + 1.0 / 2.0_f64.sqrt()) / 2.0;
    let actual = implied_precision_symmetric(1e154, 1e308).unwrap();
    assert!((actual - expected).abs() < 1e-15, "{actual}");
    assert_eq!(implied_precision_symmetric(1e200, 1.0).unwrap(), 1.0);
}

#[test]
fn finite_frequency_survives_overflowing_target_square() {
    let p = f64::from_bits(1.0_f64.to_bits() - 1);
    // Independent high-precision evaluation of 4 theta² p(1-p)/(2p-1)².
    let actual = implied_frequency_symmetric(p, 1e160).unwrap();
    let expected = 4.440892098500628e304;
    assert!((actual / expected - 1.0).abs() < 1e-14, "{actual}");
}

#[test]
fn small_target_square_keeps_frequency_accuracy() {
    let p = 0.50000000001;
    let actual = implied_frequency_symmetric(p, 1e-160).unwrap();
    let expected = 2.4999995862981964e-299;
    assert!((actual / expected - 1.0).abs() < 1e-14, "{actual}");
}

#[test]
fn implied_precision_recovers_a_subnormal_squared_norm() {
    // Decimal100 reference of the formula applied to the exact binary64 inputs.
    let actual = implied_precision_symmetric(1e-162, 1e-323).unwrap();
    assert!((actual - 0.651575620418893).abs() < 1e-15, "{actual}");
}

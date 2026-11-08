use openquant::microstructural_features::vwap;

#[test]
fn vwap_recovers_one_or_both_overflowed_sums() {
    assert_eq!(vwap(&[1e308; 2], &[1e308; 2]).unwrap(), 1.0);
    assert_eq!(vwap(&[1e308; 2], &[1.0; 2]).unwrap(), 1e308);
    assert_eq!(vwap(&[1.0; 2], &[1e308; 2]).unwrap(), 1e-308);
    assert_eq!(vwap(&[f64::MAX; 2], &[f64::MAX / 2.0; 2]).unwrap(), 2.0);
    assert_eq!(vwap(&[f64::MAX / 2.0; 2], &[f64::MAX; 2]).unwrap(), 0.5);
}

#[test]
fn vwap_uses_exact_remainders_at_maximum_finite_boundaries() {
    assert_eq!(
        vwap(&[f64::MAX, 2.0_f64.powi(969)], &[0.5, 0.5_f64.next_down()]).unwrap(),
        f64::INFINITY
    );
    // The numerator sum is exactly the overflow midpoint. A denominator just
    // above one moves the true ratio below it, so the result rounds to MAX.
    assert_eq!(vwap(&[f64::MAX, 2.0_f64.powi(970)], &[1.0, f64::from_bits(1)]).unwrap(), f64::MAX);
    assert_eq!(vwap(&[f64::MAX; 2], &[1.0, 1.0_f64.next_down()]).unwrap(), f64::MAX);
    assert_eq!(
        vwap(&[f64::MAX; 2], &[1.0, 1.0_f64.next_down().next_down()]).unwrap(),
        f64::INFINITY
    );
}

#[test]
fn vwap_preserves_ordinary_sum_rounding() {
    // The existing sequential sum rounds away both unit increments. Ordinary
    // finite sums away from range limits retain this behavior.
    assert_eq!(vwap(&[1e16, 1.0, 1.0], &[1.0; 3]).unwrap().to_bits(), (1e16_f64 / 3.0).to_bits());
}

#[test]
fn vwap_preserves_zero_sign_and_signed_invalid_behavior() {
    assert_eq!(vwap(&[-0.0; 2], &[f64::MAX; 2]).unwrap().to_bits(), (-0.0_f64).to_bits());
    assert_eq!(vwap(&[-0.0; 2], &[-f64::MAX; 2]).unwrap().to_bits(), 0.0_f64.to_bits());
    assert!(vwap(&[], &[]).unwrap().is_nan());
    assert!(vwap(&[1.0], &[0.0]).unwrap().is_nan());
    assert!(vwap(&[f64::NAN], &[1.0]).unwrap().is_nan());
    assert!(vwap(&[f64::INFINITY], &[f64::INFINITY]).unwrap().is_nan());
    assert_eq!(vwap(&[-1.0, -2.0], &[1.0, 1.0]).unwrap(), -1.5);
    assert!(vwap(&[-f64::MAX; 2], &[f64::MAX; 2]).unwrap().is_nan());
    assert!(vwap(&[1.0], &[]).is_err());
}

#[test]
fn vwap_recovers_subnormal_and_underflowed_output_ties() {
    let tiny = f64::from_bits(1);
    // The rounded denominator is two in both cases. The exact denominator
    // selects the opposite side of the subnormal rounding midpoint.
    assert_eq!(vwap(&[tiny, 0.0], &[1.0, 1.0_f64.next_down()]).unwrap(), tiny);
    assert_eq!(vwap(&[3.0 * tiny, 0.0], &[1.0, 1.0_f64.next_up()]).unwrap(), tiny);
}

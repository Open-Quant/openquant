use openquant::util::stats::{quantile, quantile_sorted, QuantileMethod};

#[test]
fn linear_quantiles_do_not_overflow_the_distance_between_finite_endpoints() {
    // Exact dyadic inputs and fractions: the expectations are hand-derived convex
    // combinations, not values obtained from the implementation under test.
    let bound = 2.0_f64.powi(1023);
    for (q, expected) in
        [(0.0, -bound), (0.25, -bound / 2.0), (0.5, 0.0), (0.75, bound / 2.0), (1.0, bound)]
    {
        assert_eq!(quantile_sorted(&[-bound, bound], q, QuantileMethod::Linear), Some(expected));
        assert_eq!(quantile(&[bound, -bound], q, QuantileMethod::Linear), Some(expected));
    }
}

#[test]
fn linear_quantile_overflow_recovery_uses_the_fraction_within_the_selected_interval() {
    let bound = 2.0_f64.powi(1023);
    // n=3 and q=3/8 gives h=3/4, within the first interval [-bound, bound].
    assert_eq!(
        quantile_sorted(&[-bound, bound, bound], 0.375, QuantileMethod::Linear),
        Some(bound / 2.0)
    );
}

#[test]
fn overflowing_span_preserves_small_cancellation_residuals() {
    let bound = 2.0_f64.powi(1023);
    let below_half = f64::from_bits(0.5_f64.to_bits() - 1);
    let above_half = f64::from_bits(0.5_f64.to_bits() + 1);
    // q = 1/2 - 2^-54 and 1/2 + 2^-53, respectively. For endpoints
    // [-2^1023, 2^1023], the exact values are -2^970 and 2^971.
    assert_eq!(
        quantile_sorted(&[-bound, bound], below_half, QuantileMethod::Linear),
        Some(-2.0_f64.powi(970))
    );
    assert_eq!(
        quantile_sorted(&[-bound, bound], above_half, QuantileMethod::Linear),
        Some(2.0_f64.powi(971))
    );
}

#[test]
fn ordinary_linear_and_non_interpolated_quantile_contracts_remain_unchanged() {
    for (q, expected) in [(0.0, 10.0), (0.25, 17.5), (0.5, 25.0), (1.0, 40.0)] {
        assert_eq!(
            quantile_sorted(&[10.0, 20.0, 30.0, 40.0], q, QuantileMethod::Linear),
            Some(expected)
        );
    }
    let bound = 2.0_f64.powi(1023);
    assert_eq!(quantile_sorted(&[-bound, bound], 0.25, QuantileMethod::Higher), Some(bound));
    assert_eq!(quantile_sorted(&[-bound, bound], 0.25, QuantileMethod::Nearest), Some(-bound));
    assert!(quantile_sorted(&[f64::NEG_INFINITY, f64::INFINITY], 0.5, QuantileMethod::Linear)
        .unwrap()
        .is_nan());
}

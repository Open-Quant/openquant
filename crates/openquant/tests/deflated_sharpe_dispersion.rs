use openquant::backtest_statistics::deflated_sharpe_ratio;

fn empirical_benchmark(values: &[f64]) -> f64 {
    deflated_sharpe_ratio(0.0, values, 3, 0.0, 3.0, false, true).unwrap()
}

#[test]
fn empirical_dispersion_recovers_extreme_symmetric_samples() {
    // For [-a,a] the exact population standard deviation is |a|.
    // The multiplier is an independent scipy.stats.norm.isf reference for N=2.
    let multiplier = 0.5197553442805939;
    for magnitude in [1e200, 1e-200, 1e-160, f64::MAX] {
        let actual = empirical_benchmark(&[-magnitude, magnitude]);
        let expected = magnitude * multiplier;
        assert!((actual / expected - 1.0).abs() < 1e-12, "{magnitude}: {actual}");
    }
}

#[test]
fn empirical_dispersion_centers_adjacent_huge_floats_before_rounding() {
    let low = 1e308_f64;
    let high = f64::from_bits(low.to_bits() + 1);
    // Two exact float inputs have population std=(high-low)/2, even when
    // their exact mean lies between representable values.
    let expected = (high - low) / 2.0 * 0.5197553442805939;
    let actual = empirical_benchmark(&[low, high]);
    assert!((actual / expected - 1.0).abs() < 1e-12);
    assert_eq!(actual, empirical_benchmark(&[high, low]));
}

#[test]
fn empirical_dispersion_retains_subnormal_samples() {
    let tiny = f64::from_bits(1);
    // Exact population std=tiny; multiplying by the N=2 multiplier rounds
    // to tiny, rather than the zero produced by squaring unscaled inputs.
    assert_eq!(empirical_benchmark(&[-tiny, tiny]), tiny);
    // Adjacent tiny values can have an unrepresentable half-ulp std. Use two
    // consecutive even subnormals so the reference std is representable.
    assert_eq!(empirical_benchmark(&[2.0 * tiny, 4.0 * tiny]), tiny);
    let low = 1e-200_f64;
    let high = f64::from_bits(low.to_bits() + 1);
    let expected = (high - low) / 2.0 * 0.5197553442805939;
    let actual = empirical_benchmark(&[low, high]);
    assert!((actual / expected - 1.0).abs() < 1e-12);
}

#[test]
fn empirical_dispersion_preserves_constants_and_nonfinite_behavior() {
    for value in [f64::MAX, -f64::MAX, f64::from_bits(1), 0.0] {
        assert_eq!(empirical_benchmark(&[value, value]), 0.0);
    }
    assert!(empirical_benchmark(&[f64::NAN, 1.0]).is_nan());
    assert!(empirical_benchmark(&[f64::INFINITY, 1.0]).is_nan());
}

#[test]
fn ordinary_empirical_dispersion_keeps_existing_rounding() {
    let values = [0.13, -0.08, 0.42, 0.08, -0.21];
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let sd = (values.iter().map(|value| (value - mean).powi(2)).sum::<f64>() / values.len() as f64)
        .sqrt();
    let parametric =
        deflated_sharpe_ratio(0.0, &[sd, values.len() as f64], 3, 0.0, 3.0, true, true).unwrap();
    assert_eq!(empirical_benchmark(&values), parametric);
}

use openquant::backtest_statistics::deflated_sharpe_ratio;

#[test]
fn expected_maximum_sharpe_preserves_small_upper_tail_probabilities() {
    // Independent scipy.stats.norm.isf reference, using survival probabilities
    // directly instead of subtracting them from one in binary64.
    for (n, expected) in [
        (1e16, 8.290805935009107),
        (1e20, 9.323612882361873),
        (1e100, 21.300497492081014),
        (1e308, 37.55603433890205),
    ] {
        let result = deflated_sharpe_ratio(0.0, &[1.0, n], 3, 0.0, 3.0, true, true).unwrap();
        assert!((result - expected).abs() < 1e-11, "n={n}: {result}");
    }
}

#[test]
fn zero_trial_dispersion_gives_zero_benchmark_for_large_finite_count() {
    let result = deflated_sharpe_ratio(0.0, &[0.0, 1e20], 3, 0.0, 3.0, true, true).unwrap();
    assert_eq!(result, 0.0);
}

#[test]
fn largest_finite_trial_count_has_finite_benchmark() {
    let result = deflated_sharpe_ratio(0.0, &[1.0, f64::MAX], 3, 0.0, 3.0, true, true).unwrap();
    assert!(result.is_finite() && result > 37.0 && result < 38.0);
}

use openquant::backtest_statistics::{minimum_track_record_length, probabilistic_sharpe_ratio};

#[test]
fn psr_retains_finite_large_sharpe_limit() {
    // With zero skew, raw kurtosis 3, and n=3, z tends to ±2.
    // Phi(2), independently evaluated with Python math.erfc.
    // Allow 2e-12 for the statrs normal-CDF approximation.
    let expected = 0.9772498680518208;
    for sr in [1e160, 1e200, 1e308] {
        let actual = probabilistic_sharpe_ratio(sr, 0.0, 3, 0.0, 3.0);
        assert!((actual - expected).abs() < 2e-12, "{sr}: {actual}");
        let actual = probabilistic_sharpe_ratio(-sr, 0.0, 3, 0.0, 3.0);
        assert!((actual - (1.0 - expected)).abs() < 2e-12, "{sr}: {actual}");
    }
}

#[test]
fn psr_handles_opposite_extreme_benchmarks() {
    // (SR - SR*) / sqrt(SR²/2) tends to 2*sqrt(2), n=3 gives z=4.
    let actual = probabilistic_sharpe_ratio(1e308, -1e308, 3, 0.0, 3.0);
    assert!((actual - 0.9999683287581669).abs() < 2e-12, "{actual}");
}

#[test]
fn min_trl_retains_finite_large_sharpe_limit() {
    // 1 + (SR²/2)*(Phi^-1(.95)/SR)² -> 1 + Phi^-1(.95)²/2.
    let expected = 2.352771727047707;
    for sr in [1e160, 1e200, 1e308] {
        let actual = minimum_track_record_length(sr, 0.0, 0.0, 3.0, 0.05).unwrap();
        assert!((actual - expected).abs() < 1e-12, "{sr}: {actual}");
    }
}

#[test]
fn equal_extreme_sharpes_preserve_degenerate_variance_contract() {
    assert_eq!(probabilistic_sharpe_ratio(1e200, 1e200, 3, 0.0, 1.0), 0.5);
    assert_eq!(minimum_track_record_length(1e200, 1e200, 0.0, 1.0, 0.05).unwrap(), f64::INFINITY);
    assert_eq!(probabilistic_sharpe_ratio(1e200, 1e200, 3, 0.0, 3.0), 0.5);
}

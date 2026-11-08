use openquant::backtest_statistics::minimum_track_record_length;

#[test]
fn min_trl_preserves_small_significance_tail() {
    // Independent scipy.stats.norm.isf(alpha) reference; theta=.1, skew=0,kurt=3.
    for (alpha, expected) in [
        (1e-20, 8622.989865878222),
        (1e-100, 45483.26255426008),
        (1e-300, 137935.97809352126),
        (f64::from_bits(1), 148715.0001388477),
    ] {
        let result = minimum_track_record_length(0.1, 0.0, 0.0, 3.0, alpha).unwrap();
        assert!((result / expected - 1.0).abs() < 1e-12, "alpha={alpha}: {result}");
    }
}

#[test]
fn min_trl_retains_endpoint_contract() {
    for alpha in [0.0, 1.0] {
        assert_eq!(minimum_track_record_length(0.1, 0.0, 0.0, 3.0, alpha).unwrap(), f64::INFINITY);
    }
    assert_eq!(minimum_track_record_length(0.1, 0.0, 0.0, 3.0, 0.5).unwrap(), 1.0);
}

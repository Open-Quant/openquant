use openquant::microstructural_features::get_trades_based_amihud_lambda;

#[test]
fn one_trade_amihud_keeps_representable_extreme_slopes() {
    for exponent in [-600, 600] {
        let volume = 2.0_f64.powi(exponent);
        assert_eq!(
            get_trades_based_amihud_lambda(&[-1.0], &[volume]).unwrap(),
            2.0_f64.powi(-exponent)
        );
    }
    let volume = 2.0_f64.powi(-500);
    let log_return = 1.5 * 2.0_f64.powi(-574);
    assert_eq!(
        get_trades_based_amihud_lambda(&[log_return], &[volume]).unwrap(),
        1.5 * 2.0_f64.powi(-74)
    );
    assert_eq!(get_trades_based_amihud_lambda(&[0.0], &[2.0_f64.powi(-600)]).unwrap(), 0.0);
}

#[test]
fn one_trade_amihud_distinguishes_maximum_finite_and_genuine_overflow() {
    // Exact binary64 quotient is below MAX by about 0.325 ulp and rounds to MAX.
    // Squaring the volume underflows, and a separately scaled product ratio
    // can also round this result to infinity.
    assert_eq!(
        get_trades_based_amihud_lambda(&[7.228216483043474e52], &[4.0208288850127243e-256])
            .unwrap(),
        f64::MAX
    );
    assert_eq!(
        get_trades_based_amihud_lambda(&[f64::MAX], &[f64::MIN_POSITIVE]).unwrap(),
        f64::INFINITY
    );
    assert_eq!(get_trades_based_amihud_lambda(&[f64::from_bits(1)], &[f64::MAX]).unwrap(), 0.0);
}

#[test]
fn one_trade_amihud_retains_invalid_and_signed_zero_behavior() {
    assert!(get_trades_based_amihud_lambda(&[1.0], &[0.0]).unwrap().is_nan());
    assert!(get_trades_based_amihud_lambda(&[1.0], &[f64::INFINITY]).unwrap().is_nan());
    assert!(get_trades_based_amihud_lambda(&[f64::NAN], &[1.0]).unwrap().is_nan());
    assert_eq!(get_trades_based_amihud_lambda(&[f64::INFINITY], &[1.0]).unwrap(), f64::INFINITY);
    assert_eq!(get_trades_based_amihud_lambda(&[2.0], &[-1.0]).unwrap(), -2.0);
    assert_eq!(
        get_trades_based_amihud_lambda(&[0.0], &[-1.0]).unwrap().to_bits(),
        (-0.0_f64).to_bits()
    );
}

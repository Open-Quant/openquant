use openquant::microstructural_features::get_trades_based_amihud_lambda;

#[test]
fn amihud_preserves_representable_slopes_across_volume_scales() {
    for exponent in [-600, 600] {
        let scale = 2.0_f64.powi(exponent);
        let actual = get_trades_based_amihud_lambda(&[1.0, -2.0], &[scale, 2.0 * scale]).unwrap();
        assert_eq!(actual, 2.0_f64.powi(-exponent));
    }
    let scale = 2.0_f64.powi(511);
    assert_eq!(get_trades_based_amihud_lambda(&[scale; 4], &[scale; 4]).unwrap(), 1.0);
}

#[test]
fn amihud_recovers_subnormal_product_precision() {
    let volume = 2.0_f64.powi(-500);
    let log_return = 1.5 * 2.0_f64.powi(-574);
    assert_eq!(
        get_trades_based_amihud_lambda(&[log_return], &[volume]).unwrap(),
        1.5 * 2.0_f64.powi(-74)
    );
}

#[test]
fn amihud_does_not_normalize_away_small_cross_products() {
    let volume = [2.0_f64.powi(600), 2.0_f64.powi(-800)];
    let returns = [2.0_f64.powi(-400), 2.0_f64.powi(850)];
    // Exact slope rounds to 2^-1000. Normalizing each vector independently
    // would erase both cross-products before forming the ratio.
    assert_eq!(get_trades_based_amihud_lambda(&returns, &volume).unwrap(), 2.0_f64.powi(-1000));
}

#[test]
fn amihud_retains_zero_invalid_and_ordinary_behavior() {
    assert!(get_trades_based_amihud_lambda(&[], &[]).unwrap().is_nan());
    assert!(get_trades_based_amihud_lambda(&[1.0], &[0.0]).unwrap().is_nan());
    assert!(get_trades_based_amihud_lambda(&[f64::NAN], &[1.0]).unwrap().is_nan());
    assert!(get_trades_based_amihud_lambda(&[1.0], &[f64::INFINITY]).unwrap().is_nan());
    assert_eq!(get_trades_based_amihud_lambda(&[f64::INFINITY], &[1.0]).unwrap(), f64::INFINITY);
    assert_eq!(get_trades_based_amihud_lambda(&[0.0], &[1.0]).unwrap(), 0.0);
    assert_eq!(get_trades_based_amihud_lambda(&[0.0], &[2.0_f64.powi(-600)]).unwrap(), 0.0);
    assert_eq!(get_trades_based_amihud_lambda(&[0.0], &[f64::MAX]).unwrap(), 0.0);
    assert_eq!(get_trades_based_amihud_lambda(&[2.0], &[-1.0]).unwrap(), -2.0);
    assert_eq!(
        get_trades_based_amihud_lambda(&[0.0], &[-1.0]).unwrap().to_bits(),
        (-0.0_f64).to_bits()
    );
    assert_eq!(get_trades_based_amihud_lambda(&[1.0, -2.0], &[1.0, 2.0]).unwrap(), 1.0);
}

#[test]
fn amihud_exact_products_distinguish_maximum_finite_from_overflow() {
    let volume = [4.0208288850127243e-256; 2];
    let returns = [7.228216483043474e52; 2];
    assert_eq!(get_trades_based_amihud_lambda(&returns, &volume).unwrap(), f64::MAX);
    assert_eq!(
        get_trades_based_amihud_lambda(&[returns[0].next_up(); 2], &volume).unwrap(),
        f64::INFINITY
    );
    // Midpoint between MAX and the next mathematical value 2^1024: ties round
    // to infinity. The immediately preceding return gives a finite slope.
    let half_max = f64::MAX / 2.0;
    let boundary_return = 2.0_f64.powi(1023);
    assert_eq!(get_trades_based_amihud_lambda(&[half_max; 2], &[0.5; 2]).unwrap(), f64::MAX);
    assert_eq!(
        get_trades_based_amihud_lambda(&[half_max, boundary_return], &[0.5; 2]).unwrap(),
        f64::INFINITY
    );
}

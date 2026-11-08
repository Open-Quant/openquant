use openquant::microstructural_features::{
    get_trades_based_hasbrouck_lambda, get_trades_based_kyle_lambda,
};

#[test]
fn signed_lambdas_recover_cancellation_with_normal_rounded_sums() {
    let changes = [1e16, -3.0, -1e16];
    let flags = [1.0, -1.0, 1.0];
    assert_eq!(get_trades_based_kyle_lambda(&changes, &[1.0; 3], &flags).unwrap(), 1.0);
    assert_eq!(get_trades_based_hasbrouck_lambda(&changes, &[1.0; 3], &flags).unwrap(), 1.0);
}

#[test]
fn kyle_recovers_extreme_finite_product_sums() {
    for scale in [1e200, 1e-200] {
        assert_eq!(
            get_trades_based_kyle_lambda(&[scale, -0.5 * scale], &[scale; 2], &[1.0; 2]).unwrap(),
            0.25
        );
    }
}

#[test]
fn hasbrouck_recovers_overflow_and_partial_product_underflow() {
    assert_eq!(
        get_trades_based_hasbrouck_lambda(
            &[2.0_f64.powi(1023), -2.0_f64.powi(1022)],
            &[2.0_f64.powi(1022); 2],
            &[1.0; 2],
        )
        .unwrap(),
        2.0_f64.powi(510)
    );
    assert_eq!(
        get_trades_based_hasbrouck_lambda(
            &[2.0_f64.powi(-537), -2.0_f64.powi(-538)],
            &[f64::from_bits(1); 2],
            &[1.0; 2],
        )
        .unwrap(),
        0.25
    );
}

#[test]
fn signed_lambdas_preserve_zero_sign_and_distinguish_exact_cancellation() {
    for values in [
        get_trades_based_kyle_lambda(&[-0.0; 2], &[1.0; 2], &[1.0; 2]),
        get_trades_based_hasbrouck_lambda(&[-0.0; 2], &[1.0; 2], &[1.0; 2]),
        get_trades_based_kyle_lambda(&[-0.0; 2], &[1e-200; 2], &[1.0; 2]),
    ] {
        assert_eq!(values.unwrap().to_bits(), (-0.0_f64).to_bits());
    }
    assert_eq!(
        get_trades_based_kyle_lambda(&[1.0, -1.0], &[1.0; 2], &[1.0; 2]).unwrap().to_bits(),
        0.0_f64.to_bits()
    );
}

#[test]
fn signed_lambdas_keep_arbitrary_flags_and_invalid_input_behavior() {
    assert_eq!(get_trades_based_kyle_lambda(&[3.0, -4.0], &[1.0, 2.0], &[2.0, -0.5]).unwrap(), 2.0);
    assert_eq!(
        get_trades_based_hasbrouck_lambda(&[3.0, -4.0], &[1.0, 4.0], &[2.0, -0.5]).unwrap(),
        2.0
    );
    assert_eq!(get_trades_based_kyle_lambda(&[1.0, 2.0], &[-1.0, -2.0], &[1.0; 2]).unwrap(), -1.0);
    assert!(get_trades_based_hasbrouck_lambda(&[1.0], &[-1.0], &[1.0]).unwrap().is_nan());
    assert!(get_trades_based_kyle_lambda(&[1.0], &[1.0], &[f64::INFINITY]).unwrap().is_nan());
    assert!(get_trades_based_kyle_lambda(&[1.0], &[1.0], &[0.0]).unwrap().is_nan());
    assert!(get_trades_based_kyle_lambda(&[1.0], &[], &[1.0]).is_err());
}

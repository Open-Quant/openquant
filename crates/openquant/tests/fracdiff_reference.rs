//! Value tests for `fracdiff` from closed forms (AFML chapter 5).
//!
//! The pre-existing `tests/fracdiff.rs` asserts lengths, "last weight is 1" and "first output is
//! NaN"; replacing the fixed-width convolution with a constant passes it. See
//! `docs/test-sensitivity-audit.md`.
//!
//! The weights of (1 - B)^d are w_0 = 1, w_k = -w_{k-1} (d - k + 1) / k, i.e. the binomial series
//! (-1)^k C(d, k). The library returns them oldest-first, so w_0 is LAST.

use openquant::fracdiff::{frac_diff, frac_diff_ffd, get_weights, get_weights_ffd};

/// d = 0.5:  w = 1, -1/2, -1/8, -1/16, -5/128   (all dyadic, so exactly representable)
///   w_1 = -1 * 0.5 / 1            = -0.5
///   w_2 = 0.5 * (0.5 - 1) / 2     = -0.125
///   w_3 = 0.125 * (0.5 - 2) / 3   = -0.0625
///   w_4 = 0.0625 * (0.5 - 3) / 4  = -0.0390625
#[test]
fn weights_for_half_difference_closed_form() {
    assert_eq!(get_weights(0.5, 5), vec![-0.0390625, -0.0625, -0.125, -0.5, 1.0]);
}

/// Integer d gives the ordinary finite-difference stencils: d = 1 -> (1, -1, 0, ...),
/// d = 2 -> (1, -2, 1, 0, ...).
#[test]
fn weights_for_integer_orders_are_finite_difference_stencils() {
    assert_eq!(get_weights(1.0, 4), vec![0.0, 0.0, -1.0, 1.0]);
    assert_eq!(get_weights(2.0, 4), vec![0.0, 1.0, -2.0, 1.0]);
}

/// FFD drops weights below the threshold in absolute value. For d = 0.5 the magnitudes are
/// 1, 0.5, 0.125, 0.0625, 0.0390625, 0.02734375, ...: a threshold of 0.05 keeps the first four.
#[test]
fn ffd_weights_are_cut_at_threshold() {
    assert_eq!(get_weights_ffd(0.5, 0.05, 100), vec![-0.0625, -0.125, -0.5, 1.0]);
}

const SERIES: [f64; 7] = [1.0, 4.0, 9.0, 16.0, 25.0, 36.0, 49.0]; // t^2

/// d = 1 is the first difference, d = 2 the second. For x_t = t^2: first differences
/// 3, 5, 7, 9, 11, 13 and second differences 2, 2, 2, 2, 2. The window is 2 (resp. 3) wide, so
/// the first 1 (resp. 2) outputs are NaN. All values are small integers: exact equality.
#[test]
fn ffd_with_integer_order_is_the_ordinary_difference() {
    let d1 = frac_diff_ffd(&SERIES, 1.0, 1e-5);
    assert!(d1[0].is_nan());
    assert_eq!(&d1[1..], &[3.0, 5.0, 7.0, 9.0, 11.0, 13.0]);

    let d2 = frac_diff_ffd(&SERIES, 2.0, 1e-5);
    assert!(d2[0].is_nan() && d2[1].is_nan());
    assert_eq!(&d2[2..], &[2.0, 2.0, 2.0, 2.0, 2.0]);
}

/// d = 0.5, threshold 0.05 (four weights, see above), hand-worked at t = 3 and t = 4:
///   y_3 = 16 - 0.5*9  - 0.125*4 - 0.0625*1 = 10.9375
///   y_4 = 25 - 0.5*16 - 0.125*9 - 0.0625*4 = 15.625
#[test]
fn ffd_half_difference_hand_worked() {
    let y = frac_diff_ffd(&SERIES, 0.5, 0.05);
    assert!(y[..3].iter().all(|v| v.is_nan()));
    assert_eq!(y[3], 10.9375);
    assert_eq!(y[4], 15.625);
}

/// Expanding-window version (snippet 5.2) with d = 1: the weights are (0, ..., 0, -1, 1), the
/// normalised cumulative |w| is (0, ..., 0, 0.5, 1), two entries exceed the 0.01 threshold so two
/// leading outputs are skipped, and the rest is again the first difference.
#[test]
fn expanding_window_with_order_one_is_the_first_difference() {
    let y = frac_diff(&SERIES, 1.0, 0.01);
    assert!(y[0].is_nan() && y[1].is_nan());
    assert_eq!(&y[2..], &[5.0, 7.0, 9.0, 11.0, 13.0]);
}

/// d = 0 is the identity operator: w = (1, 0, 0, ...).
#[test]
fn order_zero_is_identity() {
    let y = frac_diff_ffd(&SERIES, 0.0, 1e-5);
    assert_eq!(y, SERIES.to_vec());
}

//! Extreme finite returns must not turn arithmetic overflow into zero risk.
use chrono::{Duration, NaiveDate};
use openquant::util::volatility::get_daily_vol;

fn estimate(returns: &[f64], span: usize) -> Vec<f64> {
    let anchor = NaiveDate::from_ymd_opt(2024, 1, 1).unwrap().and_hms_opt(0, 0, 0).unwrap();
    let mut close = vec![(anchor, 1.0)];
    // All later bars fall within one day, so their eligible previous close is
    // always the unit-price anchor. This supplies independent, known returns.
    close.extend(
        returns
            .iter()
            .enumerate()
            .map(|(i, &ret)| (anchor + Duration::days(2) + Duration::seconds(i as i64), ret + 1.0)),
    );
    let out = get_daily_vol(&close, span);
    assert_eq!(out.len(), returns.len());
    assert!(out.iter().zip(&close[1..]).all(|((ts, _), (expected, _))| ts == expected));
    out.into_iter().map(|(_, value)| value).collect()
}

fn assert_values(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    for (i, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        if expected.is_nan() {
            assert!(actual.is_nan(), "row {i}: expected NaN, got {actual}");
        } else if expected.is_infinite() {
            assert_eq!(actual, expected, "row {i}");
        } else {
            let tolerance = 1e-12_f64.max(expected.abs() * 1e-14);
            assert!((actual - expected).abs() <= tolerance, "row {i}: {actual} != {expected}");
        }
    }
}

#[test]
fn initial_return_square_overflow_does_not_destroy_finite_variance() {
    for level in [1e153, f64::MAX.sqrt(), f64::MAX.sqrt().next_up()] {
        // Closed-form unbiased weighted sums for weights 1/4, 1/2, 1
        // and 1/8, 1/4, 1/2, 1. Compute std without squaring the level.
        assert_values(
            &estimate(&[level, level, 0.0, 0.0], 3),
            &[f64::NAN, 0.0, level * (3.0_f64 / 7.0).sqrt(), level * (9.0_f64 / 35.0).sqrt()],
        );
    }
}

#[test]
fn overflowing_variance_is_reported_as_infinity_not_zero() {
    for level in [1e155, 1e200, 1e308] {
        // pandas 2.2.3: the f64 variance overflows although std could be
        // represented by a scaled algorithm. It must not claim zero risk.
        assert_values(
            &estimate(&[level, level, 0.0, 0.0], 3),
            &[f64::NAN, 0.0, f64::INFINITY, f64::INFINITY],
        );
    }
}

#[test]
fn identical_extreme_returns_have_exactly_zero_variance() {
    for (level, span) in [(1e150, 4), (1e308, 3), (f64::MAX, 100)] {
        let actual = estimate(&[level; 8], span);
        assert!(actual[0].is_nan());
        assert!(actual[1..].iter().all(|&value| value == 0.0), "{level}, {span}: {actual:?}");
    }
}

#[test]
fn undefined_overflow_arithmetic_stays_nan_instead_of_becoming_zero() {
    // pandas 2.2.3: after weighted mean overflow, Inf - Inf makes variance NaN.
    assert_values(
        &estimate(&[8e307, 9e307, 1.25e308, 1.1111111111111112e308, 0.0, 0.0], 3),
        &[f64::NAN, f64::INFINITY, f64::INFINITY, f64::NAN, f64::NAN, f64::NAN],
    );
}

#[test]
fn first_finite_return_after_missing_values_is_initialized_safely() {
    assert_values(
        &estimate(&[f64::NAN, f64::NAN, 1e155, 1e155, 0.0, 0.0], 3),
        &[f64::NAN, f64::NAN, f64::NAN, 0.0, f64::INFINITY, f64::INFINITY],
    );
}

#[test]
fn underflowed_weights_do_not_reset_overflowed_variance_to_false_finite_risk() {
    let mut returns = vec![1e200, 1e200, 0.0];
    returns.extend(vec![f64::NAN; 1100]);
    returns.extend([0.0, 1.0]);
    let actual = estimate(&returns, 3);
    // A first-observation-only initialization is essential. Resetting when
    // old_wt underflows to zero invents a 0.707 estimate, but a high-precision
    // weighted sum still gives about 1.439e34. pandas remains NaN here.
    assert!(actual[actual.len() - 2..].iter().all(|value| value.is_nan()));
}

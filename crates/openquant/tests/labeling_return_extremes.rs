use chrono::{Duration, NaiveDate};
use openquant::labeling::{get_bins, meta_labels, triple_barrier_labels, Event};

fn assert_return(start: f64, end: f64, side: Option<f64>, expected: f64) {
    let t = NaiveDate::from_ymd_opt(2024, 1, 1).unwrap().and_hms_opt(0, 0, 0).unwrap();
    let next = t + Duration::seconds(1);
    let close = [(t, start), (next, end)];
    let events = [(t, Event { t1: Some(next), trgt: 0.01, side, pt: 0.0, sl: 0.0 })];
    let rows = triple_barrier_labels(&events, &close);
    let bins = get_bins(&events, &close);
    let mut actuals = vec![rows[0].ret, bins[0].1];
    if side.is_some() {
        actuals.push(meta_labels(&events, &close)[0].ret);
    }
    for actual in actuals {
        if expected.is_nan() {
            assert!(actual.is_nan());
        } else {
            assert_eq!(
                actual.to_bits(),
                expected.to_bits(),
                "start={start},end={end},side={side:?}"
            );
        }
    }
    let label = if expected > 0.0 {
        1
    } else if side.is_none() && expected < 0.0 {
        -1
    } else {
        0
    };
    assert_eq!(rows[0].label, label);
    assert_eq!(bins[0].3, label);
}

#[test]
fn finite_side_recovers_intermediate_overflow() {
    assert_return(1e-308, 3.0, Some(1e-308), 3.0);
    assert_return(1e-308, 3.0, Some(-1e-308), -3.0);
    assert_return(1e-308, 3.0, Some(0.0), 0.0);
    assert_return(1e-308, 3.0, Some(-0.0), -0.0);
}

#[test]
fn near_equal_prices_recover_quotient_subtraction_cancellation() {
    // Frozen exact-Fraction binary-input references.
    assert_return(f64::MAX.next_down(), f64::MAX, None, f64::from_bits(0x3ca0000000000001));
    assert_return(f64::MAX.next_down(), f64::MAX, Some(1.0), f64::from_bits(0x3ca0000000000001));
    assert_return(f64::MAX.next_down(), f64::MAX, Some(1e308), f64::from_bits(0x7c91ccf385ebc8a1));
}

#[test]
fn recovery_avoids_naive_reordered_product_overflow_or_underflow() {
    assert_return(
        f64::from_bits(1),
        1e-308,
        Some(f64::from_bits(1)),
        f64::from_bits(0x730d67819e8d1),
    );
    assert_return(f64::MAX / 2.0, f64::MAX, Some(f64::MAX), f64::MAX);
}

#[test]
fn recovery_keeps_subnormal_ties_and_zero_signs() {
    for (side, expected) in [
        (f64::from_bits(1), 0.0),
        (-f64::from_bits(1), -0.0),
        (f64::from_bits(3), f64::from_bits(2)),
        (-f64::from_bits(3), -f64::from_bits(2)),
    ] {
        assert_return(2.0, 3.0, Some(side), expected);
    }
    assert_return(100.0, 100.0, Some(-1.0), -0.0);
    assert_return(100.0, 90.0, Some(0.0), -0.0);
    assert_return(100.0, 90.0, Some(-0.0), 0.0);
}

#[test]
fn genuine_overflow_and_invalid_inputs_keep_their_contract() {
    assert_return(f64::from_bits(1), f64::MAX, None, f64::INFINITY);
    assert_return(f64::from_bits(1), f64::MAX, Some(-1.0), f64::NEG_INFINITY);
    assert_return(0.0, 3.0, Some(0.0), f64::NAN);
    assert_return(-1.0, 3.0, Some(1.0), -4.0);
    assert_return(1.0, 3.0, Some(f64::NAN), f64::NAN);
    assert_return(100.0, 110.0, Some(1.0), 110.0 / 100.0 - 1.0);
}

#[test]
fn corrected_underflow_also_corrects_the_emitted_meta_label() {
    // Exact Fraction reference rounds to zero, while the old double-rounded
    // calculation produces one minimum subnormal and a false positive label.
    assert_return(
        f64::from_bits(0x38c27f5c1326cabb),
        f64::from_bits(0x38c27f5c1326cabc),
        Some(f64::from_bits(0x00093fae0993655d)),
        0.0,
    );
}

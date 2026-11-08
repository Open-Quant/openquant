use chrono::{Duration, NaiveDate};
use openquant::labeling::{triple_barrier_events, TripleBarrierConfig};

fn touched_from(start_price: f64, price: f64, side: f64, target: f64, pt: f64, sl: f64) -> bool {
    let t = NaiveDate::from_ymd_opt(2024, 1, 1).unwrap().and_hms_opt(0, 0, 0).unwrap();
    let next = t + Duration::seconds(1);
    let close = [(t, start_price), (next, price)];
    let events = triple_barrier_events(
        &close,
        &[t],
        &[(t, target)],
        TripleBarrierConfig { pt, sl, min_ret: 0.0, vertical_barrier_times: None },
        Some(&[(t, side)]),
    );
    events[0].1.t1 == Some(next)
}

fn touched(price: f64, side: f64, target: f64, pt: f64, sl: f64) -> bool {
    touched_from(1e-308, price, side, target, pt, sl)
}

#[test]
fn overflowing_return_and_profit_threshold_still_touch() {
    // Exact binary-input Fraction reference:3/float(1e-308)-1 >2*float(1e308).
    assert!(touched(3.0, 1.0, 1e308, 2.0, 0.0));
}

#[test]
fn overflowing_return_and_stop_threshold_still_touch() {
    assert!(touched(3.0, -1.0, 1e308, 0.0, 2.0));
}

#[test]
fn a_small_finite_side_does_not_turn_intermediate_overflow_into_a_touch() {
    // The signed return is near±3, not infinity; neither bound±10 is reached.
    assert!(!touched(3.0, 1e-308, 1.0, 10.0, 0.0));
    assert!(!touched(3.0, -1e-308, 1.0, 0.0, 10.0));
}

#[test]
fn disabled_barriers_and_nonfinite_event_coefficients_keep_legacy_behavior() {
    assert!(!touched(3.0, 1.0, 1e308, 0.0, 0.0));
    assert!(!touched(3.0, 1.0, 1e308, f64::INFINITY, 0.0));
    assert!(!touched(3.0, 1.0, 1e308, 2.0, f64::NAN));
}

#[test]
fn exact_remainders_below_a_threshold_do_not_touch() {
    let tiny = f64::from_bits(1);
    // Exact signed return is±(1-tiny), strictly inside the bounds±1.
    assert!(!touched_from(tiny, 1.0, tiny, 1.0, 1.0, 0.0));
    assert!(!touched_from(tiny, 1.0, -tiny, 1.0, 0.0, 1.0));
}

#[test]
fn negative_and_nonfinite_prices_keep_legacy_comparisons() {
    assert!(touched_from(1.0, f64::INFINITY, f64::MAX, 1.0, 1.0, 1.0));
    assert!(touched_from(1.0, f64::NEG_INFINITY, f64::MAX, 1.0, 1.0, 1.0));
    assert!(touched_from(1.0, -f64::MAX, f64::MAX, 1.0, 1.0, 1.0));
    assert!(!touched_from(1.0, f64::NAN, f64::MAX, 1.0, 1.0, 1.0));
    assert!(touched_from(1.0, f64::MAX, f64::MAX, f64::MAX.next_down(), f64::MAX, 0.0));
}

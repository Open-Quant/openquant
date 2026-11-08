use chrono::{Duration, NaiveDate, NaiveDateTime};
use openquant::labeling::add_vertical_barrier;

fn start() -> NaiveDateTime {
    NaiveDate::from_ymd_opt(2024, 1, 1).unwrap().and_hms_opt(0, 0, 0).unwrap()
}

#[test]
fn huge_positive_horizons_have_no_bar() {
    let t = start();
    let close = [(t, 1.0), (t + Duration::days(1), 1.0)];
    for offsets in
        [(i64::MAX, 0, 0, 0), (0, i64::MAX, 0, 0), (0, 0, i64::MAX, 0), (0, 0, 0, i64::MAX)]
    {
        assert!(add_vertical_barrier(&[t], &close, offsets.0, offsets.1, offsets.2, offsets.3)
            .is_empty());
    }
}

#[test]
fn huge_negative_horizons_select_the_first_bar() {
    let t = start();
    let close = [(t, 1.0), (t + Duration::days(1), 1.0)];
    for offsets in
        [(i64::MIN, 0, 0, 0), (0, i64::MIN, 0, 0), (0, 0, i64::MIN, 0), (0, 0, 0, i64::MIN)]
    {
        assert_eq!(
            add_vertical_barrier(&[t], &close, offsets.0, offsets.1, offsets.2, offsets.3),
            [(t, t)]
        );
    }
}

#[test]
fn large_components_can_cancel_to_a_small_horizon() {
    let t = start();
    let next = t + Duration::seconds(1);
    let close = [(t, 1.0), (next, 1.0)];
    // Individual Chrono durations cannot represent either component.
    let days = i64::MAX / 86400;
    assert_eq!(add_vertical_barrier(&[t], &close, days, 0, 0, -days * 86400 + 1), [(t, next)]);
}

#[test]
fn date_range_overflow_is_resolved_per_event() {
    let max = NaiveDateTime::MAX;
    let min = NaiveDateTime::MIN;
    let close = [(min, 1.0), (max, 1.0)];
    assert_eq!(add_vertical_barrier(&[min, max], &close, 0, 0, 0, 1), [(min, max)]);
    assert_eq!(add_vertical_barrier(&[min, max], &close, 0, 0, 0, -1), [(min, min), (max, max)]);
    assert!(add_vertical_barrier(&[max], &[], i64::MAX, 0, 0, 0).is_empty());
}

#[test]
fn intermediate_duration_overflow_can_cancel_to_one_second() {
    let t = start();
    let next = t + Duration::seconds(1);
    let close = [(t, 1.0), (next, 1.0)];
    // Each duration fits Chrono, but adding the first two exceeds its range.
    assert_eq!(
        add_vertical_barrier(
            &[t],
            &close,
            60_000_000_000,
            1_440_000_000_000,
            -86_400_000_000_000,
            -5_184_000_000_000_000 + 1,
        ),
        [(t, next)]
    );
}

#[test]
fn ordinary_offsets_preserve_nanoseconds_and_leap_second_semantics() {
    let day = NaiveDate::from_ymd_opt(2016, 12, 31).unwrap();
    let leap = day.and_hms_nano_opt(23, 59, 59, 1_500_000_000).unwrap();
    let ordinary = day.and_hms_nano_opt(12, 0, 0, 123_456_789).unwrap();
    for event in [ordinary, leap] {
        let close: Vec<_> =
            (-10..=10).map(|seconds| (event + Duration::seconds(seconds), 1.0)).collect();
        for seconds in -12..=12 {
            // The existing function combines durations before adding to the event,
            // which is significant for Chrono's leap-second representation.
            let delta = Duration::days(1)
                + Duration::hours(-24)
                + Duration::minutes(1)
                + Duration::seconds(seconds - 60);
            let old_target = event + delta;
            let expected: Vec<_> = close
                .iter()
                .find(|(t, _)| *t >= old_target)
                .map(|(t, _)| (event, *t))
                .into_iter()
                .collect();
            assert_eq!(add_vertical_barrier(&[event], &close, 1, -24, 1, seconds - 60), expected);
        }
    }
}

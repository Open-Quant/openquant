use chrono::{Duration, NaiveDate};
use openquant::labeling::{triple_barrier_events, TripleBarrierConfig};

#[test]
fn target_filter_requires_an_ordered_strictly_greater_comparison() {
    let t = NaiveDate::from_ymd_opt(2024, 1, 1).unwrap().and_hms_opt(0, 0, 0).unwrap();
    let times: Vec<_> = (0..4).map(|i| t + Duration::seconds(i)).collect();
    let close: Vec<_> = times.iter().map(|&t| (t, 1.0)).collect();
    let target: Vec<_> =
        times.iter().copied().zip([f64::NEG_INFINITY, 0.01, f64::INFINITY, f64::NAN]).collect();
    for (cutoff, expected_indices) in [
        (f64::NAN, vec![]),
        (0.005, vec![1, 2]),
        (f64::NEG_INFINITY, vec![1, 2]),
        (f64::INFINITY, vec![]),
    ] {
        let actual = triple_barrier_events(
            &close,
            &times,
            &target,
            TripleBarrierConfig { pt: 0.0, sl: 0.0, min_ret: cutoff, vertical_barrier_times: None },
            None,
        );
        let actual_times: Vec<_> = actual.iter().map(|(time, _)| *time).collect();
        let expected_times: Vec<_> = expected_indices.into_iter().map(|i| times[i]).collect();
        assert_eq!(actual_times, expected_times, "cutoff={cutoff}");
    }
}

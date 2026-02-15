use chrono::NaiveDateTime;
use csv::ReaderBuilder;
use openquant::filters::{cusum_filter_timestamps, Threshold};
use openquant::labeling::{
    add_vertical_barrier, drop_labels, get_bins, get_events, meta_labels, triple_barrier_events,
    triple_barrier_labels, Event, TripleBarrierConfig,
};
use openquant::util::volatility::get_daily_vol;
use serde::Deserialize;
use std::path::Path;

#[derive(Debug, Deserialize)]
struct Row {
    date_time: String,
    close: f64,
}

fn fixture_dir() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/filters")
        .canonicalize()
        .expect("fixture dir")
}

fn load_close() -> Vec<(NaiveDateTime, f64)> {
    let path = fixture_dir().join("dollar_bar_sample.csv");
    let mut rdr = ReaderBuilder::new().has_headers(true).from_path(&path).expect("read csv");
    let mut out = Vec::new();
    for result in rdr.deserialize::<Row>() {
        let row = result.expect("row");
        let ts = NaiveDateTime::parse_from_str(&row.date_time, "%Y-%m-%d %H:%M:%S%.f")
            .expect("parse date");
        out.push((ts, row.close));
    }
    out
}

fn events_to_map(
    events: Vec<(NaiveDateTime, Event)>,
) -> std::collections::HashMap<NaiveDateTime, Event> {
    events.into_iter().collect()
}

#[test]
fn test_daily_volatility() {
    let close = load_close();
    let daily_vol = get_daily_vol(&close, 100);
    assert_eq!(daily_vol.len(), 960);
    let last = daily_vol.last().unwrap().1;
    assert!((last - 0.008968238932170641).abs() < 1e-4);

    // tz-localized version should match values
    let tz_close: Vec<_> = close.iter().map(|(ts, price)| (*ts, *price)).collect();
    let tz_vol = get_daily_vol(&tz_close, 100);
    assert_eq!(daily_vol.len(), tz_vol.len());
    for (a, b) in daily_vol.iter().zip(tz_vol.iter()) {
        assert!((a.1 - b.1).abs() < 1e-12);
    }
}

#[test]
fn test_vertical_barriers() {
    let close = load_close();
    let timestamps: Vec<NaiveDateTime> = close.iter().map(|(ts, _)| *ts).collect();
    let cusum_events = cusum_filter_timestamps(
        &close.iter().map(|(_, p)| *p).collect::<Vec<_>>(),
        &timestamps,
        Threshold::Scalar(0.02),
    );

    for days in 1..=5 {
        let vbars = add_vertical_barrier(&cusum_events, &close, days, 0, 0, 0);
        for (start, end) in vbars {
            assert!((end - start).num_days() >= 1, "days={days}");
        }
    }
    for hours in 1..=5 {
        let vbars = add_vertical_barrier(&cusum_events, &close, 0, hours, 0, 0);
        for (start, end) in vbars {
            assert!((end - start).num_seconds() >= 3600, "hours={hours}");
        }
    }
    for minutes in 1..=5 {
        let vbars = add_vertical_barrier(&cusum_events, &close, 0, 0, minutes, 0);
        for (start, end) in vbars {
            assert!((end - start).num_seconds() >= 60, "minutes={minutes}");
        }
    }
    for seconds in 1..=5 {
        let vbars = add_vertical_barrier(&cusum_events, &close, 0, 0, 0, seconds);
        for (start, end) in vbars {
            assert!((end - start).num_seconds() >= 1, "seconds={seconds}");
        }
    }
}

#[test]
fn test_triple_barrier_events() {
    let close = load_close();
    let prices: Vec<f64> = close.iter().map(|(_, p)| *p).collect();
    let timestamps: Vec<NaiveDateTime> = close.iter().map(|(ts, _)| *ts).collect();
    let cusum_events = cusum_filter_timestamps(&prices, &timestamps, Threshold::Scalar(0.02));
    let daily_vol = get_daily_vol(&close, 100);
    let vertical_barriers = add_vertical_barrier(&cusum_events, &close, 1, 0, 0, 0);

    let events = get_events(
        &close,
        &cusum_events,
        (1.0, 1.0),
        &daily_vol,
        0.005,
        3,
        Some(&vertical_barriers),
        None,
    );

    assert_eq!(events.len(), 8);
    assert!((events[0].1.trgt - 0.010166261175903357).abs() < 1e-3);
    assert!((events.last().unwrap().1.trgt - 0.006455887663302871).abs() < 1e-4);
    let expected_index: Vec<NaiveDateTime> = cusum_events.iter().skip(1).copied().collect();
    let event_index: Vec<NaiveDateTime> = events.iter().map(|(ts, _)| *ts).collect();
    assert_eq!(event_index, expected_index);

    // meta-labeling with side=1
    let side: Vec<(NaiveDateTime, f64)> = close.iter().map(|(ts, _)| (*ts, 1.0)).collect();
    let meta_events = get_events(
        &close,
        &cusum_events,
        (1.0, 1.0),
        &daily_vol,
        0.005,
        3,
        Some(&vertical_barriers),
        Some(&side),
    );
    let meta_map = events_to_map(meta_events.clone());
    for (ts, ev) in &events {
        let m = meta_map.get(ts).unwrap();
        assert_eq!(m.t1, ev.t1);
        assert!((m.trgt - ev.trgt).abs() < 1e-12);
    }
    assert_eq!(meta_events.len(), 8);

    // No vertical barriers
    let no_vertical_events =
        get_events(&close, &cusum_events, (1.0, 1.0), &daily_vol, 0.005, 3, None, None);
    assert_eq!(no_vertical_events.len(), 8);
    let diff_count = no_vertical_events
        .iter()
        .filter(|(ts, ev)| {
            let t_with_v = events.iter().find(|(t, _)| t == ts).unwrap();
            t_with_v.1.t1 != ev.t1
        })
        .count();
    assert_eq!(diff_count, 2);
}

#[test]
fn test_triple_barrier_labeling() {
    let close = load_close();
    let prices: Vec<f64> = close.iter().map(|(_, p)| *p).collect();
    let timestamps: Vec<NaiveDateTime> = close.iter().map(|(ts, _)| *ts).collect();
    let cusum_events = cusum_filter_timestamps(&prices, &timestamps, Threshold::Scalar(0.02));
    let daily_vol = get_daily_vol(&close, 100);
    let vertical_barriers = add_vertical_barrier(&cusum_events, &close, 1, 0, 0, 0);

    // standard labeling
    let events = get_events(
        &close,
        &cusum_events,
        (1.0, 1.0),
        &daily_vol,
        0.005,
        3,
        Some(&vertical_barriers),
        None,
    );
    let labels = get_bins(&events, &close);
    assert_eq!(labels.len(), 8);
    assert!(labels.iter().all(|(_, _, _, bin, _)| matches!(bin, -1 | 0 | 1)));

    // meta labeling with side=1
    let side: Vec<(NaiveDateTime, f64)> = close.iter().map(|(ts, _)| (*ts, 1.0)).collect();
    let meta_events = get_events(
        &close,
        &cusum_events,
        (1.0, 1.0),
        &daily_vol,
        0.005,
        3,
        Some(&vertical_barriers),
        Some(&side),
    );
    let meta_labels = get_bins(&meta_events, &close);
    assert_eq!(meta_labels.len(), 8);
    assert!(meta_labels.iter().all(|(_, _, _, bin, _)| matches!(bin, 0 | 1)));
    assert!(meta_labels.iter().any(|(_, _, _, bin, _)| *bin == 1));
}

#[test]
fn test_pt_sl_levels() {
    let close = load_close();
    let prices: Vec<f64> = close.iter().map(|(_, p)| *p).collect();
    let timestamps: Vec<NaiveDateTime> = close.iter().map(|(ts, _)| *ts).collect();
    let cusum_events = cusum_filter_timestamps(&prices, &timestamps, Threshold::Scalar(0.02));
    let target = get_daily_vol(&close, 100);
    let vertical_barriers = add_vertical_barrier(&cusum_events, &close, 1, 0, 0, 0);

    // Very high pt/sl should mostly defer to vertical barriers.
    let events_high = get_events(
        &close,
        &cusum_events,
        (1000.0, 1000.0),
        &target,
        0.005,
        3,
        Some(&vertical_barriers),
        None,
    );
    let vbar_map: std::collections::HashMap<NaiveDateTime, NaiveDateTime> =
        vertical_barriers.iter().copied().collect();
    let high_vertical_hits = events_high
        .iter()
        .filter(|(ts, ev)| ev.t1.is_some() && ev.t1 == vbar_map.get(ts).copied())
        .count();
    assert!(high_vertical_hits >= 6);

    // Very small pt/sl should trigger earlier first-touch exits.
    let events_small = get_events(
        &close,
        &cusum_events,
        (1e-8, 1e-8),
        &target,
        0.005,
        3,
        Some(&vertical_barriers),
        None,
    );
    let small_vertical_hits = events_small
        .iter()
        .filter(|(ts, ev)| ev.t1.is_some() && ev.t1 == vbar_map.get(ts).copied())
        .count();
    assert!(small_vertical_hits < high_vertical_hits);

    let labels_small = get_bins(&events_small, &close);
    assert!(labels_small.iter().all(|(_, _, _, bin, _)| matches!(bin, -1 | 0 | 1)));
}

#[test]
fn test_drop_labels() {
    let close = load_close();
    let prices: Vec<f64> = close.iter().map(|(_, p)| *p).collect();
    let timestamps: Vec<NaiveDateTime> = close.iter().map(|(ts, _)| *ts).collect();
    let cusum_events = cusum_filter_timestamps(&prices, &timestamps, Threshold::Scalar(0.02));
    let target = get_daily_vol(&close, 100);
    let vertical_barriers = add_vertical_barrier(&cusum_events, &close, 1, 0, 0, 0);
    let events = get_events(
        &close,
        &cusum_events,
        (1.0, 1.0),
        &target,
        0.005,
        3,
        Some(&vertical_barriers),
        None,
    );
    let labels = get_bins(&events, &close);

    let dropped = drop_labels(&labels, 0.30);
    assert!(dropped.len() <= labels.len());

    let dropped2 = drop_labels(&labels, 0.20);
    assert!(dropped2.len() <= labels.len());
}

#[test]
fn test_triple_barrier_disabled_barrier_configurations() {
    let close = load_close();
    let prices: Vec<f64> = close.iter().map(|(_, p)| *p).collect();
    let timestamps: Vec<NaiveDateTime> = close.iter().map(|(ts, _)| *ts).collect();
    let cusum_events = cusum_filter_timestamps(&prices, &timestamps, Threshold::Scalar(0.02));
    let target = get_daily_vol(&close, 100);
    let vertical_barriers = add_vertical_barrier(&cusum_events, &close, 1, 0, 0, 0);

    let cfg_none = TripleBarrierConfig {
        pt: 0.0,
        sl: 0.0,
        min_ret: 0.005,
        vertical_barrier_times: Some(&vertical_barriers),
    };
    let events_none = triple_barrier_events(&close, &cusum_events, &target, cfg_none, None);
    assert!(!events_none.is_empty());
    let vbar_map: std::collections::HashMap<NaiveDateTime, NaiveDateTime> =
        vertical_barriers.iter().copied().collect();
    assert!(events_none.iter().all(|(ts, ev)| match vbar_map.get(ts).copied() {
        Some(v) => ev.t1 == Some(v),
        None => ev.t1.is_some(),
    }));

    let labels_none = triple_barrier_labels(&events_none, &close);
    assert!(!labels_none.is_empty());
    assert!(labels_none.iter().all(|row| matches!(row.label, -1 | 0 | 1)));

    let cfg_pt_only = TripleBarrierConfig {
        pt: 1.0,
        sl: 0.0,
        min_ret: 0.005,
        vertical_barrier_times: Some(&vertical_barriers),
    };
    let events_pt_only = triple_barrier_events(&close, &cusum_events, &target, cfg_pt_only, None);
    assert_eq!(events_pt_only.len(), events_none.len());

    let cfg_sl_only = TripleBarrierConfig {
        pt: 0.0,
        sl: 1.0,
        min_ret: 0.005,
        vertical_barrier_times: Some(&vertical_barriers),
    };
    let events_sl_only = triple_barrier_events(&close, &cusum_events, &target, cfg_sl_only, None);
    assert_eq!(events_sl_only.len(), events_none.len());
}

#[test]
fn test_vertical_barrier_first_and_meta_label_regime() {
    let t0 = NaiveDateTime::parse_from_str("2024-01-01 09:30:00", "%Y-%m-%d %H:%M:%S").unwrap();
    let t1 = NaiveDateTime::parse_from_str("2024-01-01 09:31:00", "%Y-%m-%d %H:%M:%S").unwrap();
    let t2 = NaiveDateTime::parse_from_str("2024-01-01 09:32:00", "%Y-%m-%d %H:%M:%S").unwrap();
    let t3 = NaiveDateTime::parse_from_str("2024-01-01 09:33:00", "%Y-%m-%d %H:%M:%S").unwrap();

    let close = vec![(t0, 100.0), (t1, 100.2), (t2, 100.3), (t3, 100.4)];
    let t_events = vec![t0];
    let target = vec![(t0, 0.01)];
    let vertical = vec![(t0, t1)];

    let events = triple_barrier_events(
        &close,
        &t_events,
        &target,
        TripleBarrierConfig {
            pt: 10.0,
            sl: 10.0,
            min_ret: 0.0,
            vertical_barrier_times: Some(&vertical),
        },
        None,
    );
    assert_eq!(events.len(), 1);
    assert_eq!(events[0].1.t1, Some(t1));

    let labels = triple_barrier_labels(&events, &close);
    assert_eq!(labels.len(), 1);
    assert_eq!(labels[0].label, 1);

    let side_prediction = vec![(t0, -1.0)];
    let meta_events = triple_barrier_events(
        &close,
        &t_events,
        &target,
        TripleBarrierConfig {
            pt: 10.0,
            sl: 10.0,
            min_ret: 0.0,
            vertical_barrier_times: Some(&vertical),
        },
        Some(&side_prediction),
    );
    let meta = meta_labels(&meta_events, &close);
    assert_eq!(meta.len(), 1);
    assert!(matches!(meta[0].label, 0 | 1));
    assert_eq!(meta[0].label, 0);
}

#[test]
fn test_meta_label_asymmetric_pt_sl() {
    let t0 = NaiveDateTime::parse_from_str("2024-01-01 09:30:00", "%Y-%m-%d %H:%M:%S").unwrap();
    let t1 = NaiveDateTime::parse_from_str("2024-01-01 09:31:00", "%Y-%m-%d %H:%M:%S").unwrap();
    let t2 = NaiveDateTime::parse_from_str("2024-01-01 09:32:00", "%Y-%m-%d %H:%M:%S").unwrap();
    let t3 = NaiveDateTime::parse_from_str("2024-01-01 09:33:00", "%Y-%m-%d %H:%M:%S").unwrap();

    let close = vec![(t0, 100.0), (t1, 99.0), (t2, 98.0), (t3, 97.0)];
    let t_events = vec![t0];
    let target = vec![(t0, 0.01)];
    let vertical = vec![(t0, t3)];

    // Side=-1 makes a down move profitable; pt/sl are intentionally asymmetric.
    let side_prediction = vec![(t0, -1.0)];
    let events = triple_barrier_events(
        &close,
        &t_events,
        &target,
        TripleBarrierConfig {
            pt: 1.0,
            sl: 100.0,
            min_ret: 0.0,
            vertical_barrier_times: Some(&vertical),
        },
        Some(&side_prediction),
    );
    let labels = meta_labels(&events, &close);
    assert_eq!(labels.len(), 1);
    assert_eq!(labels[0].label, 1);
}

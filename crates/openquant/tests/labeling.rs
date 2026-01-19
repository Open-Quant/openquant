use chrono::NaiveDateTime;
use csv::ReaderBuilder;
use openquant::filters::{cusum_filter_timestamps, Threshold};
use openquant::labeling::{
    add_vertical_barrier, drop_labels, get_bins, get_events, Event,
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
    let mut rdr = ReaderBuilder::new()
        .has_headers(true)
        .from_path(&path)
        .expect("read csv");
    let mut out = Vec::new();
    for result in rdr.deserialize::<Row>() {
        let row = result.expect("row");
        let ts = NaiveDateTime::parse_from_str(&row.date_time, "%Y-%m-%d %H:%M:%S%.f")
            .expect("parse date");
        out.push((ts, row.close));
    }
    out
}

fn events_to_map(events: Vec<(NaiveDateTime, Event)>) -> std::collections::HashMap<NaiveDateTime, Event> {
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
    let tz_close: Vec<_> = close
        .iter()
        .map(|(ts, price)| (*ts, *price))
        .collect();
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
    let cusum_events =
        cusum_filter_timestamps(&close.iter().map(|(_, p)| *p).collect::<Vec<_>>(), &timestamps, Threshold::Scalar(0.02));

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
    let no_vertical_events = get_events(
        &close,
        &cusum_events,
        (1.0, 1.0),
        &daily_vol,
        0.005,
        3,
        None,
        None,
    );
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
    let zero_vertical: Vec<_> = labels
        .iter()
        .filter(|(_, ret, trgt, bin, _)| ret.abs() < *trgt && *bin == 0)
        .collect();
    assert!(!zero_vertical.is_empty());

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
    let cond1: Vec<_> = meta_labels
        .iter()
        .filter(|(_, ret, trgt, bin, _)| *bin == 1 && *ret > 0.0 && ret.abs() > *trgt)
        .collect();
    assert_eq!(meta_labels.len(), 8);
    assert!(!cond1.is_empty());
}

#[test]
fn test_pt_sl_levels() {
    let close = load_close();
    let prices: Vec<f64> = close.iter().map(|(_, p)| *p).collect();
    let timestamps: Vec<NaiveDateTime> = close.iter().map(|(ts, _)| *ts).collect();
    let cusum_events = cusum_filter_timestamps(&prices, &timestamps, Threshold::Scalar(0.02));
    let target = get_daily_vol(&close, 100);
    let vertical_barriers = add_vertical_barrier(&cusum_events, &close, 1, 0, 0, 0);

    // high pt/sl -> all bins zero
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
    let labels_high = get_bins(&events_high, &close);
    let count_ones = labels_high.iter().filter(|(_, _, _, bin, _)| *bin == 1).count();
    assert_eq!(count_ones, 0);

    // very small pt/sl -> no zeros
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
    let labels_small = get_bins(&events_small, &close);
    let zeros = labels_small.iter().filter(|(_, _, _, bin, _)| *bin == 0).count();
    assert_eq!(zeros, 0);

    // tp huge, sl tight -> bins less than 1
    let events_mix = get_events(
        &close,
        &cusum_events,
        (10000.0, 1e-8),
        &target,
        0.005,
        3,
        Some(&vertical_barriers),
        None,
    );
    let labels_mix = get_bins(&events_mix, &close);
    assert!(labels_mix.iter().all(|(_, _, _, bin, _)| *bin <= 0));

    // bins differ from previous scenario
    for i in 0..5 {
        assert_ne!(labels_small[i].3, labels_high[i].3);
    }
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
    assert!(!dropped.iter().any(|(_, _, _, bin, _)| *bin == 0));

    let dropped2 = drop_labels(&labels, 0.20);
    assert!(dropped2.iter().any(|(_, _, _, bin, _)| *bin == 0));
}

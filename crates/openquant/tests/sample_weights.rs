use chrono::NaiveDateTime;
use csv::ReaderBuilder;
use openquant::filters::{cusum_filter_timestamps, Threshold};
use openquant::labeling::{add_vertical_barrier, get_events};
use openquant::sample_weights::{get_weights_by_return, get_weights_by_time_decay};
use openquant::util::volatility::get_daily_vol;
use serde::Deserialize;
use std::path::Path;

#[derive(Debug, Deserialize)]
struct Row {
    date_time: String,
    close: f64,
}

#[derive(Deserialize)]
struct ReferenceEvent {
    t0: String,
    t1: String,
    trgt: f64,
}

#[derive(Deserialize)]
struct Reference {
    events: Vec<ReferenceEvent>,
    weights_by_return: Vec<f64>,
    time_decay: std::collections::HashMap<String, Vec<f64>>,
}

fn load_reference() -> Reference {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/sample_weights/reference.json");
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

fn parse_ts(value: &str) -> NaiveDateTime {
    NaiveDateTime::parse_from_str(value, "%Y-%m-%d %H:%M:%S%.f").unwrap()
}

fn load_close() -> Vec<(NaiveDateTime, f64)> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/filters/dollar_bar_sample.csv")
        .canonicalize()
        .expect("fixture dir");
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

type EventsSetup = (
    Vec<(NaiveDateTime, NaiveDateTime, f64)>,
    Vec<(NaiveDateTime, f64)>,
    Vec<NaiveDateTime>,
    Vec<f64>,
);

fn setup_events() -> EventsSetup {
    let close = load_close();
    let prices: Vec<f64> = close.iter().map(|(_, p)| *p).collect();
    let timestamps: Vec<NaiveDateTime> = close.iter().map(|(ts, _)| *ts).collect();
    let daily_vol = get_daily_vol(&close, 100);
    let cusum_events = cusum_filter_timestamps(&prices, &timestamps, Threshold::Scalar(0.02));
    let vertical_barriers = add_vertical_barrier(&cusum_events, &close, 2, 0, 0, 0);
    let side: Vec<(NaiveDateTime, f64)> = close.iter().map(|(ts, _)| (*ts, 1.0)).collect();
    let events = get_events(
        &close,
        &cusum_events,
        (4.0, 4.0),
        &daily_vol,
        0.005,
        3,
        Some(&vertical_barriers),
        Some(&side),
    );
    let mut events_simple = Vec::new();
    for (ts, ev) in events {
        if let Some(t1) = ev.t1 {
            events_simple.push((ts, t1, ev.trgt));
        }
    }
    (events_simple, close.clone(), timestamps, prices)
}

#[test]
fn test_ret_attribution() {
    let (events, close, _, _) = setup_events();
    let weights = get_weights_by_return(&events, &close).expect("weights");
    assert_eq!(weights.len(), events.len());

    // mlfinlab's test compares these with a tolerance of 1e5, so nothing ever checked them, and
    // this port inherited that. The reference here is the AFML snippets run in pandas
    // (tests/fixtures/sample_weights/generate.py); it reproduces mlfinlab's quoted 0.781807 and
    // 1.627944. Both sides sum the same log returns in f64, so 1e-10 is rounding room only.
    let reference = load_reference();
    assert_eq!(events.len(), reference.events.len());
    for (event, want) in events.iter().zip(&reference.events) {
        assert_eq!(event.0, parse_ts(&want.t0));
        assert_eq!(event.1, parse_ts(&want.t1));
        assert!((event.2 - want.trgt).abs() < 1e-12);
    }
    for (got, want) in weights.iter().zip(&reference.weights_by_return) {
        assert!((got.1 - want).abs() < 1e-10, "got {}, want {want}", got.1);
    }
}

#[test]
fn test_time_decay_weights() {
    let (events, close, _, _) = setup_events();
    let standard = get_weights_by_time_decay(&events, &close, 0.5).expect("standard");
    let no_decay = get_weights_by_time_decay(&events, &close, 1.0).expect("nodecay");
    let neg_decay = get_weights_by_time_decay(&events, &close, -0.5).expect("neg");
    let converge = get_weights_by_time_decay(&events, &close, 0.0).expect("conv");
    let pos_decay = get_weights_by_time_decay(&events, &close, 1.5).expect("pos");

    let len = events.len();
    assert_eq!(standard.len(), len);
    assert_eq!(no_decay.len(), len);
    assert_eq!(neg_decay.len(), len);
    assert_eq!(converge.len(), len);
    assert_eq!(pos_decay.len(), len);

    assert_eq!(standard.last().unwrap().1, 1.0);
    let reference = load_reference();
    for (decay, got) in [
        ("0.5", &standard),
        ("1.0", &no_decay),
        ("-0.5", &neg_decay),
        ("0.0", &converge),
        ("1.5", &pos_decay),
    ] {
        let want = &reference.time_decay[decay];
        assert_eq!(got.len(), want.len());
        for (g, w) in got.iter().zip(want) {
            assert!((g.1 - w).abs() < 1e-10, "decay {decay}: got {}, want {w}", g.1);
        }
    }
    assert!(no_decay.iter().all(|(_, w)| (*w - 1.0).abs() < 1e-12));
    assert_eq!(neg_decay.iter().filter(|(_, w)| *w == 0.0).count(), 3);
    assert_eq!(
        pos_decay.first().unwrap().1,
        pos_decay.iter().map(|(_, w)| *w).fold(f64::MIN, f64::max)
    );
    assert!(pos_decay[pos_decay.len() - 2].1 >= pos_decay.last().unwrap().1);
}

#[test]
fn test_value_error_raise() {
    let (mut events, close, _, _) = setup_events();
    // Introduce NaN via zero timestamp to trigger validation
    events[0].0 = chrono::DateTime::<chrono::Utc>::from_timestamp(0, 0).unwrap().naive_utc();
    assert!(get_weights_by_return(&events, &close).is_err());
    assert!(get_weights_by_time_decay(&events, &close, 0.5).is_err());
}

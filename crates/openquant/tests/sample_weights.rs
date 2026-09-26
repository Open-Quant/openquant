use chrono::NaiveDateTime;
use csv::ReaderBuilder;
use openquant::filters::{cusum_filter_timestamps, Threshold};
use openquant::labeling::{add_vertical_barrier, get_events};
use openquant::sample_weights::{
    get_weights_by_return, get_weights_by_time_decay, SampleWeightsError,
};
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
        .join("../../tests/fixtures/shared/dollar_bar_sample.csv")
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
    let cusum_events =
        cusum_filter_timestamps(&prices, &timestamps, Threshold::Scalar(0.02)).unwrap();
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

    let len = events.len();
    assert_eq!(standard.len(), len);
    assert_eq!(no_decay.len(), len);
    assert_eq!(neg_decay.len(), len);
    assert_eq!(converge.len(), len);

    assert_eq!(standard.last().unwrap().1, 1.0);
    let reference = load_reference();
    for (decay, got) in
        [("0.5", &standard), ("1.0", &no_decay), ("-0.5", &neg_decay), ("0.0", &converge)]
    {
        let want = &reference.time_decay[decay];
        assert_eq!(got.len(), want.len());
        for (g, w) in got.iter().zip(want) {
            assert!((g.1 - w).abs() < 1e-10, "decay {decay}: got {}, want {w}", g.1);
        }
    }
    assert!(no_decay.iter().all(|(_, w)| (*w - 1.0).abs() < 1e-12));
    assert_eq!(neg_decay.iter().filter(|(_, w)| *w == 0.0).count(), 3);
}

/// #186 item 20: `decay` was not checked against AFML's domain `(-1, 1]`. The mlfinlab
/// reference includes `decay = 1.5`, which weights old events *more* than new ones, and
/// `decay = -1` divided by zero.
#[test]
fn test_time_decay_rejects_decay_outside_afml_domain() {
    let (events, close, _, _) = setup_events();
    for decay in [1.5, -1.0, -2.0, f64::NAN, f64::INFINITY] {
        let err = get_weights_by_time_decay(&events, &close, decay).unwrap_err();
        assert!(
            matches!(err, SampleWeightsError::InvalidDecay(d) if d.to_bits() == decay.to_bits()),
            "{decay}: {err:?}"
        );
    }
    // The edges of the domain: just above -1, and exactly 1.
    assert!(get_weights_by_time_decay(&events, &close, -0.999).is_ok());
    assert!(get_weights_by_time_decay(&events, &close, 1.0).is_ok());
}

#[test]
fn test_value_error_raise() {
    let (mut events, close, _, _) = setup_events();
    // An event that ends before it starts has no span to weight.
    events[1].1 = events[1].0 - chrono::Duration::seconds(1);
    let want = SampleWeightsError::EndBeforeStart { index: 1 };
    assert_eq!(get_weights_by_return(&events, &close).unwrap_err(), want);
    assert_eq!(get_weights_by_time_decay(&events, &close, 0.5).unwrap_err(), want);
    assert_eq!(want.to_string(), "event 1 ends before it starts");
}

/// Six one-minute bars and three events, two of which start on the same bar (issue #91).
type Event = (NaiveDateTime, NaiveDateTime, f64);

fn shared_start_setup() -> (Vec<(NaiveDateTime, f64)>, [Event; 3]) {
    let open = chrono::NaiveDate::from_ymd_opt(2024, 1, 2).unwrap().and_hms_opt(9, 30, 0).unwrap();
    let close: Vec<_> = [100.0, 101.0, 102.0, 101.0, 100.0, 103.0]
        .iter()
        .enumerate()
        .map(|(i, p)| (open + chrono::Duration::minutes(i as i64), *p))
        .collect();
    let at = |i: usize| close[i].0;
    let events = [(at(0), at(2), 1.0), (at(0), at(3), 1.0), (at(2), at(4), 1.0)];
    (close, events)
}

#[test]
fn test_time_decay_keeps_one_weight_per_event_when_starts_coincide() {
    let (close, [a, b, c]) = shared_start_setup();
    // Concurrency per bar is [2, 2, 3, 2, 1, 0], so the average uniqueness of A, B and C is
    // 4/9, 11/24 and 11/18. In start order, with the tie between A and B broken by input order,
    // the cumulative uniqueness is 32/72, 65/72 and 109/72. With decay 0.5 the line through
    // (109/72, 1) has slope 0.5 / (109/72) = 36/109 and intercept 0.5.
    let weight = |x: f64| 0.5 + 36.0 / 109.0 * x;
    let (w_a, w_b, w_c) = (weight(32.0 / 72.0), weight(65.0 / 72.0), 1.0);

    let got = get_weights_by_time_decay(&[a, b, c], &close, 0.5).expect("weights");
    assert_eq!(got.len(), 3);
    assert_eq!(got.iter().map(|(ts, _)| *ts).collect::<Vec<_>>(), vec![a.0, b.0, c.0]);
    for (g, w) in got.iter().zip([w_a, w_b, w_c]) {
        assert!((g.1 - w).abs() < 1e-12, "got {}, want {w}", g.1);
    }

    // Reordering the input reorders the output with it: each event keeps its own weight.
    let got = get_weights_by_time_decay(&[c, a, b], &close, 0.5).expect("weights");
    assert_eq!(got.iter().map(|(ts, _)| *ts).collect::<Vec<_>>(), vec![c.0, a.0, b.0]);
    for (g, w) in got.iter().zip([w_c, w_a, w_b]) {
        assert!((g.1 - w).abs() < 1e-12, "got {}, want {w}", g.1);
    }

    // Tied starts take cumulative positions in input order, so listing B before A puts B first:
    // B sits at 11/24 = 33/72 and A at 33/72 + 4/9 = 65/72.
    let got = get_weights_by_time_decay(&[b, a, c], &close, 0.5).expect("weights");
    assert_eq!(got.len(), 3);
    for (g, w) in got.iter().zip([weight(33.0 / 72.0), weight(65.0 / 72.0), 1.0]) {
        assert!((g.1 - w).abs() < 1e-12, "got {}, want {w}", g.1);
    }
}

#[test]
fn test_return_attribution_keeps_one_weight_per_event_when_starts_coincide() {
    let (close, events) = shared_start_setup();
    let got = get_weights_by_return(&events, &close).expect("weights");
    let starts: Vec<_> = events.iter().map(|e| e.0).collect();
    assert_eq!(got.iter().map(|(ts, _)| *ts).collect::<Vec<_>>(), starts);
}

#[test]
fn test_unix_epoch_is_an_ordinary_timestamp() {
    let epoch = chrono::DateTime::<chrono::Utc>::from_timestamp(0, 0).unwrap().naive_utc();
    let close: Vec<_> =
        (0..4).map(|i| (epoch + chrono::Duration::days(i), 100.0 + i as f64)).collect();
    let events = [(close[0].0, close[1].0, 1.0), (close[2].0, close[3].0, 1.0)];
    assert_eq!(get_weights_by_return(&events, &close).expect("return").len(), 2);
    assert_eq!(get_weights_by_time_decay(&events, &close, 0.5).expect("decay").len(), 2);
}

use chrono::NaiveDateTime;
use csv::ReaderBuilder;
use openquant::filters::{
    cusum_filter_indices, cusum_filter_timestamps, z_score_filter_indices, z_score_filter_timestamps,
    Threshold,
};
use serde::Deserialize;
use serde_json::Value;
use std::fs::File;
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

fn load_data() -> (Vec<NaiveDateTime>, Vec<f64>) {
    let path = fixture_dir().join("dollar_bar_sample.csv");
    let mut rdr = ReaderBuilder::new()
        .has_headers(true)
        .from_path(&path)
        .expect("read csv");
    let mut timestamps = Vec::new();
    let mut close = Vec::new();
    for result in rdr.deserialize::<Row>() {
        let row = result.expect("row");
        let ts = NaiveDateTime::parse_from_str(&row.date_time, "%Y-%m-%d %H:%M:%S%.f")
            .expect("parse date");
        timestamps.push(ts);
        close.push(row.close);
    }
    (timestamps, close)
}

fn load_events() -> Value {
    let path = fixture_dir().join("events.json");
    let file = File::open(path).expect("events.json");
    serde_json::from_reader(file).expect("parse events")
}

fn parse_ts(strings: &[Value]) -> Vec<NaiveDateTime> {
    strings
        .iter()
        .map(|v| {
            v.as_str()
                .and_then(|s| NaiveDateTime::parse_from_str(s, "%Y-%m-%dT%H:%M:%S%.f").ok())
                .expect("timestamp")
        })
        .collect()
}

fn as_timestamps(indices: &[usize], timestamps: &[NaiveDateTime]) -> Vec<NaiveDateTime> {
    indices
        .iter()
        .map(|i| *timestamps.get(*i).expect("timestamp idx"))
        .collect()
}

#[test]
fn cusum_filter_matches_fixture() {
    let (timestamps, close) = load_data();
    let events = load_events();
    let thresholds = [0.005, 0.007, 0.01, 0.015, 0.02, 0.03, 0.04];

    for thresh in thresholds.iter() {
        let key_ts = format!("{thresh}_timestamps");
        let key_idx = format!("{thresh}_index");
        let expected_ts = parse_ts(
            events["cusum"][&key_ts]
                .as_array()
                .expect("cusum timestamps"),
        );
        let expected_idx = parse_ts(
            events["cusum"][&key_idx]
                .as_array()
                .expect("cusum indices as timestamps"),
        );

        let got_ts = cusum_filter_timestamps(
            &close,
            &timestamps,
            Threshold::Scalar(*thresh),
        );
        let got_idx = as_timestamps(
            &cusum_filter_indices(&close, Threshold::Scalar(*thresh)),
            &timestamps,
        );

        assert_eq!(got_ts, expected_ts, "threshold={thresh} timestamps");
        assert_eq!(got_idx, expected_idx, "threshold={thresh} indices");
    }
}

#[test]
fn cusum_dynamic_threshold_matches_fixture() {
    let (timestamps, close) = load_data();
    let events = load_events();
    let expected_ts = parse_ts(
        events["cusum_dynamic"]["timestamps"]
            .as_array()
            .expect("cusum_dynamic timestamps"),
    );
    let expected_idx = parse_ts(
        events["cusum_dynamic"]["index"]
            .as_array()
            .expect("cusum_dynamic index as timestamps"),
    );
    let dyn_threshold: Vec<f64> = close.iter().map(|v| v * 1e-5).collect();

    let got_ts = cusum_filter_timestamps(
        &close,
        &timestamps,
        Threshold::Dynamic(dyn_threshold.clone()),
    );
    let got_idx = as_timestamps(
        &cusum_filter_indices(&close, Threshold::Dynamic(dyn_threshold)),
        &timestamps,
    );

    assert_eq!(got_ts, expected_ts, "dynamic threshold timestamps");
    assert_eq!(got_idx, expected_idx, "dynamic threshold indices");
}

#[test]
fn z_score_filter_matches_fixture() {
    let (timestamps, close) = load_data();
    let events = load_events();
    let expected_ts = parse_ts(
        events["z_score"]["timestamps"]
            .as_array()
            .expect("z_score timestamps"),
    );
    let expected_idx = parse_ts(
        events["z_score"]["index"]
            .as_array()
            .expect("z_score index as timestamps"),
    );

    let got_ts = z_score_filter_timestamps(&close, &timestamps, 100, 100, 2.0);
    let got_idx = as_timestamps(
        &z_score_filter_indices(&close, 100, 100, 2.0),
        &timestamps,
    );

    assert_eq!(got_ts, expected_ts, "z-score timestamps");
    assert_eq!(got_idx, expected_idx, "z-score indices");
}

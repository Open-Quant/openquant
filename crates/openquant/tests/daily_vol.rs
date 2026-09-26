//! `get_daily_vol` against AFML snippet 3.1 run in pandas. The reference and the script that
//! produced it are in `tests/fixtures/volatility/`; nothing in it comes from this library.
use chrono::NaiveDateTime;
use csv::ReaderBuilder;
use openquant::util::volatility::get_daily_vol;
use serde::Deserialize;
use std::path::Path;

#[derive(Deserialize)]
struct Row {
    date_time: String,
    close: f64,
}

#[derive(Deserialize)]
struct Sample {
    timestamp: String,
    value: Option<f64>,
}

#[derive(Deserialize)]
struct Reference {
    lookback: usize,
    n_values: usize,
    first_is_nan: bool,
    mean_excluding_nan: f64,
    samples: Vec<Sample>,
}

const TS: &str = "%Y-%m-%d %H:%M:%S%.f";

fn fixtures() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures")
}

fn load_close() -> Vec<(NaiveDateTime, f64)> {
    let mut rdr = ReaderBuilder::new()
        .has_headers(true)
        .from_path(fixtures().join("shared/dollar_bar_sample.csv"))
        .unwrap();
    rdr.deserialize::<Row>()
        .map(|row| {
            let row = row.unwrap();
            (NaiveDateTime::parse_from_str(&row.date_time, TS).unwrap(), row.close)
        })
        .collect()
}

#[test]
fn test_daily_vol_matches_afml_snippet_3_1_in_pandas() {
    let reference: Reference = serde_json::from_str(
        &std::fs::read_to_string(fixtures().join("volatility/daily_vol_reference.json")).unwrap(),
    )
    .unwrap();
    let vol = get_daily_vol(&load_close(), reference.lookback);

    // Row for row with pandas, including the NaN for the first observation (one point has no
    // sample variance).
    assert_eq!(vol.len(), reference.n_values);
    assert_eq!(vol[0].1.is_nan(), reference.first_is_nan);
    assert!(vol[1..].iter().all(|(_, v)| v.is_finite()));

    // Both sides evaluate the same closed-form weighted sums in f64; 1e-12 allows for the
    // different order of accumulation and nothing more.
    for sample in reference.samples.iter().filter(|s| s.value.is_some()) {
        let ts = NaiveDateTime::parse_from_str(&sample.timestamp, TS).unwrap();
        let got = vol.iter().find(|(t, _)| *t == ts).map(|(_, v)| *v).expect("timestamp present");
        let want = sample.value.unwrap();
        assert!((got - want).abs() < 1e-12, "{ts}: got {got}, want {want}");
    }

    let finite: Vec<f64> = vol.iter().map(|(_, v)| *v).filter(|v| v.is_finite()).collect();
    let mean = finite.iter().sum::<f64>() / finite.len() as f64;
    assert!((mean - reference.mean_excluding_nan).abs() < 1e-12, "mean {mean}");
}

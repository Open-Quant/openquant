//! `get_daily_vol` against AFML snippet 3.1 run in pandas. The reference and the script that
//! produced it are in `tests/fixtures/volatility/`; nothing in it comes from this library.
use chrono::{Duration, NaiveDate, NaiveDateTime};
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

fn assert_daily_values(prices: &[f64], expected: &[f64]) {
    let start = NaiveDate::from_ymd_opt(2024, 1, 1).unwrap().and_hms_opt(0, 0, 0).unwrap();
    let close: Vec<_> = prices
        .iter()
        .enumerate()
        .map(|(i, &price)| (start + Duration::days(i as i64), price))
        .collect();
    let actual = get_daily_vol(&close, 3);
    assert_eq!(actual.len(), expected.len());
    for (i, ((timestamp, value), &want)) in actual.iter().zip(expected).enumerate() {
        assert_eq!(*timestamp, close[i + 2].0);
        if want.is_nan() {
            assert!(value.is_nan(), "row {i}: expected NaN, got {value}");
        } else {
            assert!((value - want).abs() < 1e-12, "row {i}: expected {want}, got {value}");
        }
    }
}

// Golden values from pandas 2.2.3: Series(prices[2:] / prices[:-2] - 1)
// .ewm(span=3).std(), independently checked with unbiased finite-observation
// weighted sums. Missing positions still decay older weights (ignore_na=False).
#[test]
fn daily_vol_recovers_after_a_missing_price() {
    assert_daily_values(
        &[100.0, 110.0, 120.0, f64::NAN, 150.0, 140.0, 170.0, 180.0],
        &[
            f64::NAN,
            f64::NAN,
            0.03535533905932741,
            0.03535533905932741,
            0.07528605062021397,
            0.09357397530988633,
        ],
    );
}

#[test]
fn daily_vol_waits_for_two_finite_returns_after_leading_missing_prices() {
    assert_daily_values(
        &[f64::NAN, f64::NAN, 100.0, 110.0, 120.0, 130.0, 140.0],
        &[f64::NAN, f64::NAN, f64::NAN, 0.012856486930664428, 0.015745916432444287],
    );
}

#[test]
fn daily_vol_carries_the_last_estimate_across_missing_returns() {
    assert_daily_values(
        &[100.0, 110.0, 120.0, 125.0, f64::NAN, 150.0, 160.0, 170.0],
        &[
            f64::NAN,
            0.044997704257325646,
            0.044997704257325646,
            0.037440353071835054,
            0.037440353071835054,
            0.0423121471969789,
        ],
    );
}

#[test]
fn daily_vol_is_undefined_when_every_return_is_missing() {
    assert_daily_values(&[f64::NAN; 6], &[f64::NAN; 4]);
}

#[test]
fn daily_vol_treats_infinite_returns_as_missing_like_pandas() {
    assert_daily_values(
        &[100.0, 110.0, 120.0, f64::INFINITY, 150.0, 140.0, 170.0, 180.0],
        &[
            f64::NAN,
            f64::NAN,
            0.03535533905932741,
            0.8361845163924918,
            0.6709403362030547,
            0.5297712044961865,
        ],
    );
}

#[test]
fn daily_vol_does_not_underflow_to_zero_across_a_long_missing_gap() {
    let mut prices = vec![100.0, 101.0, 102.0, 103.0];
    prices.extend(vec![f64::NAN; 530]);
    // pandas 2.2.3 carries the unbiased two-observation estimate across this
    // gap. Weight squares are subnormal near the end, but their ratio is not.
    let expected = ((102.0_f64 / 100.0 - 103.0 / 101.0).powi(2) / 2.0).sqrt();
    let mut values = vec![expected; prices.len() - 2];
    values[0] = f64::NAN;
    assert_daily_values(&prices, &values);
}

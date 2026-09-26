//! Input validation of `pipeline::run_mid_frequency_pipeline` (#186 item 22).
//!
//! Each case used to pass validation: a NaN or infinite CUSUM threshold surfaced as
//! `NoEvents`, a non-positive step size left bet sizes unrounded, out-of-range probabilities
//! were sized as given, and `asset_prices` could have any number of rows.

use chrono::{Duration, NaiveDate, NaiveDateTime};
use nalgebra::DMatrix;
use openquant::pipeline::{
    run_mid_frequency_pipeline, PipelineError, ResearchPipelineConfig, ResearchPipelineInput,
};

struct Data {
    timestamps: Vec<NaiveDateTime>,
    close: Vec<f64>,
    probs: Vec<f64>,
    asset_prices: DMatrix<f64>,
    asset_names: Vec<String>,
}

fn data() -> Data {
    let t0 = NaiveDate::from_ymd_opt(2024, 1, 1).unwrap().and_hms_opt(0, 0, 0).unwrap();
    Data {
        timestamps: (0..6).map(|d| t0 + Duration::days(d)).collect(),
        close: vec![100.0, 110.0, 110.0, 99.0, 99.0, 108.9],
        probs: vec![0.5, 0.9, 0.5, 0.9, 0.5, 0.5],
        asset_prices: DMatrix::from_column_slice(6, 1, &[100.0, 101.0, 102.0, 101.5, 103.0, 104.0]),
        asset_names: vec!["A".to_string()],
    }
}

fn run(d: &Data, config: &ResearchPipelineConfig) -> Result<(), PipelineError> {
    let input = ResearchPipelineInput {
        timestamps: &d.timestamps,
        close: &d.close,
        model_probabilities: &d.probs,
        model_sides: None,
        asset_prices: &d.asset_prices,
        asset_names: &d.asset_names,
    };
    run_mid_frequency_pipeline(input, config).map(|_| ())
}

fn config() -> ResearchPipelineConfig {
    ResearchPipelineConfig { cusum_threshold: 0.05, ..Default::default() }
}

#[test]
fn valid_input_runs() {
    run(&data(), &config()).unwrap();
}

#[test]
fn non_finite_cusum_threshold_is_invalid_not_no_events() {
    for h in [f64::NAN, f64::INFINITY, 0.0, -0.01] {
        let cfg = ResearchPipelineConfig { cusum_threshold: h, ..config() };
        assert_eq!(
            run(&data(), &cfg).unwrap_err(),
            PipelineError::InvalidParameter("cusum_threshold must be finite and > 0"),
            "{h}"
        );
    }
}

#[test]
fn non_positive_or_non_finite_step_size_is_invalid() {
    for step in [0.0, -0.1, f64::NAN, f64::INFINITY] {
        let cfg = ResearchPipelineConfig { step_size: step, ..config() };
        assert_eq!(
            run(&data(), &cfg).unwrap_err(),
            PipelineError::InvalidParameter("step_size must be finite and > 0"),
            "{step}"
        );
    }
}

#[test]
fn out_of_range_model_probabilities_are_invalid() {
    // Bar 0 is never an event, so its probability used to be ignored entirely.
    for (bar, p) in [(0, 1.5), (1, -0.1), (3, f64::NAN)] {
        let mut d = data();
        d.probs[bar] = p;
        assert_eq!(
            run(&d, &config()).unwrap_err(),
            PipelineError::InvalidParameter("model_probabilities must be in [0, 1]"),
            "bar {bar}: {p}"
        );
    }
}

#[test]
fn asset_prices_must_have_one_row_per_bar() {
    let mut d = data();
    d.asset_prices = DMatrix::from_column_slice(4, 1, &[100.0, 101.0, 102.0, 103.0]);
    assert_eq!(
        run(&d, &config()).unwrap_err(),
        PipelineError::LengthMismatch("asset_prices.nrows", 4, "timestamps", 6)
    );
}

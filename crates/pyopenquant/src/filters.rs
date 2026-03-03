use openquant::filters::Threshold;
use pyo3::prelude::*;

use crate::helpers::{format_naive_datetimes, parse_naive_datetimes};

#[pyfunction(name = "cusum_filter_indices")]
fn filters_cusum_filter_indices(close: Vec<f64>, threshold: f64) -> Vec<usize> {
    openquant::filters::cusum_filter_indices(&close, Threshold::Scalar(threshold))
}

#[pyfunction(name = "cusum_filter_timestamps")]
fn filters_cusum_filter_timestamps(
    close: Vec<f64>,
    timestamps: Vec<String>,
    threshold: f64,
) -> PyResult<Vec<String>> {
    let ts = parse_naive_datetimes(timestamps)?;
    if close.len() != ts.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "close/timestamps length mismatch: {} vs {}",
            close.len(),
            ts.len()
        )));
    }
    let out =
        openquant::filters::cusum_filter_timestamps(&close, &ts, Threshold::Scalar(threshold));
    Ok(format_naive_datetimes(out))
}

#[pyfunction(name = "z_score_filter_indices")]
fn filters_z_score_filter_indices(
    close: Vec<f64>,
    mean_window: usize,
    std_window: usize,
    threshold: f64,
) -> Vec<usize> {
    openquant::filters::z_score_filter_indices(&close, mean_window, std_window, threshold)
}

#[pyfunction(name = "z_score_filter_timestamps")]
fn filters_z_score_filter_timestamps(
    close: Vec<f64>,
    timestamps: Vec<String>,
    mean_window: usize,
    std_window: usize,
    threshold: f64,
) -> PyResult<Vec<String>> {
    let ts = parse_naive_datetimes(timestamps)?;
    if close.len() != ts.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "close/timestamps length mismatch: {} vs {}",
            close.len(),
            ts.len()
        )));
    }
    let out = openquant::filters::z_score_filter_timestamps(
        &close,
        &ts,
        mean_window,
        std_window,
        threshold,
    );
    Ok(format_naive_datetimes(out))
}

pub fn register(py: Python<'_>, parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(py, "filters")?;
    m.add_function(wrap_pyfunction!(filters_cusum_filter_indices, &m)?)?;
    m.add_function(wrap_pyfunction!(filters_cusum_filter_timestamps, &m)?)?;
    m.add_function(wrap_pyfunction!(filters_z_score_filter_indices, &m)?)?;
    m.add_function(wrap_pyfunction!(filters_z_score_filter_timestamps, &m)?)?;
    parent.add_submodule(&m)?;
    parent.add("filters", m)?;
    Ok(())
}

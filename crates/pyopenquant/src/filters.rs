use openquant::filters::Threshold;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyString};

use crate::helpers::{format_naive_datetimes, parse_naive_datetimes, to_py_err};

/// Reads the CUSUM `threshold` argument: a number (one threshold for every bar) or a sequence
/// of per-bar thresholds aligned with `close` (a list, NumPy array or pandas Series).
///
/// A per-bar vector must have one value per price. Element 0 is never read by the filter
/// (the first return is at bar 1), so it may be anything, including NaN, as the first value
/// of a volatility estimate usually is; every other element must be finite and non-negative.
/// A scalar must be finite and non-negative too: a NaN threshold silently never fires, and a
/// negative one fires on every bar.
fn cusum_threshold(threshold: &Bound<'_, PyAny>, n_close: usize) -> PyResult<Threshold> {
    // Iterables first: a length-1 NumPy array would otherwise convert to a float. Iterating
    // (rather than extracting a `Vec`) also accepts arrays and Series, which are not
    // registered as `collections.abc.Sequence`. A 0-d array is not iterable and reads as a
    // number below.
    if threshold.is_instance_of::<PyString>() || threshold.is_instance_of::<PyBytes>() {
        return Err(PyTypeError::new_err(
            "threshold must be a number or a sequence of per-bar numbers, not a string",
        ));
    }
    if let Ok(iter) = threshold.try_iter() {
        let values = iter
            .map(|item| item.and_then(|v| v.extract::<f64>()))
            .collect::<PyResult<Vec<f64>>>()
            .map_err(|_| PyTypeError::new_err("per-bar thresholds must all be numbers"))?;
        if values.len() != n_close {
            return Err(PyValueError::new_err(format!(
                "threshold has {} values but close has {}; pass one threshold per bar",
                values.len(),
                n_close
            )));
        }
        if let Some((i, v)) =
            values.iter().enumerate().skip(1).find(|(_, v)| !v.is_finite() || **v < 0.0)
        {
            return Err(PyValueError::new_err(format!(
                "threshold[{i}] = {v}; per-bar thresholds after element 0 must be finite and \
                 non-negative"
            )));
        }
        return Ok(Threshold::Dynamic(values));
    }
    let value: f64 = threshold.extract().map_err(|_| {
        PyTypeError::new_err("threshold must be a number or a sequence of per-bar numbers")
    })?;
    if !value.is_finite() || value < 0.0 {
        return Err(PyValueError::new_err(format!(
            "threshold must be finite and non-negative, got {value}"
        )));
    }
    Ok(Threshold::Scalar(value))
}

#[pyfunction(name = "cusum_filter_indices")]
fn filters_cusum_filter_indices(
    close: Vec<f64>,
    threshold: &Bound<'_, PyAny>,
) -> PyResult<Vec<usize>> {
    let threshold = cusum_threshold(threshold, close.len())?;
    openquant::filters::cusum_filter_indices(&close, threshold).map_err(to_py_err)
}

#[pyfunction(name = "cusum_filter_timestamps")]
fn filters_cusum_filter_timestamps(
    close: Vec<f64>,
    timestamps: Vec<String>,
    threshold: &Bound<'_, PyAny>,
) -> PyResult<Vec<String>> {
    let ts = parse_naive_datetimes(timestamps)?;
    if close.len() != ts.len() {
        return Err(PyValueError::new_err(format!(
            "close/timestamps length mismatch: {} vs {}",
            close.len(),
            ts.len()
        )));
    }
    let threshold = cusum_threshold(threshold, close.len())?;
    let out =
        openquant::filters::cusum_filter_timestamps(&close, &ts, threshold).map_err(to_py_err)?;
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
        return Err(PyValueError::new_err(format!(
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
    )
    .map_err(to_py_err)?;
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

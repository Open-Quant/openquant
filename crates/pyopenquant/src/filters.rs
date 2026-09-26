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

/// Symmetric CUSUM filter returning event positions (AFML section 2.5.2.1, Snippet 2.4).
///
/// For each bar `t >= 1`, with `r_t = ln(close[t] / close[t-1])`, the accumulators are
/// updated as `S+ = max(0, S+ + r_t)` and `S- = min(0, S- + r_t)`. Bar `t` is an event when
/// `S- < -h` (checked first) or `S+ > h`, where `h` is the threshold for bar `t`; only the
/// accumulator that fired is reset to zero. The decision at bar `t` uses only data up to
/// `t`. Unlike AFML's snippet, which differences whatever series it is given, this always
/// takes log returns of prices.
///
/// Parameters
/// ----------
/// close : list[float]
///     Strictly positive close prices, oldest first.
/// threshold : float | Sequence[float]
///     CUSUM threshold `h` in log-return units: one number for every bar, or one value per
///     bar aligned with `close` (a list, NumPy array or pandas Series), typically a multiple
///     of a daily-volatility estimate. Element 0 of a per-bar sequence is never read (the
///     first return is at bar 1), so it may be NaN; every other value, and a scalar, must be
///     finite and non-negative.
///
/// Returns
/// -------
/// list[int]
///     0-based positions into `close` of the event bars, in increasing order; empty for
///     fewer than two prices.
///
/// Raises
/// ------
/// TypeError
///     If `threshold` is a string, or is neither a number nor a sequence of numbers.
/// ValueError
///     If a per-bar `threshold` does not have one value per price, or a threshold that is
///     read is negative or not finite.
#[pyfunction(name = "cusum_filter_indices")]
fn filters_cusum_filter_indices(
    close: Vec<f64>,
    threshold: &Bound<'_, PyAny>,
) -> PyResult<Vec<usize>> {
    let threshold = cusum_threshold(threshold, close.len())?;
    openquant::filters::cusum_filter_indices(&close, threshold).map_err(to_py_err)
}

/// Symmetric CUSUM filter returning event timestamps (AFML section 2.5.2.1, Snippet 2.4).
///
/// Runs `cusum_filter_indices` on `close` and maps each event position to its timestamp.
///
/// For each bar `t >= 1`, with `r_t = ln(close[t] / close[t-1])`, the accumulators are
/// updated as `S+ = max(0, S+ + r_t)` and `S- = min(0, S- + r_t)`. Bar `t` is an event when
/// `S- < -h` (checked first) or `S+ > h`, where `h` is the threshold for bar `t`; only the
/// accumulator that fired is reset to zero. The decision at bar `t` uses only data up to
/// `t`. Unlike AFML's snippet, which differences whatever series it is given, this always
/// takes log returns of prices.
///
/// Parameters
/// ----------
/// close : list[float]
///     Strictly positive close prices, oldest first.
/// timestamps : list[str]
///     One timestamp per price, as `"%Y-%m-%d %H:%M:%S"` (an optional fractional second is
///     accepted).
/// threshold : float | Sequence[float]
///     CUSUM threshold `h` in log-return units: one number for every bar, or one value per
///     bar aligned with `close` (a list, NumPy array or pandas Series), typically a multiple
///     of a daily-volatility estimate. Element 0 of a per-bar sequence is never read (the
///     first return is at bar 1), so it may be NaN; every other value, and a scalar, must be
///     finite and non-negative.
///
/// Returns
/// -------
/// list[str]
///     Timestamps of the event bars, in bar order, formatted as `"%Y-%m-%d %H:%M:%S"` (with
///     fractional seconds only when present).
///
/// Raises
/// ------
/// TypeError
///     If `threshold` is a string, or is neither a number nor a sequence of numbers.
/// ValueError
///     If a timestamp cannot be parsed, `close` and `timestamps` differ in length, a
///     per-bar `threshold` does not have one value per price, or a threshold that is read
///     is negative or not finite.
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

/// Rolling z-score filter returning event positions (ported from mlfinlab; not in AFML).
///
/// Bar `i` is an event when `close[i] >= mean + threshold * std`, where `mean` is the mean
/// of the last `mean_window` prices and `std` the sample standard deviation (ddof = 1, as
/// pandas) of the last `std_window` prices, both windows including bar `i`. Evaluation
/// starts at bar `max(mean_window, std_window) - 1`. The filter is one-sided (only upward
/// excursions fire), works on price levels and has no reset, so consecutive bars above the
/// band are consecutive events.
///
/// Parameters
/// ----------
/// close : list[float]
///     Close prices, oldest first.
/// mean_window : int
///     Number of prices in the rolling mean, including the current bar.
/// std_window : int
///     Number of prices in the rolling standard deviation, including the current bar.
/// threshold : float
///     Number of standard deviations above the rolling mean at which a bar fires.
///
/// Returns
/// -------
/// list[int]
///     0-based positions into `close` of the event bars, in increasing order; empty if
///     `close` is empty, both windows are zero, or `close` is shorter than the longer window.
#[pyfunction(name = "z_score_filter_indices")]
fn filters_z_score_filter_indices(
    close: Vec<f64>,
    mean_window: usize,
    std_window: usize,
    threshold: f64,
) -> Vec<usize> {
    openquant::filters::z_score_filter_indices(&close, mean_window, std_window, threshold)
}

/// Rolling z-score filter returning event timestamps (ported from mlfinlab; not in AFML).
///
/// Runs `z_score_filter_indices` on `close` and maps each event position to its timestamp.
///
/// Bar `i` is an event when `close[i] >= mean + threshold * std`, where `mean` is the mean
/// of the last `mean_window` prices and `std` the sample standard deviation (ddof = 1, as
/// pandas) of the last `std_window` prices, both windows including bar `i`. Evaluation
/// starts at bar `max(mean_window, std_window) - 1`. The filter is one-sided (only upward
/// excursions fire), works on price levels and has no reset, so consecutive bars above the
/// band are consecutive events.
///
/// Parameters
/// ----------
/// close : list[float]
///     Close prices, oldest first.
/// timestamps : list[str]
///     One timestamp per price, as `"%Y-%m-%d %H:%M:%S"` (an optional fractional second is
///     accepted).
/// mean_window : int
///     Number of prices in the rolling mean, including the current bar.
/// std_window : int
///     Number of prices in the rolling standard deviation, including the current bar.
/// threshold : float
///     Number of standard deviations above the rolling mean at which a bar fires.
///
/// Returns
/// -------
/// list[str]
///     Timestamps of the event bars, in bar order, formatted as `"%Y-%m-%d %H:%M:%S"` (with
///     fractional seconds only when present); empty in the cases where
///     `z_score_filter_indices` is.
///
/// Raises
/// ------
/// ValueError
///     If a timestamp cannot be parsed or `close` and `timestamps` differ in length.
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

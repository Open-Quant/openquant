use pyo3::prelude::*;

use crate::helpers::{
    format_naive_datetime, parse_naive_datetime, parse_naive_datetimes, to_py_err,
};

/// Sample weights by return attribution (AFML section 4.6, Snippet 4.10).
///
/// For each event `i`, `w_i = |sum_{t in [start_i, end_i]} r_t / c_t|`, where `r_t` is the
/// log return arriving at bar `t` (computed over the whole series, so the return arriving
/// at `start_i` is included) and `c_t` is the number of events whose span contains bar `t`.
/// The weights are then scaled to sum to the number of events (mean 1), unless they are
/// all zero. Spans are inclusive and matched to the close series by exact timestamp.
/// Concurrency is computed over the events passed in, so compute weights on the training
/// fold only. A single large move (a gap or bad print) can take most of the weight.
///
/// Parameters
/// ----------
/// events : list[tuple[str, str, float]]
///     `(start, end, label)` per event, timestamps as `"%Y-%m-%d %H:%M:%S"` (an optional
///     fractional second is accepted). The label is ignored.
/// close_timestamps : list[str]
///     Bar timestamps in ascending order, in the same format.
/// close_prices : list[float]
///     Strictly positive close prices (a non-positive price gives non-finite weights).
///
/// Returns
/// -------
/// list[tuple[str, float]]
///     One `(event_start, weight)` pair per event, in input order.
///
/// Raises
/// ------
/// ValueError
///     If a timestamp does not parse, the close timestamps and prices differ in length, or
///     an event's end precedes its start.
#[pyfunction(name = "get_weights_by_return")]
fn sw_get_weights_by_return(
    events: Vec<(String, String, f64)>,
    close_timestamps: Vec<String>,
    close_prices: Vec<f64>,
) -> PyResult<Vec<(String, f64)>> {
    let parsed_events: Vec<(chrono::NaiveDateTime, chrono::NaiveDateTime, f64)> = events
        .into_iter()
        .map(|(t_in, t_out, label)| {
            let t_in_dt = parse_naive_datetime(&t_in, "datetime")?;
            let t_out_dt = parse_naive_datetime(&t_out, "datetime")?;
            Ok((t_in_dt, t_out_dt, label))
        })
        .collect::<PyResult<Vec<_>>>()?;

    let close_ts = parse_naive_datetimes(close_timestamps)?;
    if close_ts.len() != close_prices.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "close timestamps/prices length mismatch",
        ));
    }
    let close: Vec<(chrono::NaiveDateTime, f64)> = close_ts.into_iter().zip(close_prices).collect();

    let result = openquant::sample_weights::get_weights_by_return(&parsed_events, &close)
        .map_err(to_py_err)?;
    Ok(result.into_iter().map(|(ts, v)| (format_naive_datetime(&ts), v)).collect())
}

/// Sample weights by time decay along cumulative uniqueness (AFML section 4.7, Snippet 4.11).
///
/// Each event's average uniqueness is the mean of `1 / c_t` over the close bars in its
/// inclusive span (`c_t` = number of events containing bar `t`); an event covering no bar
/// has uniqueness 0. With `x_i` the cumulative uniqueness in start order and `X` its total,
/// the weight is `max(0, a + b x_i)` with `b = (1 - decay) / X` for `decay >= 0`,
/// `b = 1 / ((decay + 1) X)` for `decay < 0`, and `a = 1 - b X`, so the newest event has
/// weight 1. `decay = 1` is no decay, `0 < decay < 1` decays linearly toward `decay`, `0`
/// toward 0, and `-1 < decay < 0` zeroes the oldest `-decay` fraction. The weights are not
/// rescaled and exclude return attribution (AFML suggests multiplying the two). Close
/// prices are not used, only their timestamps.
///
/// Parameters
/// ----------
/// events : list[tuple[str, str, float]]
///     `(start, end, label)` per event, timestamps as `"%Y-%m-%d %H:%M:%S"` (an optional
///     fractional second is accepted). The label is ignored.
/// close_timestamps : list[str]
///     Bar timestamps in ascending order, in the same format.
/// close_prices : list[float]
///     Close prices; only their count is checked.
/// decay : float
///     Decay parameter, meaningful in `(-1, 1]`; it is not validated.
///
/// Returns
/// -------
/// list[tuple[str, float]]
///     One `(event_start, weight)` pair per event, in input order.
///
/// Raises
/// ------
/// ValueError
///     If a timestamp does not parse, the close timestamps and prices differ in length, or
///     an event's end precedes its start.
#[pyfunction(name = "get_weights_by_time_decay")]
fn sw_get_weights_by_time_decay(
    events: Vec<(String, String, f64)>,
    close_timestamps: Vec<String>,
    close_prices: Vec<f64>,
    decay: f64,
) -> PyResult<Vec<(String, f64)>> {
    let parsed_events: Vec<(chrono::NaiveDateTime, chrono::NaiveDateTime, f64)> = events
        .into_iter()
        .map(|(t_in, t_out, label)| {
            let t_in_dt = parse_naive_datetime(&t_in, "datetime")?;
            let t_out_dt = parse_naive_datetime(&t_out, "datetime")?;
            Ok((t_in_dt, t_out_dt, label))
        })
        .collect::<PyResult<Vec<_>>>()?;

    let close_ts = parse_naive_datetimes(close_timestamps)?;
    if close_ts.len() != close_prices.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "close timestamps/prices length mismatch",
        ));
    }
    let close: Vec<(chrono::NaiveDateTime, f64)> = close_ts.into_iter().zip(close_prices).collect();

    let result =
        openquant::sample_weights::get_weights_by_time_decay(&parsed_events, &close, decay)
            .map_err(to_py_err)?;
    Ok(result.into_iter().map(|(ts, v)| (format_naive_datetime(&ts), v)).collect())
}

pub fn register(py: Python<'_>, parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(py, "sample_weights")?;
    m.add_function(wrap_pyfunction!(sw_get_weights_by_return, &m)?)?;
    m.add_function(wrap_pyfunction!(sw_get_weights_by_time_decay, &m)?)?;
    parent.add_submodule(&m)?;
    parent.add("sample_weights", m)?;
    Ok(())
}

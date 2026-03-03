use pyo3::prelude::*;

use crate::helpers::{parse_naive_datetimes, to_py_err};

#[pyfunction(name = "get_weights_by_return")]
fn sw_get_weights_by_return(
    events: Vec<(String, String, f64)>,
    close_timestamps: Vec<String>,
    close_prices: Vec<f64>,
) -> PyResult<Vec<(String, f64)>> {
    let parsed_events: Vec<(chrono::NaiveDateTime, chrono::NaiveDateTime, f64)> = events
        .into_iter()
        .map(|(t_in, t_out, label)| {
            let t_in_dt = chrono::NaiveDateTime::parse_from_str(&t_in, "%Y-%m-%d %H:%M:%S")
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("invalid datetime '{t_in}': {e}")))?;
            let t_out_dt = chrono::NaiveDateTime::parse_from_str(&t_out, "%Y-%m-%d %H:%M:%S")
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("invalid datetime '{t_out}': {e}")))?;
            Ok((t_in_dt, t_out_dt, label))
        })
        .collect::<PyResult<Vec<_>>>()?;

    let close_ts = parse_naive_datetimes(close_timestamps)?;
    if close_ts.len() != close_prices.len() {
        return Err(pyo3::exceptions::PyValueError::new_err("close timestamps/prices length mismatch"));
    }
    let close: Vec<(chrono::NaiveDateTime, f64)> = close_ts.into_iter().zip(close_prices).collect();

    let result = openquant::sample_weights::get_weights_by_return(&parsed_events, &close)
        .map_err(to_py_err)?;
    Ok(result
        .into_iter()
        .map(|(ts, v)| (ts.format("%Y-%m-%d %H:%M:%S").to_string(), v))
        .collect())
}

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
            let t_in_dt = chrono::NaiveDateTime::parse_from_str(&t_in, "%Y-%m-%d %H:%M:%S")
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("invalid datetime '{t_in}': {e}")))?;
            let t_out_dt = chrono::NaiveDateTime::parse_from_str(&t_out, "%Y-%m-%d %H:%M:%S")
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("invalid datetime '{t_out}': {e}")))?;
            Ok((t_in_dt, t_out_dt, label))
        })
        .collect::<PyResult<Vec<_>>>()?;

    let close_ts = parse_naive_datetimes(close_timestamps)?;
    if close_ts.len() != close_prices.len() {
        return Err(pyo3::exceptions::PyValueError::new_err("close timestamps/prices length mismatch"));
    }
    let close: Vec<(chrono::NaiveDateTime, f64)> = close_ts.into_iter().zip(close_prices).collect();

    let result = openquant::sample_weights::get_weights_by_time_decay(&parsed_events, &close, decay)
        .map_err(to_py_err)?;
    Ok(result
        .into_iter()
        .map(|(ts, v)| (ts.format("%Y-%m-%d %H:%M:%S").to_string(), v))
        .collect())
}

pub fn register(py: Python<'_>, parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(py, "sample_weights")?;
    m.add_function(wrap_pyfunction!(sw_get_weights_by_return, &m)?)?;
    m.add_function(wrap_pyfunction!(sw_get_weights_by_time_decay, &m)?)?;
    parent.add_submodule(&m)?;
    parent.add("sample_weights", m)?;
    Ok(())
}

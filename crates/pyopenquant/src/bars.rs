use openquant::data_structures::{imbalance_bars, run_bars, standard_bars, time_bars, ImbalanceBarType, StandardBarType};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::helpers::{bars_to_rows, build_trades};

#[pyfunction(name = "build_time_bars")]
fn bars_build_time_bars(
    timestamps: Vec<String>,
    prices: Vec<f64>,
    volumes: Vec<f64>,
    interval_seconds: i64,
) -> PyResult<Vec<(String, String, f64, f64, f64, f64, f64, f64, usize)>> {
    if interval_seconds <= 0 {
        return Err(PyValueError::new_err("interval_seconds must be > 0"));
    }
    let trades = build_trades(timestamps, prices, volumes)?;
    let bars = time_bars(&trades, chrono::Duration::seconds(interval_seconds));
    Ok(bars_to_rows(bars))
}

#[pyfunction(name = "build_tick_bars")]
fn bars_build_tick_bars(
    timestamps: Vec<String>,
    prices: Vec<f64>,
    volumes: Vec<f64>,
    ticks_per_bar: usize,
) -> PyResult<Vec<(String, String, f64, f64, f64, f64, f64, f64, usize)>> {
    if ticks_per_bar == 0 {
        return Err(PyValueError::new_err("ticks_per_bar must be > 0"));
    }
    let trades = build_trades(timestamps, prices, volumes)?;
    let bars = standard_bars(&trades, ticks_per_bar as f64, StandardBarType::Tick);
    Ok(bars_to_rows(bars))
}

#[pyfunction(name = "build_volume_bars")]
fn bars_build_volume_bars(
    timestamps: Vec<String>,
    prices: Vec<f64>,
    volumes: Vec<f64>,
    volume_per_bar: f64,
) -> PyResult<Vec<(String, String, f64, f64, f64, f64, f64, f64, usize)>> {
    if !volume_per_bar.is_finite() || volume_per_bar <= 0.0 {
        return Err(PyValueError::new_err("volume_per_bar must be > 0"));
    }
    let trades = build_trades(timestamps, prices, volumes)?;
    let bars = standard_bars(&trades, volume_per_bar, StandardBarType::Volume);
    Ok(bars_to_rows(bars))
}

#[pyfunction(name = "build_dollar_bars")]
fn bars_build_dollar_bars(
    timestamps: Vec<String>,
    prices: Vec<f64>,
    volumes: Vec<f64>,
    dollar_value_per_bar: f64,
) -> PyResult<Vec<(String, String, f64, f64, f64, f64, f64, f64, usize)>> {
    if !dollar_value_per_bar.is_finite() || dollar_value_per_bar <= 0.0 {
        return Err(PyValueError::new_err("dollar_value_per_bar must be > 0"));
    }
    let trades = build_trades(timestamps, prices, volumes)?;
    let bars = standard_bars(&trades, dollar_value_per_bar, StandardBarType::Dollar);
    Ok(bars_to_rows(bars))
}

#[pyfunction(name = "build_run_bars")]
fn bars_build_run_bars(
    timestamps: Vec<String>,
    prices: Vec<f64>,
    volumes: Vec<f64>,
    threshold: usize,
) -> PyResult<Vec<(String, String, f64, f64, f64, f64, f64, f64, usize)>> {
    if threshold == 0 {
        return Err(PyValueError::new_err("threshold must be > 0"));
    }
    let trades = build_trades(timestamps, prices, volumes)?;
    let bars = run_bars(&trades, threshold);
    Ok(bars_to_rows(bars))
}

#[pyfunction(name = "build_imbalance_bars")]
fn bars_build_imbalance_bars(
    timestamps: Vec<String>,
    prices: Vec<f64>,
    volumes: Vec<f64>,
    threshold: f64,
    bar_type: String,
) -> PyResult<Vec<(String, String, f64, f64, f64, f64, f64, f64, usize)>> {
    if !threshold.is_finite() || threshold <= 0.0 {
        return Err(PyValueError::new_err("threshold must be > 0"));
    }
    let bt = match bar_type.to_lowercase().as_str() {
        "tick" => ImbalanceBarType::Tick,
        "volume" => ImbalanceBarType::Volume,
        "dollar" => ImbalanceBarType::Dollar,
        _ => return Err(PyValueError::new_err("bar_type must be 'tick', 'volume', or 'dollar'")),
    };
    let trades = build_trades(timestamps, prices, volumes)?;
    let bars = imbalance_bars(&trades, threshold, bt);
    Ok(bars_to_rows(bars))
}

pub fn register(py: Python<'_>, parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(py, "bars")?;
    m.add_function(wrap_pyfunction!(bars_build_time_bars, &m)?)?;
    m.add_function(wrap_pyfunction!(bars_build_tick_bars, &m)?)?;
    m.add_function(wrap_pyfunction!(bars_build_volume_bars, &m)?)?;
    m.add_function(wrap_pyfunction!(bars_build_dollar_bars, &m)?)?;
    m.add_function(wrap_pyfunction!(bars_build_run_bars, &m)?)?;
    m.add_function(wrap_pyfunction!(bars_build_imbalance_bars, &m)?)?;
    parent.add_submodule(&m)?;
    parent.add("bars", m)?;
    Ok(())
}

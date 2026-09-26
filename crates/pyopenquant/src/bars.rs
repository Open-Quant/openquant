use openquant::data_structures::{
    imbalance_bars, run_bars, standard_bars, time_bars, ImbalanceBarType, StandardBarType,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::helpers::{bars_to_rows, build_trades, to_py_err, BarRow};

/// Build time bars from a trade stream (AFML section 2.3.1.1).
///
/// A bar closes on the first trade at least `interval_seconds` after the bar's first trade;
/// that trade belongs to the closing bar. Bars are anchored to each bar's first trade, not
/// to the wall clock. A trailing partial bar is emitted as a final, shorter bar. Trades must
/// be in increasing time order; the input is not sorted or validated.
///
/// Parameters
/// ----------
/// timestamps : list[str]
///     Trade timestamps as `"%Y-%m-%d %H:%M:%S"` (an optional fractional second is
///     accepted) or `"%Y-%m-%d"` (read as midnight).
/// prices : list[float]
///     Trade prices.
/// volumes : list[float]
///     Trade sizes.
/// interval_seconds : int
///     Bar length in seconds; must be positive.
///
/// Returns
/// -------
/// list[tuple[str, str, float, float, float, float, float, float, int]]
///     One `(start_timestamp, timestamp, open, high, low, close, volume, dollar_value,
///     tick_count)` row per bar, where `start_timestamp` and `timestamp` are the times of
///     the bar's first and last trade and `dollar_value` is the sum of `price * volume`.
///
/// Raises
/// ------
/// ValueError
///     If `interval_seconds <= 0`, the three lists differ in length, or a timestamp does
///     not parse.
#[pyfunction(name = "build_time_bars")]
fn bars_build_time_bars(
    timestamps: Vec<String>,
    prices: Vec<f64>,
    volumes: Vec<f64>,
    interval_seconds: i64,
) -> PyResult<Vec<BarRow>> {
    if interval_seconds <= 0 {
        return Err(PyValueError::new_err("interval_seconds must be > 0"));
    }
    let trades = build_trades(timestamps, prices, volumes)?;
    let bars =
        time_bars(&trades, chrono::Duration::seconds(interval_seconds)).map_err(to_py_err)?;
    Ok(bars_to_rows(bars))
}

/// Build tick bars that close after a fixed number of trades (AFML section 2.3.1.2).
///
/// A bar closes once `ticks_per_bar` trades have accumulated. A trailing partial bar is
/// dropped. Trades must be in increasing time order; the input is not sorted or validated.
///
/// Parameters
/// ----------
/// timestamps : list[str]
///     Trade timestamps as `"%Y-%m-%d %H:%M:%S"` (an optional fractional second is
///     accepted) or `"%Y-%m-%d"` (read as midnight).
/// prices : list[float]
///     Trade prices.
/// volumes : list[float]
///     Trade sizes.
/// ticks_per_bar : int
///     Number of trades per bar; must be positive.
///
/// Returns
/// -------
/// list[tuple[str, str, float, float, float, float, float, float, int]]
///     One `(start_timestamp, timestamp, open, high, low, close, volume, dollar_value,
///     tick_count)` row per bar, where `start_timestamp` and `timestamp` are the times of
///     the bar's first and last trade and `dollar_value` is the sum of `price * volume`.
///
/// Raises
/// ------
/// ValueError
///     If `ticks_per_bar` is 0, the three lists differ in length, or a timestamp does not
///     parse.
#[pyfunction(name = "build_tick_bars")]
fn bars_build_tick_bars(
    timestamps: Vec<String>,
    prices: Vec<f64>,
    volumes: Vec<f64>,
    ticks_per_bar: usize,
) -> PyResult<Vec<BarRow>> {
    if ticks_per_bar == 0 {
        return Err(PyValueError::new_err("ticks_per_bar must be > 0"));
    }
    let trades = build_trades(timestamps, prices, volumes)?;
    let bars =
        standard_bars(&trades, ticks_per_bar as f64, StandardBarType::Tick).map_err(to_py_err)?;
    Ok(bars_to_rows(bars))
}

/// Build volume bars that close when cumulative volume reaches a threshold.
///
/// AFML section 2.3.1.3. The trade that crosses the threshold belongs to the bar it
/// closes, so bars overshoot. A trailing partial bar is dropped. Trades must be in
/// increasing time order; the input is not sorted or validated.
///
/// Parameters
/// ----------
/// timestamps : list[str]
///     Trade timestamps as `"%Y-%m-%d %H:%M:%S"` (an optional fractional second is
///     accepted) or `"%Y-%m-%d"` (read as midnight).
/// prices : list[float]
///     Trade prices.
/// volumes : list[float]
///     Trade sizes.
/// volume_per_bar : float
///     Cumulative volume that closes a bar; must be positive and finite.
///
/// Returns
/// -------
/// list[tuple[str, str, float, float, float, float, float, float, int]]
///     One `(start_timestamp, timestamp, open, high, low, close, volume, dollar_value,
///     tick_count)` row per bar, where `start_timestamp` and `timestamp` are the times of
///     the bar's first and last trade and `dollar_value` is the sum of `price * volume`.
///
/// Raises
/// ------
/// ValueError
///     If `volume_per_bar` is not a positive finite number, the three lists differ in
///     length, or a timestamp does not parse.
#[pyfunction(name = "build_volume_bars")]
fn bars_build_volume_bars(
    timestamps: Vec<String>,
    prices: Vec<f64>,
    volumes: Vec<f64>,
    volume_per_bar: f64,
) -> PyResult<Vec<BarRow>> {
    if !volume_per_bar.is_finite() || volume_per_bar <= 0.0 {
        return Err(PyValueError::new_err("volume_per_bar must be > 0"));
    }
    let trades = build_trades(timestamps, prices, volumes)?;
    let bars =
        standard_bars(&trades, volume_per_bar, StandardBarType::Volume).map_err(to_py_err)?;
    Ok(bars_to_rows(bars))
}

/// Build dollar bars that close when cumulative `price * volume` reaches a threshold.
///
/// AFML section 2.3.1.4. The trade that crosses the threshold belongs to the bar it
/// closes, so bars overshoot. A trailing partial bar is dropped. Trades must be in
/// increasing time order; the input is not sorted or validated.
///
/// Parameters
/// ----------
/// timestamps : list[str]
///     Trade timestamps as `"%Y-%m-%d %H:%M:%S"` (an optional fractional second is
///     accepted) or `"%Y-%m-%d"` (read as midnight).
/// prices : list[float]
///     Trade prices.
/// volumes : list[float]
///     Trade sizes.
/// dollar_value_per_bar : float
///     Cumulative traded value that closes a bar; must be positive and finite.
///
/// Returns
/// -------
/// list[tuple[str, str, float, float, float, float, float, float, int]]
///     One `(start_timestamp, timestamp, open, high, low, close, volume, dollar_value,
///     tick_count)` row per bar, where `start_timestamp` and `timestamp` are the times of
///     the bar's first and last trade and `dollar_value` is the sum of `price * volume`.
///
/// Raises
/// ------
/// ValueError
///     If `dollar_value_per_bar` is not a positive finite number, the three lists differ in
///     length, or a timestamp does not parse.
#[pyfunction(name = "build_dollar_bars")]
fn bars_build_dollar_bars(
    timestamps: Vec<String>,
    prices: Vec<f64>,
    volumes: Vec<f64>,
    dollar_value_per_bar: f64,
) -> PyResult<Vec<BarRow>> {
    if !dollar_value_per_bar.is_finite() || dollar_value_per_bar <= 0.0 {
        return Err(PyValueError::new_err("dollar_value_per_bar must be > 0"));
    }
    let trades = build_trades(timestamps, prices, volumes)?;
    let bars =
        standard_bars(&trades, dollar_value_per_bar, StandardBarType::Dollar).map_err(to_py_err)?;
    Ok(bars_to_rows(bars))
}

/// Build run bars that close after a run of same-direction ticks.
///
/// A fixed-threshold simplification of AFML section 2.3.2.3 (AFML uses an adaptive
/// expected threshold). Trade direction follows the tick rule: up-tick +1, down-tick -1,
/// an unchanged price keeps the previous sign and so extends the current run. A bar
/// closes when `threshold` consecutive moves occur in the same direction. A trailing
/// partial bar is dropped, and fewer than two trades yield no bars. Trades must be in
/// increasing time order; the input is not sorted or validated.
///
/// Parameters
/// ----------
/// timestamps : list[str]
///     Trade timestamps as `"%Y-%m-%d %H:%M:%S"` (an optional fractional second is
///     accepted) or `"%Y-%m-%d"` (read as midnight).
/// prices : list[float]
///     Trade prices.
/// volumes : list[float]
///     Trade sizes.
/// threshold : int
///     Run length that closes a bar; must be positive.
///
/// Returns
/// -------
/// list[tuple[str, str, float, float, float, float, float, float, int]]
///     One `(start_timestamp, timestamp, open, high, low, close, volume, dollar_value,
///     tick_count)` row per bar, where `start_timestamp` and `timestamp` are the times of
///     the bar's first and last trade and `dollar_value` is the sum of `price * volume`.
///
/// Raises
/// ------
/// ValueError
///     If `threshold` is 0, the three lists differ in length, or a timestamp does not
///     parse.
#[pyfunction(name = "build_run_bars")]
fn bars_build_run_bars(
    timestamps: Vec<String>,
    prices: Vec<f64>,
    volumes: Vec<f64>,
    threshold: usize,
) -> PyResult<Vec<BarRow>> {
    if threshold == 0 {
        return Err(PyValueError::new_err("threshold must be > 0"));
    }
    let trades = build_trades(timestamps, prices, volumes)?;
    let bars = run_bars(&trades, threshold).map_err(to_py_err)?;
    Ok(bars_to_rows(bars))
}

/// Build imbalance bars that close when the absolute signed imbalance reaches a threshold.
///
/// A fixed-threshold simplification of AFML sections 2.3.2.1-2 (AFML's threshold is an
/// exponentially weighted expectation). Each trade after the first contributes its
/// tick-rule sign (up-tick +1, down-tick -1, unchanged price keeps the previous sign)
/// times a weight: 1, its volume, or its `price * volume`, per `bar_type`. A trailing
/// partial bar is dropped, and fewer than two trades yield no bars. Trades must be in
/// increasing time order; the input is not sorted or validated.
///
/// Parameters
/// ----------
/// timestamps : list[str]
///     Trade timestamps as `"%Y-%m-%d %H:%M:%S"` (an optional fractional second is
///     accepted) or `"%Y-%m-%d"` (read as midnight).
/// prices : list[float]
///     Trade prices.
/// volumes : list[float]
///     Trade sizes.
/// threshold : float
///     Absolute imbalance that closes a bar; must be positive and finite.
/// bar_type : str
///     Imbalance weight: `"tick"`, `"volume"` or `"dollar"` (case-insensitive).
///
/// Returns
/// -------
/// list[tuple[str, str, float, float, float, float, float, float, int]]
///     One `(start_timestamp, timestamp, open, high, low, close, volume, dollar_value,
///     tick_count)` row per bar, where `start_timestamp` and `timestamp` are the times of
///     the bar's first and last trade and `dollar_value` is the sum of `price * volume`.
///
/// Raises
/// ------
/// ValueError
///     If `threshold` is not a positive finite number, `bar_type` is not one of the
///     accepted values, the three lists differ in length, or a timestamp does not parse.
#[pyfunction(name = "build_imbalance_bars")]
fn bars_build_imbalance_bars(
    timestamps: Vec<String>,
    prices: Vec<f64>,
    volumes: Vec<f64>,
    threshold: f64,
    bar_type: String,
) -> PyResult<Vec<BarRow>> {
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
    let bars = imbalance_bars(&trades, threshold, bt).map_err(to_py_err)?;
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

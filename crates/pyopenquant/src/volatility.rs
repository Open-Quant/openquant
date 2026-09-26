use pyo3::prelude::*;

use crate::helpers::{format_naive_datetime, pair_timestamps_values, to_py_err};

/// Daily volatility: exponentially weighted standard deviation of daily returns.
///
/// AFML Snippet 3.1, matching `mlfinlab.util.volatility.get_daily_vol`. For each bar, the
/// return is the simple return `p_t / p_j - 1` against the last bar `j` strictly more than one
/// day earlier; the volatility is pandas' `ewm(span=lookback).std()` of those returns
/// (`adjust=True`, unbiased). Bars with no bar more than a day before them are omitted, and
/// the first returned value is NaN (one return has no sample variance). Bars must be in
/// increasing time. A typical use is as the target of the labeling functions, which drop
/// NaN targets.
///
/// Parameters
/// ----------
/// close_timestamps : list[str]
///     Bar timestamps as `"%Y-%m-%d %H:%M:%S"` (an optional fractional second is accepted),
///     in increasing order.
/// close_prices : list[float]
///     Close price of each bar.
/// lookback : int
///     EWM span, in number of returns.
///
/// Returns
/// -------
/// list[tuple[str, float]]
///     `(timestamp, volatility)` rows; empty when there are fewer than two bars or
///     `lookback` is 0.
///
/// Raises
/// ------
/// ValueError
///     If the timestamps and prices differ in length, or a timestamp does not parse.
#[pyfunction(name = "get_daily_vol")]
fn volatility_get_daily_vol(
    close_timestamps: Vec<String>,
    close_prices: Vec<f64>,
    lookback: usize,
) -> PyResult<Vec<(String, f64)>> {
    let close =
        pair_timestamps_values(close_timestamps, close_prices, "close_timestamps", "close_prices")?;
    let result = openquant::util::volatility::get_daily_vol(&close, lookback);
    Ok(result.into_iter().map(|(ts, v)| (format_naive_datetime(&ts), v)).collect())
}

/// Parkinson high-low volatility over a rolling window.
///
/// Each output is `sqrt(mean(ln(high / low)^2) / (4 ln 2))` over the last `window` bars, per
/// bar (not annualised), matching mlfinlab's `get_parksinson_vol`. The first `window - 1`
/// outputs are NaN, as is any output whose window contains a NaN; `window=0` gives all NaN.
///
/// Parameters
/// ----------
/// high : list[float]
///     High price of each bar, oldest first.
/// low : list[float]
///     Low price of each bar.
/// window : int
///     Number of bars in the rolling window.
///
/// Returns
/// -------
/// list[float]
///     Volatility per bar, same length as `high`.
///
/// Raises
/// ------
/// ValueError
///     If `low` differs in length from `high`.
#[pyfunction(name = "get_parkinson_vol")]
fn volatility_get_parkinson_vol(
    high: Vec<f64>,
    low: Vec<f64>,
    window: usize,
) -> PyResult<Vec<f64>> {
    openquant::util::volatility::get_parkinson_vol(&high, &low, window).map_err(to_py_err)
}

/// Garman-Klass volatility over a rolling window.
///
/// Each output is the square root of the mean over the last `window` bars of
/// `0.5 ln(high / low)^2 - (2 ln 2 - 1) ln(close / open)^2`, per bar (not annualised),
/// matching `mlfinlab.util.volatility.get_garman_class_vol`. The first `window - 1` outputs
/// are NaN, as is any output whose window contains a NaN; `window=0` gives all NaN.
///
/// Parameters
/// ----------
/// open : list[float]
///     Open price of each bar, oldest first.
/// high : list[float]
///     High price of each bar.
/// low : list[float]
///     Low price of each bar.
/// close : list[float]
///     Close price of each bar.
/// window : int
///     Number of bars in the rolling window.
///
/// Returns
/// -------
/// list[float]
///     Volatility per bar, same length as `open`.
///
/// Raises
/// ------
/// ValueError
///     If `high`, `low` or `close` differs in length from `open`.
#[pyfunction(name = "get_garman_class_vol")]
fn volatility_get_garman_class_vol(
    open: Vec<f64>,
    high: Vec<f64>,
    low: Vec<f64>,
    close: Vec<f64>,
    window: usize,
) -> PyResult<Vec<f64>> {
    openquant::util::volatility::get_garman_class_vol(&open, &high, &low, &close, window)
        .map_err(to_py_err)
}

/// Yang-Zhang (2000) volatility over a rolling window.
///
/// For the `n = window` bars ending at each bar, returns
/// `sqrt(var_o + k var_c + (1 - k) var_rs)` with `k = 0.34 / (1.34 + (n + 1) / (n - 1))`,
/// where `var_o` is the sample variance of the overnight returns `ln(open_i / close_{i-1})`,
/// `var_c` that of the open-to-close returns `ln(close_i / open_i)`, and `var_rs` the mean
/// Rogers-Satchell term. The result is per bar, not annualised. The overnight return needs the
/// previous close, so the first `window` outputs are NaN; a window containing a NaN gives NaN
/// and `window < 2` gives all NaN. This differs from mlfinlab's `get_yang_zhang_vol`, which
/// uses `ln(close_i / open_{i-1})` and does not demean.
///
/// Parameters
/// ----------
/// open : list[float]
///     Open price of each bar, oldest first.
/// high : list[float]
///     High price of each bar.
/// low : list[float]
///     Low price of each bar.
/// close : list[float]
///     Close price of each bar.
/// window : int
///     Number of bars in the rolling window.
///
/// Returns
/// -------
/// list[float]
///     Volatility per bar, same length as `open`.
///
/// Raises
/// ------
/// ValueError
///     If `high`, `low` or `close` differs in length from `open`.
#[pyfunction(name = "get_yang_zhang_vol")]
fn volatility_get_yang_zhang_vol(
    open: Vec<f64>,
    high: Vec<f64>,
    low: Vec<f64>,
    close: Vec<f64>,
    window: usize,
) -> PyResult<Vec<f64>> {
    openquant::util::volatility::get_yang_zhang_vol(&open, &high, &low, &close, window)
        .map_err(to_py_err)
}

pub fn register(py: Python<'_>, parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(py, "volatility")?;
    m.add_function(wrap_pyfunction!(volatility_get_daily_vol, &m)?)?;
    m.add_function(wrap_pyfunction!(volatility_get_parkinson_vol, &m)?)?;
    m.add_function(wrap_pyfunction!(volatility_get_garman_class_vol, &m)?)?;
    m.add_function(wrap_pyfunction!(volatility_get_yang_zhang_vol, &m)?)?;
    parent.add_submodule(&m)?;
    parent.add("volatility", m)?;
    Ok(())
}

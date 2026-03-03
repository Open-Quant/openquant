use pyo3::prelude::*;

use crate::helpers::pair_timestamps_values;

#[pyfunction(name = "get_daily_vol")]
fn volatility_get_daily_vol(
    close_timestamps: Vec<String>,
    close_prices: Vec<f64>,
    lookback: usize,
) -> PyResult<Vec<(String, f64)>> {
    let close = pair_timestamps_values(close_timestamps, close_prices, "close_timestamps", "close_prices")?;
    let result = openquant::util::volatility::get_daily_vol(&close, lookback);
    Ok(result
        .into_iter()
        .map(|(ts, v)| (ts.format("%Y-%m-%d %H:%M:%S").to_string(), v))
        .collect())
}

#[pyfunction(name = "get_parksinson_vol")]
fn volatility_get_parksinson_vol(high: Vec<f64>, low: Vec<f64>, window: usize) -> Vec<f64> {
    openquant::util::volatility::get_parksinson_vol(&high, &low, window)
}

#[pyfunction(name = "get_garman_class_vol")]
fn volatility_get_garman_class_vol(
    open: Vec<f64>,
    high: Vec<f64>,
    low: Vec<f64>,
    close: Vec<f64>,
    window: usize,
) -> Vec<f64> {
    openquant::util::volatility::get_garman_class_vol(&open, &high, &low, &close, window)
}

#[pyfunction(name = "get_yang_zhang_vol")]
fn volatility_get_yang_zhang_vol(
    open: Vec<f64>,
    high: Vec<f64>,
    low: Vec<f64>,
    close: Vec<f64>,
    window: usize,
) -> Vec<f64> {
    openquant::util::volatility::get_yang_zhang_vol(&open, &high, &low, &close, window)
}

pub fn register(py: Python<'_>, parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(py, "volatility")?;
    m.add_function(wrap_pyfunction!(volatility_get_daily_vol, &m)?)?;
    m.add_function(wrap_pyfunction!(volatility_get_parksinson_vol, &m)?)?;
    m.add_function(wrap_pyfunction!(volatility_get_garman_class_vol, &m)?)?;
    m.add_function(wrap_pyfunction!(volatility_get_yang_zhang_vol, &m)?)?;
    parent.add_submodule(&m)?;
    parent.add("volatility", m)?;
    Ok(())
}

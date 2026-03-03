use pyo3::prelude::*;

use crate::helpers::to_py_err;

// --- Bar-based features ---

#[pyfunction(name = "get_roll_measure")]
fn ms_get_roll_measure(close: Vec<f64>, window: usize) -> Vec<f64> {
    openquant::microstructural_features::get_roll_measure(&close, window)
}

#[pyfunction(name = "get_roll_impact")]
fn ms_get_roll_impact(close: Vec<f64>, dollar_volume: Vec<f64>, window: usize) -> Vec<f64> {
    openquant::microstructural_features::get_roll_impact(&close, &dollar_volume, window)
}

#[pyfunction(name = "get_corwin_schultz_estimator")]
fn ms_get_corwin_schultz_estimator(high: Vec<f64>, low: Vec<f64>, window: usize) -> Vec<f64> {
    openquant::microstructural_features::get_corwin_schultz_estimator(&high, &low, window)
}

#[pyfunction(name = "get_bekker_parkinson_vol")]
fn ms_get_bekker_parkinson_vol(high: Vec<f64>, low: Vec<f64>, window: usize) -> Vec<f64> {
    openquant::microstructural_features::get_bekker_parkinson_vol(&high, &low, window)
}

#[pyfunction(name = "get_bar_based_kyle_lambda")]
fn ms_get_bar_based_kyle_lambda(close: Vec<f64>, volume: Vec<f64>, window: usize) -> Vec<f64> {
    openquant::microstructural_features::get_bar_based_kyle_lambda(&close, &volume, window)
}

#[pyfunction(name = "get_bar_based_amihud_lambda")]
fn ms_get_bar_based_amihud_lambda(close: Vec<f64>, dollar_volume: Vec<f64>, window: usize) -> Vec<f64> {
    openquant::microstructural_features::get_bar_based_amihud_lambda(&close, &dollar_volume, window)
}

#[pyfunction(name = "get_bar_based_hasbrouck_lambda")]
fn ms_get_bar_based_hasbrouck_lambda(close: Vec<f64>, dollar_volume: Vec<f64>, window: usize) -> Vec<f64> {
    openquant::microstructural_features::get_bar_based_hasbrouck_lambda(&close, &dollar_volume, window)
}

// --- Trade-based features ---

#[pyfunction(name = "get_trades_based_kyle_lambda")]
fn ms_get_trades_based_kyle_lambda(price_diff: Vec<f64>, volume: Vec<f64>, aggressor_flags: Vec<f64>) -> f64 {
    openquant::microstructural_features::get_trades_based_kyle_lambda(&price_diff, &volume, &aggressor_flags)
}

#[pyfunction(name = "get_trades_based_amihud_lambda")]
fn ms_get_trades_based_amihud_lambda(log_ret: Vec<f64>, dollar_volume: Vec<f64>) -> f64 {
    openquant::microstructural_features::get_trades_based_amihud_lambda(&log_ret, &dollar_volume)
}

#[pyfunction(name = "get_trades_based_hasbrouck_lambda")]
fn ms_get_trades_based_hasbrouck_lambda(log_ret: Vec<f64>, dollar_volume: Vec<f64>, aggressor_flags: Vec<f64>) -> f64 {
    openquant::microstructural_features::get_trades_based_hasbrouck_lambda(&log_ret, &dollar_volume, &aggressor_flags)
}

// --- VPIN ---

#[pyfunction(name = "vwap")]
fn ms_vwap(dollar_volume: Vec<f64>, volume: Vec<f64>) -> f64 {
    openquant::microstructural_features::vwap(&dollar_volume, &volume)
}

#[pyfunction(name = "get_avg_tick_size")]
fn ms_get_avg_tick_size(tick_sizes: Vec<f64>) -> f64 {
    openquant::microstructural_features::get_avg_tick_size(&tick_sizes)
}

#[pyfunction(name = "get_vpin")]
fn ms_get_vpin(volume: Vec<f64>, buy_volume: Vec<f64>, window: usize) -> Vec<f64> {
    openquant::microstructural_features::get_vpin(&volume, &buy_volume, window)
}

#[pyfunction(name = "get_bvc_buy_volume")]
fn ms_get_bvc_buy_volume(close: Vec<f64>, volume: Vec<f64>, window: usize) -> Vec<f64> {
    openquant::microstructural_features::get_bvc_buy_volume(&close, &volume, window)
}

// --- Encoding ---

#[pyfunction(name = "encode_tick_rule_array")]
fn ms_encode_tick_rule_array(arr: Vec<i32>) -> PyResult<String> {
    openquant::microstructural_features::encode_tick_rule_array(&arr).map_err(to_py_err)
}

#[pyfunction(name = "quantile_mapping")]
fn ms_quantile_mapping(array: Vec<f64>, num_letters: usize) -> PyResult<Vec<(f64, char)>> {
    openquant::microstructural_features::quantile_mapping(&array, num_letters).map_err(to_py_err)
}

#[pyfunction(name = "sigma_mapping")]
fn ms_sigma_mapping(array: Vec<f64>, step: f64) -> PyResult<Vec<(f64, char)>> {
    openquant::microstructural_features::sigma_mapping(&array, step).map_err(to_py_err)
}

#[pyfunction(name = "encode_array")]
fn ms_encode_array(array: Vec<f64>, encoding: Vec<(f64, char)>) -> String {
    openquant::microstructural_features::encode_array(&array, &encoding)
}

// --- Entropy ---

#[pyfunction(name = "get_shannon_entropy")]
fn ms_get_shannon_entropy(message: String) -> f64 {
    openquant::microstructural_features::get_shannon_entropy(&message)
}

#[pyfunction(name = "get_lempel_ziv_entropy")]
fn ms_get_lempel_ziv_entropy(message: String) -> f64 {
    openquant::microstructural_features::get_lempel_ziv_entropy(&message)
}

#[pyfunction(name = "get_plug_in_entropy")]
fn ms_get_plug_in_entropy(message: String, word_length: usize) -> f64 {
    openquant::microstructural_features::get_plug_in_entropy(&message, word_length)
}

#[pyfunction(name = "get_konto_entropy")]
fn ms_get_konto_entropy(message: String, window: usize) -> f64 {
    openquant::microstructural_features::get_konto_entropy(&message, window)
}

pub fn register(py: Python<'_>, parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(py, "microstructural")?;
    // Bar-based
    m.add_function(wrap_pyfunction!(ms_get_roll_measure, &m)?)?;
    m.add_function(wrap_pyfunction!(ms_get_roll_impact, &m)?)?;
    m.add_function(wrap_pyfunction!(ms_get_corwin_schultz_estimator, &m)?)?;
    m.add_function(wrap_pyfunction!(ms_get_bekker_parkinson_vol, &m)?)?;
    m.add_function(wrap_pyfunction!(ms_get_bar_based_kyle_lambda, &m)?)?;
    m.add_function(wrap_pyfunction!(ms_get_bar_based_amihud_lambda, &m)?)?;
    m.add_function(wrap_pyfunction!(ms_get_bar_based_hasbrouck_lambda, &m)?)?;
    // Trade-based
    m.add_function(wrap_pyfunction!(ms_get_trades_based_kyle_lambda, &m)?)?;
    m.add_function(wrap_pyfunction!(ms_get_trades_based_amihud_lambda, &m)?)?;
    m.add_function(wrap_pyfunction!(ms_get_trades_based_hasbrouck_lambda, &m)?)?;
    // VPIN
    m.add_function(wrap_pyfunction!(ms_vwap, &m)?)?;
    m.add_function(wrap_pyfunction!(ms_get_avg_tick_size, &m)?)?;
    m.add_function(wrap_pyfunction!(ms_get_vpin, &m)?)?;
    m.add_function(wrap_pyfunction!(ms_get_bvc_buy_volume, &m)?)?;
    // Encoding
    m.add_function(wrap_pyfunction!(ms_encode_tick_rule_array, &m)?)?;
    m.add_function(wrap_pyfunction!(ms_quantile_mapping, &m)?)?;
    m.add_function(wrap_pyfunction!(ms_sigma_mapping, &m)?)?;
    m.add_function(wrap_pyfunction!(ms_encode_array, &m)?)?;
    // Entropy
    m.add_function(wrap_pyfunction!(ms_get_shannon_entropy, &m)?)?;
    m.add_function(wrap_pyfunction!(ms_get_lempel_ziv_entropy, &m)?)?;
    m.add_function(wrap_pyfunction!(ms_get_plug_in_entropy, &m)?)?;
    m.add_function(wrap_pyfunction!(ms_get_konto_entropy, &m)?)?;
    parent.add_submodule(&m)?;
    parent.add("microstructural", m)?;
    Ok(())
}

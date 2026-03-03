use pyo3::prelude::*;

use crate::helpers::{pair_timestamps_values, parse_naive_datetimes, to_py_err};

#[pyfunction(name = "get_signal")]
#[pyo3(signature = (prob, num_classes, pred=None))]
fn bet_sizing_get_signal(prob: Vec<f64>, num_classes: usize, pred: Option<Vec<f64>>) -> Vec<f64> {
    openquant::bet_sizing::get_signal(&prob, num_classes, pred.as_deref())
}

#[pyfunction(name = "discrete_signal")]
fn bet_sizing_discrete_signal(signal0: Vec<f64>, step_size: f64) -> Vec<f64> {
    openquant::bet_sizing::discrete_signal(&signal0, step_size)
}

#[pyfunction(name = "bet_size")]
fn bet_sizing_bet_size(w_param: f64, price_div: f64, func: String) -> PyResult<f64> {
    openquant::bet_sizing::bet_size_checked(w_param, price_div, &func).map_err(to_py_err)
}

#[pyfunction(name = "bet_size_sigmoid")]
fn bet_sizing_bet_size_sigmoid(w_param: f64, price_div: f64) -> f64 {
    openquant::bet_sizing::bet_size_sigmoid(w_param, price_div)
}

#[pyfunction(name = "bet_size_power")]
fn bet_sizing_bet_size_power(w_param: f64, price_div: f64) -> PyResult<f64> {
    openquant::bet_sizing::bet_size_power_checked(w_param, price_div).map_err(to_py_err)
}

#[pyfunction(name = "inv_price")]
fn bet_sizing_inv_price(forecast_price: f64, w_param: f64, m_bet_size: f64, func: String) -> PyResult<f64> {
    openquant::bet_sizing::inv_price_checked(forecast_price, w_param, m_bet_size, &func).map_err(to_py_err)
}

#[pyfunction(name = "inv_price_sigmoid")]
fn bet_sizing_inv_price_sigmoid(forecast_price: f64, w_param: f64, m_bet_size: f64) -> f64 {
    openquant::bet_sizing::inv_price_sigmoid(forecast_price, w_param, m_bet_size)
}

#[pyfunction(name = "inv_price_power")]
fn bet_sizing_inv_price_power(forecast_price: f64, w_param: f64, m_bet_size: f64) -> f64 {
    openquant::bet_sizing::inv_price_power(forecast_price, w_param, m_bet_size)
}

#[pyfunction(name = "get_w")]
fn bet_sizing_get_w(price_div: f64, m_bet_size: f64, func: String) -> PyResult<f64> {
    openquant::bet_sizing::get_w_checked(price_div, m_bet_size, &func).map_err(to_py_err)
}

#[pyfunction(name = "get_w_sigmoid")]
fn bet_sizing_get_w_sigmoid(price_div: f64, m_bet_size: f64) -> f64 {
    openquant::bet_sizing::get_w_sigmoid(price_div, m_bet_size)
}

#[pyfunction(name = "get_w_power")]
fn bet_sizing_get_w_power(price_div: f64, m_bet_size: f64) -> PyResult<f64> {
    openquant::bet_sizing::get_w_power_checked(price_div, m_bet_size).map_err(to_py_err)
}

#[pyfunction(name = "get_target_pos")]
fn bet_sizing_get_target_pos(w: f64, f: f64, m_p: f64, max_pos: f64, func: String) -> PyResult<f64> {
    openquant::bet_sizing::get_target_pos_checked(w, f, m_p, max_pos, &func).map_err(to_py_err)
}

#[pyfunction(name = "get_target_pos_sigmoid")]
fn bet_sizing_get_target_pos_sigmoid(w_param: f64, forecast_price: f64, market_price: f64, max_pos: f64) -> f64 {
    openquant::bet_sizing::get_target_pos_sigmoid(w_param, forecast_price, market_price, max_pos)
}

#[pyfunction(name = "get_target_pos_power")]
fn bet_sizing_get_target_pos_power(w_param: f64, forecast_price: f64, market_price: f64, max_pos: f64) -> f64 {
    openquant::bet_sizing::get_target_pos_power(w_param, forecast_price, market_price, max_pos)
}

#[pyfunction(name = "limit_price")]
fn bet_sizing_limit_price(t_pos: f64, pos: f64, f: f64, w: f64, max_pos: f64, func: String) -> PyResult<f64> {
    openquant::bet_sizing::limit_price_checked(t_pos, pos, f, w, max_pos, &func).map_err(to_py_err)
}

#[pyfunction(name = "limit_price_sigmoid")]
fn bet_sizing_limit_price_sigmoid(t_pos: f64, pos: f64, f: f64, w: f64, max_pos: f64) -> f64 {
    openquant::bet_sizing::limit_price_sigmoid(t_pos, pos, f, w, max_pos)
}

#[pyfunction(name = "limit_price_power")]
fn bet_sizing_limit_price_power(t_pos: f64, pos: f64, f: f64, w: f64, max_pos: f64) -> f64 {
    openquant::bet_sizing::limit_price_power(t_pos, pos, f, w, max_pos)
}

#[pyfunction(name = "avg_active_signals")]
fn bet_sizing_avg_active_signals(
    signal_timestamps: Vec<String>,
    signal_values: Vec<f64>,
    t1_timestamps: Vec<String>,
) -> PyResult<Vec<(String, f64)>> {
    let signal = pair_timestamps_values(signal_timestamps, signal_values, "signal_timestamps", "signal_values")?;
    let t1 = parse_naive_datetimes(t1_timestamps)?;
    let result = openquant::bet_sizing::avg_active_signals(&signal, &t1);
    Ok(result
        .into_iter()
        .map(|(ts, v)| (ts.format("%Y-%m-%d %H:%M:%S").to_string(), v))
        .collect())
}

#[pyfunction(name = "bet_size_dynamic")]
fn bet_sizing_bet_size_dynamic(
    pos: Vec<f64>,
    max_pos: Vec<f64>,
    m_p: Vec<f64>,
    f: Vec<f64>,
) -> PyResult<Vec<(f64, f64, f64)>> {
    openquant::bet_sizing::bet_size_dynamic_checked(&pos, &max_pos, &m_p, &f).map_err(to_py_err)
}

#[pyfunction(name = "cdf_mixture")]
fn bet_sizing_cdf_mixture(mu1: f64, mu2: f64, sigma1: f64, sigma2: f64, p1: f64, x: f64) -> f64 {
    openquant::bet_sizing::cdf_mixture(mu1, mu2, sigma1, sigma2, p1, x)
}

#[pyfunction(name = "single_bet_size_mixed")]
fn bet_sizing_single_bet_size_mixed(c: f64, fit: [f64; 5]) -> f64 {
    openquant::bet_sizing::single_bet_size_mixed(c, &fit)
}

#[pyfunction(name = "get_concurrent_sides")]
fn bet_sizing_get_concurrent_sides(
    t1_starts: Vec<String>,
    t1_ends: Vec<String>,
    side: Vec<f64>,
) -> PyResult<Vec<(String, f64, f64)>> {
    let starts = parse_naive_datetimes(t1_starts)?;
    let ends = parse_naive_datetimes(t1_ends)?;
    if starts.len() != ends.len() || starts.len() != side.len() {
        return Err(pyo3::exceptions::PyValueError::new_err("t1_starts/t1_ends/side length mismatch"));
    }
    let t1: Vec<(chrono::NaiveDateTime, chrono::NaiveDateTime)> =
        starts.into_iter().zip(ends).collect();
    let result = openquant::bet_sizing::get_concurrent_sides(&t1, &side);
    Ok(result
        .into_iter()
        .map(|(ts, long, short)| (ts.format("%Y-%m-%d %H:%M:%S").to_string(), long, short))
        .collect())
}

#[pyfunction(name = "bet_size_budget")]
fn bet_sizing_bet_size_budget(
    t1_starts: Vec<String>,
    t1_ends: Vec<String>,
    side: Vec<f64>,
) -> PyResult<Vec<(String, f64)>> {
    let starts = parse_naive_datetimes(t1_starts)?;
    let ends = parse_naive_datetimes(t1_ends)?;
    if starts.len() != ends.len() || starts.len() != side.len() {
        return Err(pyo3::exceptions::PyValueError::new_err("t1_starts/t1_ends/side length mismatch"));
    }
    let t1: Vec<(chrono::NaiveDateTime, chrono::NaiveDateTime)> =
        starts.into_iter().zip(ends).collect();
    let result = openquant::bet_sizing::bet_size_budget(&t1, &side);
    Ok(result
        .into_iter()
        .map(|(ts, v)| (ts.format("%Y-%m-%d %H:%M:%S").to_string(), v))
        .collect())
}

#[pyfunction(name = "bet_size_probability")]
fn bet_sizing_bet_size_probability(
    event_starts: Vec<String>,
    event_ends: Vec<String>,
    probs: Vec<f64>,
    sides: Vec<f64>,
    num_classes: usize,
    step_size: f64,
    average_active: bool,
) -> PyResult<Vec<(String, f64)>> {
    let starts = parse_naive_datetimes(event_starts)?;
    let ends = parse_naive_datetimes(event_ends)?;
    if starts.len() != ends.len() || starts.len() != probs.len() || starts.len() != sides.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "event_starts/event_ends/probs/sides length mismatch",
        ));
    }
    let events: Vec<(chrono::NaiveDateTime, chrono::NaiveDateTime, f64, f64)> = starts
        .into_iter()
        .zip(ends)
        .zip(probs)
        .zip(sides)
        .map(|(((s, e), p), sd)| (s, e, p, sd))
        .collect();
    let result = openquant::bet_sizing::bet_size_probability(&events, num_classes, step_size, average_active);
    Ok(result
        .into_iter()
        .map(|(ts, v)| (ts.format("%Y-%m-%d %H:%M:%S").to_string(), v))
        .collect())
}

#[pyfunction(name = "mp_avg_active_signals")]
fn bet_sizing_mp_avg_active_signals(
    signal_timestamps: Vec<String>,
    signal_values: Vec<f64>,
    t1_timestamps: Vec<String>,
    molecule_timestamps: Vec<String>,
) -> PyResult<Vec<(String, f64)>> {
    let signal = pair_timestamps_values(signal_timestamps, signal_values, "signal_timestamps", "signal_values")?;
    let t1 = parse_naive_datetimes(t1_timestamps)?;
    let molecule = parse_naive_datetimes(molecule_timestamps)?;
    let result = openquant::bet_sizing::mp_avg_active_signals(&signal, &t1, &molecule);
    Ok(result
        .into_iter()
        .map(|(ts, v)| (ts.format("%Y-%m-%d %H:%M:%S").to_string(), v))
        .collect())
}

#[pyfunction(name = "bet_size_reserve")]
fn bet_sizing_bet_size_reserve(
    t1_starts: Vec<String>,
    t1_ends: Vec<String>,
    side: Vec<f64>,
    fit: [f64; 5],
) -> PyResult<Vec<(String, f64, f64, f64)>> {
    let starts = parse_naive_datetimes(t1_starts)?;
    let ends = parse_naive_datetimes(t1_ends)?;
    if starts.len() != ends.len() || starts.len() != side.len() {
        return Err(pyo3::exceptions::PyValueError::new_err("t1_starts/t1_ends/side length mismatch"));
    }
    let t1: Vec<(chrono::NaiveDateTime, chrono::NaiveDateTime)> =
        starts.into_iter().zip(ends).collect();
    let result = openquant::bet_sizing::bet_size_reserve(&t1, &side, &fit);
    Ok(result
        .into_iter()
        .map(|(ts, l, s, b)| (ts.format("%Y-%m-%d %H:%M:%S").to_string(), l, s, b))
        .collect())
}

#[pyfunction(name = "bet_size_reserve_with_fit")]
fn bet_sizing_bet_size_reserve_with_fit(
    t1_starts: Vec<String>,
    t1_ends: Vec<String>,
    side: Vec<f64>,
    fit: [f64; 5],
) -> PyResult<Vec<(String, f64, f64, f64, f64)>> {
    let starts = parse_naive_datetimes(t1_starts)?;
    let ends = parse_naive_datetimes(t1_ends)?;
    if starts.len() != ends.len() || starts.len() != side.len() {
        return Err(pyo3::exceptions::PyValueError::new_err("t1_starts/t1_ends/side length mismatch"));
    }
    let t1: Vec<(chrono::NaiveDateTime, chrono::NaiveDateTime)> =
        starts.into_iter().zip(ends).collect();
    let result = openquant::bet_sizing::bet_size_reserve_with_fit(&t1, &side, &fit);
    Ok(result
        .into_iter()
        .map(|(ts, l, s, c, b)| (ts.format("%Y-%m-%d %H:%M:%S").to_string(), l, s, c, b))
        .collect())
}

#[pyfunction(name = "bet_size_reserve_full")]
fn bet_sizing_bet_size_reserve_full(
    t1_starts: Vec<String>,
    t1_ends: Vec<String>,
    side: Vec<f64>,
    fit_runs: usize,
    epsilon: f64,
    max_iter: usize,
    return_parameters: bool,
) -> PyResult<(Vec<(String, f64, f64, f64, f64)>, Option<[f64; 5]>)> {
    let starts = parse_naive_datetimes(t1_starts)?;
    let ends = parse_naive_datetimes(t1_ends)?;
    if starts.len() != ends.len() || starts.len() != side.len() {
        return Err(pyo3::exceptions::PyValueError::new_err("t1_starts/t1_ends/side length mismatch"));
    }
    let t1: Vec<(chrono::NaiveDateTime, chrono::NaiveDateTime)> =
        starts.into_iter().zip(ends).collect();
    let (events, params) = openquant::bet_sizing::bet_size_reserve_full(&t1, &side, fit_runs, epsilon, max_iter, return_parameters);
    let out_events = events
        .into_iter()
        .map(|(ts, l, s, c, b)| (ts.format("%Y-%m-%d %H:%M:%S").to_string(), l, s, c, b))
        .collect();
    Ok((out_events, params))
}

pub fn register(py: Python<'_>, parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(py, "bet_sizing")?;
    m.add_function(wrap_pyfunction!(bet_sizing_get_signal, &m)?)?;
    m.add_function(wrap_pyfunction!(bet_sizing_discrete_signal, &m)?)?;
    m.add_function(wrap_pyfunction!(bet_sizing_bet_size, &m)?)?;
    m.add_function(wrap_pyfunction!(bet_sizing_bet_size_sigmoid, &m)?)?;
    m.add_function(wrap_pyfunction!(bet_sizing_bet_size_power, &m)?)?;
    m.add_function(wrap_pyfunction!(bet_sizing_inv_price, &m)?)?;
    m.add_function(wrap_pyfunction!(bet_sizing_inv_price_sigmoid, &m)?)?;
    m.add_function(wrap_pyfunction!(bet_sizing_inv_price_power, &m)?)?;
    m.add_function(wrap_pyfunction!(bet_sizing_get_w, &m)?)?;
    m.add_function(wrap_pyfunction!(bet_sizing_get_w_sigmoid, &m)?)?;
    m.add_function(wrap_pyfunction!(bet_sizing_get_w_power, &m)?)?;
    m.add_function(wrap_pyfunction!(bet_sizing_get_target_pos, &m)?)?;
    m.add_function(wrap_pyfunction!(bet_sizing_get_target_pos_sigmoid, &m)?)?;
    m.add_function(wrap_pyfunction!(bet_sizing_get_target_pos_power, &m)?)?;
    m.add_function(wrap_pyfunction!(bet_sizing_limit_price, &m)?)?;
    m.add_function(wrap_pyfunction!(bet_sizing_limit_price_sigmoid, &m)?)?;
    m.add_function(wrap_pyfunction!(bet_sizing_limit_price_power, &m)?)?;
    m.add_function(wrap_pyfunction!(bet_sizing_avg_active_signals, &m)?)?;
    m.add_function(wrap_pyfunction!(bet_sizing_bet_size_dynamic, &m)?)?;
    m.add_function(wrap_pyfunction!(bet_sizing_cdf_mixture, &m)?)?;
    m.add_function(wrap_pyfunction!(bet_sizing_single_bet_size_mixed, &m)?)?;
    m.add_function(wrap_pyfunction!(bet_sizing_get_concurrent_sides, &m)?)?;
    m.add_function(wrap_pyfunction!(bet_sizing_bet_size_budget, &m)?)?;
    m.add_function(wrap_pyfunction!(bet_sizing_bet_size_probability, &m)?)?;
    m.add_function(wrap_pyfunction!(bet_sizing_mp_avg_active_signals, &m)?)?;
    m.add_function(wrap_pyfunction!(bet_sizing_bet_size_reserve, &m)?)?;
    m.add_function(wrap_pyfunction!(bet_sizing_bet_size_reserve_with_fit, &m)?)?;
    m.add_function(wrap_pyfunction!(bet_sizing_bet_size_reserve_full, &m)?)?;
    parent.add_submodule(&m)?;
    parent.add("bet_sizing", m)?;
    Ok(())
}

use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::helpers::to_py_err;

#[pyfunction(name = "sharpe_symmetric")]
fn sr_sharpe_symmetric(precision: f64, annual_bet_frequency: f64) -> PyResult<f64> {
    openquant::strategy_risk::sharpe_symmetric(precision, annual_bet_frequency).map_err(to_py_err)
}

#[pyfunction(name = "implied_precision_symmetric")]
fn sr_implied_precision_symmetric(target_sharpe: f64, annual_bet_frequency: f64) -> PyResult<f64> {
    openquant::strategy_risk::implied_precision_symmetric(target_sharpe, annual_bet_frequency)
        .map_err(to_py_err)
}

#[pyfunction(name = "implied_frequency_symmetric")]
fn sr_implied_frequency_symmetric(precision: f64, target_sharpe: f64) -> PyResult<f64> {
    openquant::strategy_risk::implied_frequency_symmetric(precision, target_sharpe)
        .map_err(to_py_err)
}

#[pyfunction(name = "sharpe_asymmetric")]
fn sr_sharpe_asymmetric(
    precision: f64,
    annual_bet_frequency: f64,
    pi_plus: f64,
    pi_minus: f64,
) -> PyResult<f64> {
    let payout = openquant::strategy_risk::AsymmetricPayout { pi_plus, pi_minus };
    openquant::strategy_risk::sharpe_asymmetric(precision, annual_bet_frequency, payout)
        .map_err(to_py_err)
}

#[pyfunction(name = "implied_precision_asymmetric")]
fn sr_implied_precision_asymmetric(
    target_sharpe: f64,
    annual_bet_frequency: f64,
    pi_plus: f64,
    pi_minus: f64,
) -> PyResult<f64> {
    let payout = openquant::strategy_risk::AsymmetricPayout { pi_plus, pi_minus };
    openquant::strategy_risk::implied_precision_asymmetric(target_sharpe, annual_bet_frequency, payout)
        .map_err(to_py_err)
}

#[pyfunction(name = "implied_frequency_asymmetric")]
fn sr_implied_frequency_asymmetric(
    precision: f64,
    target_sharpe: f64,
    pi_plus: f64,
    pi_minus: f64,
) -> PyResult<f64> {
    let payout = openquant::strategy_risk::AsymmetricPayout { pi_plus, pi_minus };
    openquant::strategy_risk::implied_frequency_asymmetric(precision, target_sharpe, payout)
        .map_err(to_py_err)
}

#[pyfunction(name = "estimate_strategy_failure_probability")]
#[pyo3(signature = (
    bet_outcomes,
    years_elapsed,
    target_sharpe,
    investor_horizon_years,
    bootstrap_iterations=1000,
    seed=42,
    kde_bandwidth=None
))]
fn sr_estimate_strategy_failure_probability(
    py: Python<'_>,
    bet_outcomes: Vec<f64>,
    years_elapsed: f64,
    target_sharpe: f64,
    investor_horizon_years: f64,
    bootstrap_iterations: usize,
    seed: u64,
    kde_bandwidth: Option<f64>,
) -> PyResult<PyObject> {
    let cfg = openquant::strategy_risk::StrategyRiskConfig {
        years_elapsed,
        target_sharpe,
        investor_horizon_years,
        bootstrap_iterations,
        seed,
        kde_bandwidth,
    };
    let report = openquant::strategy_risk::estimate_strategy_failure_probability(&bet_outcomes, cfg)
        .map_err(to_py_err)?;

    let d = PyDict::new(py);
    d.set_item("pi_plus", report.payout.pi_plus)?;
    d.set_item("pi_minus", report.payout.pi_minus)?;
    d.set_item("annual_bet_frequency", report.annual_bet_frequency)?;
    d.set_item("implied_precision_threshold", report.implied_precision_threshold)?;
    d.set_item("bootstrap_precision_mean", report.bootstrap_precision_mean)?;
    d.set_item("bootstrap_precision_std", report.bootstrap_precision_std)?;
    d.set_item("empirical_failure_probability", report.empirical_failure_probability)?;
    d.set_item("kde_failure_probability", report.kde_failure_probability)?;
    d.set_item("bootstrap_precision_samples", report.bootstrap_precision_samples)?;
    Ok(d.into_pyobject(py).unwrap().into_any().unbind())
}

pub fn register(py: Python<'_>, parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(py, "strategy_risk")?;
    m.add_function(wrap_pyfunction!(sr_sharpe_symmetric, &m)?)?;
    m.add_function(wrap_pyfunction!(sr_implied_precision_symmetric, &m)?)?;
    m.add_function(wrap_pyfunction!(sr_implied_frequency_symmetric, &m)?)?;
    m.add_function(wrap_pyfunction!(sr_sharpe_asymmetric, &m)?)?;
    m.add_function(wrap_pyfunction!(sr_implied_precision_asymmetric, &m)?)?;
    m.add_function(wrap_pyfunction!(sr_implied_frequency_asymmetric, &m)?)?;
    m.add_function(wrap_pyfunction!(sr_estimate_strategy_failure_probability, &m)?)?;
    parent.add_submodule(&m)?;
    parent.add("strategy_risk", m)?;
    Ok(())
}

use openquant::risk_metrics::RiskMetrics;
use pyo3::prelude::*;

use crate::helpers::{matrix_from_rows, to_py_err};

#[pyfunction(name = "calculate_value_at_risk")]
fn risk_calculate_value_at_risk(returns: Vec<f64>, confidence_level: f64) -> PyResult<f64> {
    RiskMetrics.calculate_value_at_risk(&returns, confidence_level).map_err(to_py_err)
}

#[pyfunction(name = "calculate_expected_shortfall")]
fn risk_calculate_expected_shortfall(returns: Vec<f64>, confidence_level: f64) -> PyResult<f64> {
    RiskMetrics.calculate_expected_shortfall(&returns, confidence_level).map_err(to_py_err)
}

#[pyfunction(name = "calculate_conditional_drawdown_risk")]
fn risk_calculate_conditional_drawdown_risk(
    returns: Vec<f64>,
    confidence_level: f64,
) -> PyResult<f64> {
    RiskMetrics
        .calculate_conditional_drawdown_risk(&returns, confidence_level)
        .map_err(to_py_err)
}

#[pyfunction(name = "calculate_variance")]
fn risk_calculate_variance(
    covariance: Vec<Vec<f64>>,
    weights: Vec<f64>,
) -> PyResult<f64> {
    let cov = matrix_from_rows(covariance)?;
    RiskMetrics.calculate_variance(&cov, &weights).map_err(to_py_err)
}

#[pyfunction(name = "calculate_value_at_risk_from_matrix")]
fn risk_calculate_value_at_risk_from_matrix(
    returns: Vec<Vec<f64>>,
    confidence_level: f64,
) -> PyResult<f64> {
    let m = matrix_from_rows(returns)?;
    RiskMetrics.calculate_value_at_risk_from_matrix(&m, confidence_level).map_err(to_py_err)
}

#[pyfunction(name = "calculate_expected_shortfall_from_matrix")]
fn risk_calculate_expected_shortfall_from_matrix(
    returns: Vec<Vec<f64>>,
    confidence_level: f64,
) -> PyResult<f64> {
    let m = matrix_from_rows(returns)?;
    RiskMetrics.calculate_expected_shortfall_from_matrix(&m, confidence_level).map_err(to_py_err)
}

#[pyfunction(name = "calculate_conditional_drawdown_risk_from_matrix")]
fn risk_calculate_conditional_drawdown_risk_from_matrix(
    returns: Vec<Vec<f64>>,
    confidence_level: f64,
) -> PyResult<f64> {
    let m = matrix_from_rows(returns)?;
    RiskMetrics
        .calculate_conditional_drawdown_risk_from_matrix(&m, confidence_level)
        .map_err(to_py_err)
}

pub fn register(py: Python<'_>, parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(py, "risk")?;
    m.add_function(wrap_pyfunction!(risk_calculate_value_at_risk, &m)?)?;
    m.add_function(wrap_pyfunction!(risk_calculate_expected_shortfall, &m)?)?;
    m.add_function(wrap_pyfunction!(risk_calculate_conditional_drawdown_risk, &m)?)?;
    m.add_function(wrap_pyfunction!(risk_calculate_variance, &m)?)?;
    m.add_function(wrap_pyfunction!(risk_calculate_value_at_risk_from_matrix, &m)?)?;
    m.add_function(wrap_pyfunction!(risk_calculate_expected_shortfall_from_matrix, &m)?)?;
    m.add_function(wrap_pyfunction!(risk_calculate_conditional_drawdown_risk_from_matrix, &m)?)?;
    parent.add_submodule(&m)?;
    parent.add("risk", m)?;
    Ok(())
}

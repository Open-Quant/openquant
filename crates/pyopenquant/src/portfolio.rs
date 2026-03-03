use pyo3::prelude::*;
use std::collections::HashMap;

use crate::helpers::{matrix_from_rows, to_py_err};

fn parse_bounds(bounds: Option<Vec<(usize, f64, f64)>>) -> Option<HashMap<usize, (f64, f64)>> {
    bounds.map(|v| v.into_iter().map(|(i, lo, hi)| (i, (lo, hi))).collect())
}

#[pyfunction(name = "allocate_inverse_variance")]
fn portfolio_allocate_inverse_variance(
    prices: Vec<Vec<f64>>,
) -> PyResult<(Vec<f64>, f64, f64, f64)> {
    let m = matrix_from_rows(prices)?;
    let out = openquant::portfolio_optimization::allocate_inverse_variance(&m).map_err(to_py_err)?;
    Ok((out.weights, out.portfolio_risk, out.portfolio_return, out.portfolio_sharpe))
}

#[pyfunction(name = "allocate_min_vol")]
#[pyo3(signature = (prices, bounds=None, tuple_bounds=None))]
fn portfolio_allocate_min_vol(
    prices: Vec<Vec<f64>>,
    bounds: Option<Vec<(usize, f64, f64)>>,
    tuple_bounds: Option<(f64, f64)>,
) -> PyResult<(Vec<f64>, f64, f64, f64)> {
    let m = matrix_from_rows(prices)?;
    let out = openquant::portfolio_optimization::allocate_min_vol(&m, parse_bounds(bounds), tuple_bounds).map_err(to_py_err)?;
    Ok((out.weights, out.portfolio_risk, out.portfolio_return, out.portfolio_sharpe))
}

#[pyfunction(name = "allocate_max_sharpe")]
#[pyo3(signature = (prices, risk_free_rate=None, bounds=None, tuple_bounds=None))]
fn portfolio_allocate_max_sharpe(
    prices: Vec<Vec<f64>>,
    risk_free_rate: Option<f64>,
    bounds: Option<Vec<(usize, f64, f64)>>,
    tuple_bounds: Option<(f64, f64)>,
) -> PyResult<(Vec<f64>, f64, f64, f64)> {
    let m = matrix_from_rows(prices)?;
    let out = openquant::portfolio_optimization::allocate_max_sharpe(
        &m,
        risk_free_rate.unwrap_or(0.0),
        parse_bounds(bounds),
        tuple_bounds,
    )
    .map_err(to_py_err)?;
    Ok((out.weights, out.portfolio_risk, out.portfolio_return, out.portfolio_sharpe))
}

#[pyfunction(name = "allocate_efficient_risk")]
#[pyo3(signature = (prices, target_return, bounds=None, tuple_bounds=None))]
fn portfolio_allocate_efficient_risk(
    prices: Vec<Vec<f64>>,
    target_return: f64,
    bounds: Option<Vec<(usize, f64, f64)>>,
    tuple_bounds: Option<(f64, f64)>,
) -> PyResult<(Vec<f64>, f64, f64, f64)> {
    let m = matrix_from_rows(prices)?;
    let out = openquant::portfolio_optimization::allocate_efficient_risk(
        &m,
        target_return,
        parse_bounds(bounds),
        tuple_bounds,
    )
    .map_err(to_py_err)?;
    Ok((out.weights, out.portfolio_risk, out.portfolio_return, out.portfolio_sharpe))
}

#[pyfunction(name = "allocate_with_solution")]
#[pyo3(signature = (
    prices,
    solution,
    risk_free_rate=0.0,
    target_return=0.0,
    bounds=None,
    tuple_bounds=None,
    resample_by=None,
    returns_method=None
))]
fn portfolio_allocate_with_solution(
    prices: Vec<Vec<f64>>,
    solution: String,
    risk_free_rate: f64,
    target_return: f64,
    bounds: Option<Vec<(usize, f64, f64)>>,
    tuple_bounds: Option<(f64, f64)>,
    resample_by: Option<String>,
    returns_method: Option<String>,
) -> PyResult<(Vec<f64>, f64, f64, f64)> {
    let m = matrix_from_rows(prices)?;
    let rm = match returns_method {
        Some(name) => openquant::portfolio_optimization::returns_method_from_str(&name).map_err(to_py_err)?,
        None => openquant::portfolio_optimization::ReturnsMethod::Mean,
    };
    let opts = openquant::portfolio_optimization::AllocationOptions {
        risk_free_rate,
        target_return,
        bounds: parse_bounds(bounds),
        tuple_bounds,
        resample_by: resample_by.as_deref(),
        returns_method: rm,
    };
    let out = openquant::portfolio_optimization::allocate_with_solution(&m, &solution, &opts).map_err(to_py_err)?;
    Ok((out.weights, out.portfolio_risk, out.portfolio_return, out.portfolio_sharpe))
}

#[pyfunction(name = "allocate_from_inputs")]
#[pyo3(signature = (
    expected_returns,
    covariance,
    solution,
    risk_free_rate=0.0,
    target_return=0.0,
    bounds=None,
    tuple_bounds=None
))]
fn portfolio_allocate_from_inputs(
    expected_returns: Vec<f64>,
    covariance: Vec<Vec<f64>>,
    solution: String,
    risk_free_rate: f64,
    target_return: f64,
    bounds: Option<Vec<(usize, f64, f64)>>,
    tuple_bounds: Option<(f64, f64)>,
) -> PyResult<(Vec<f64>, f64, f64, f64)> {
    let cov = matrix_from_rows(covariance)?;
    let opts = openquant::portfolio_optimization::AllocationOptions {
        risk_free_rate,
        target_return,
        bounds: parse_bounds(bounds),
        tuple_bounds,
        resample_by: None,
        returns_method: openquant::portfolio_optimization::ReturnsMethod::Mean,
    };
    let out = openquant::portfolio_optimization::allocate_from_inputs(
        &expected_returns,
        &cov,
        &solution,
        &opts,
    )
    .map_err(to_py_err)?;
    Ok((out.weights, out.portfolio_risk, out.portfolio_return, out.portfolio_sharpe))
}

pub fn register(py: Python<'_>, parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(py, "portfolio")?;
    m.add_function(wrap_pyfunction!(portfolio_allocate_inverse_variance, &m)?)?;
    m.add_function(wrap_pyfunction!(portfolio_allocate_min_vol, &m)?)?;
    m.add_function(wrap_pyfunction!(portfolio_allocate_max_sharpe, &m)?)?;
    m.add_function(wrap_pyfunction!(portfolio_allocate_efficient_risk, &m)?)?;
    m.add_function(wrap_pyfunction!(portfolio_allocate_with_solution, &m)?)?;
    m.add_function(wrap_pyfunction!(portfolio_allocate_from_inputs, &m)?)?;
    parent.add_submodule(&m)?;
    parent.add("portfolio", m)?;
    Ok(())
}

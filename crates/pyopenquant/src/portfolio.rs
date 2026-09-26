use pyo3::prelude::*;
use std::collections::HashMap;

use crate::helpers::{matrix_from_rows, to_py_err};

fn parse_bounds(bounds: Option<Vec<(usize, f64, f64)>>) -> Option<HashMap<usize, (f64, f64)>> {
    bounds.map(|v| v.into_iter().map(|(i, lo, hi)| (i, (lo, hi))).collect())
}

/// Inverse-variance portfolio from a price matrix, long-only and fully invested.
///
/// Weights are `w_i = (1 / Sigma_ii) / sum_j(1 / Sigma_jj)` (correlation ignored), where
/// `Sigma` is the sample covariance of simple returns `p_t / p_{t-1} - 1`. Expected
/// returns and covariance are annualised with 252 periods a year, so the reported figures
/// are annual. Markowitz (1952); see AFML chapter 16 for why mean-variance portfolios are
/// fragile.
///
/// Parameters
/// ----------
/// prices : list[list[float]]
///     Price matrix, one inner list per date (ascending in time, at least 2 rows) and one
///     column per asset.
///
/// Returns
/// -------
/// tuple[list[float], float, float, float]
///     `(weights, portfolio_risk, portfolio_return, portfolio_sharpe)`: weights per asset in
///     column order summing to 1, annualised volatility `sqrt(w' Sigma w)`, annualised
///     expected return `mu' w`, and `portfolio_return / portfolio_risk` (0 when the risk is
///     0).
///
/// Raises
/// ------
/// ValueError
///     If `prices` is empty or ragged, or the core rejects the input (e.g. fewer than two
///     rows, a zero price used as a return denominator, or an asset with zero return
///     variance).
#[pyfunction(name = "allocate_inverse_variance")]
fn portfolio_allocate_inverse_variance(
    prices: Vec<Vec<f64>>,
) -> PyResult<(Vec<f64>, f64, f64, f64)> {
    let m = matrix_from_rows(prices)?;
    let out =
        openquant::portfolio_optimization::allocate_inverse_variance(&m).map_err(to_py_err)?;
    Ok((out.weights, out.portfolio_risk, out.portfolio_return, out.portfolio_sharpe))
}

/// Minimum-volatility portfolio from a price matrix, `min w' Sigma w` with weight bounds.
///
/// Solves the quadratic programme subject to `sum(w) = 1` and the bounds (the bounds are
/// part of the problem, not applied afterwards). Returns are simple returns and both the
/// expected returns and covariance are annualised with 252 periods a year. Markowitz
/// (1952); AFML chapter 16.
///
/// Parameters
/// ----------
/// prices : list[list[float]]
///     Price matrix, one inner list per date (ascending in time, at least 2 rows) and one
///     column per asset.
/// bounds : list[tuple[int, float, float]] | None, default None
///     Per-asset `(asset_index, lower, upper)` weight bounds; they override `tuple_bounds`
///     for those assets. Out-of-range indices are ignored.
/// tuple_bounds : tuple[float, float] | None, default None
///     `(lower, upper)` bounds for every asset not in `bounds`; None means `(0, 1)`. Upper
///     bounds above 1 are treated as 1; negative lower bounds allow shorting.
///
/// Returns
/// -------
/// tuple[list[float], float, float, float]
///     `(weights, portfolio_risk, portfolio_return, portfolio_sharpe)`: weights per asset in
///     column order summing to 1, annualised volatility, annualised expected return, and
///     `portfolio_return / portfolio_risk` (0 when the risk is 0).
///
/// Raises
/// ------
/// ValueError
///     If `prices` is empty or ragged, or the core rejects the input (e.g. fewer than two
///     rows, a zero price, bounds that cannot sum to 1, or no feasible or convergent
///     solution).
#[pyfunction(name = "allocate_min_vol")]
#[pyo3(signature = (prices, bounds=None, tuple_bounds=None))]
fn portfolio_allocate_min_vol(
    prices: Vec<Vec<f64>>,
    bounds: Option<Vec<(usize, f64, f64)>>,
    tuple_bounds: Option<(f64, f64)>,
) -> PyResult<(Vec<f64>, f64, f64, f64)> {
    let m = matrix_from_rows(prices)?;
    let out =
        openquant::portfolio_optimization::allocate_min_vol(&m, parse_bounds(bounds), tuple_bounds)
            .map_err(to_py_err)?;
    Ok((out.weights, out.portfolio_risk, out.portfolio_return, out.portfolio_sharpe))
}

/// Maximum-Sharpe portfolio from a price matrix, with weight bounds.
///
/// Maximises `(mu' w - rf) / sqrt(w' Sigma w)` subject to `sum(w) = 1` and the bounds, via
/// the substitution `y = kappa w`. Returns are simple returns and both `mu` and `Sigma` are
/// annualised with 252 periods a year, so `risk_free_rate` is an annual rate. The Sharpe
/// ratio is in-sample. Markowitz (1952); AFML chapter 16.
///
/// Parameters
/// ----------
/// prices : list[list[float]]
///     Price matrix, one inner list per date (ascending in time, at least 2 rows) and one
///     column per asset.
/// risk_free_rate : float | None, default None
///     Annual risk-free rate; None means 0.0.
/// bounds : list[tuple[int, float, float]] | None, default None
///     Per-asset `(asset_index, lower, upper)` weight bounds; they override `tuple_bounds`
///     for those assets. Out-of-range indices are ignored.
/// tuple_bounds : tuple[float, float] | None, default None
///     `(lower, upper)` bounds for every asset not in `bounds`; None means `(0, 1)`. Upper
///     bounds above 1 are treated as 1; negative lower bounds allow shorting.
///
/// Returns
/// -------
/// tuple[list[float], float, float, float]
///     `(weights, portfolio_risk, portfolio_return, portfolio_sharpe)`: weights per asset in
///     column order summing to 1, annualised volatility, annualised expected return, and
///     `(portfolio_return - risk_free_rate) / portfolio_risk`.
///
/// Raises
/// ------
/// ValueError
///     If `prices` is empty or ragged, or the core rejects the input (e.g. fewer than two
///     rows, a zero price, bounds that cannot sum to 1, no asset with an expected return
///     above `risk_free_rate`, or a degenerate solution).
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

/// Least-risk portfolio from a price matrix whose expected return is at least a target.
///
/// Solves `min w' Sigma w` subject to `mu' w >= target_return`, `sum(w) = 1` and the
/// bounds. A target below the minimum-variance portfolio's return yields that portfolio.
/// Returns are simple returns and both `mu` and `Sigma` are annualised with 252 periods a
/// year, so `target_return` is an annual figure. Markowitz (1952); AFML chapter 16.
///
/// Parameters
/// ----------
/// prices : list[list[float]]
///     Price matrix, one inner list per date (ascending in time, at least 2 rows) and one
///     column per asset.
/// target_return : float
///     Minimum annualised expected return (a floor, not an equality).
/// bounds : list[tuple[int, float, float]] | None, default None
///     Per-asset `(asset_index, lower, upper)` weight bounds; they override `tuple_bounds`
///     for those assets. Out-of-range indices are ignored.
/// tuple_bounds : tuple[float, float] | None, default None
///     `(lower, upper)` bounds for every asset not in `bounds`; None means `(0, 1)`. Upper
///     bounds above 1 are treated as 1; negative lower bounds allow shorting.
///
/// Returns
/// -------
/// tuple[list[float], float, float, float]
///     `(weights, portfolio_risk, portfolio_return, portfolio_sharpe)`: weights per asset in
///     column order summing to 1, annualised volatility, annualised expected return, and
///     `portfolio_return / portfolio_risk` (0 when the risk is 0).
///
/// Raises
/// ------
/// ValueError
///     If `prices` is empty or ragged, or the core rejects the input (e.g. fewer than two
///     rows, a zero price, bounds that cannot sum to 1, or an unreachable
///     `target_return`).
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

/// Allocate a mean-variance portfolio from a price matrix with a named solution.
///
/// Estimates annualised expected returns `mu` and covariance `Sigma` from simple returns of
/// the (optionally resampled) prices, both scaled by `252 / step`, then solves one of:
/// `"inverse_variance"` (weights proportional to `1 / Sigma_ii`, projected onto the
/// bounds), `"min_volatility"`, `"max_sharpe"` or `"efficient_risk"` (least risk with
/// `mu' w >= target_return`). Weights are fully invested; `risk_free_rate` and
/// `target_return` are annual. Markowitz (1952); AFML chapter 16.
///
/// Parameters
/// ----------
/// prices : list[list[float]]
///     Price matrix, one inner list per date (ascending in time) and one column per asset.
/// solution : str
///     `"inverse_variance"`, `"min_volatility"`, `"max_sharpe"` or `"efficient_risk"`.
/// risk_free_rate : float, default 0.0
///     Annual risk-free rate for `"max_sharpe"` and for the reported Sharpe ratio.
/// target_return : float, default 0.0
///     Minimum annual expected return for `"efficient_risk"`; ignored otherwise.
/// bounds : list[tuple[int, float, float]] | None, default None
///     Per-asset `(asset_index, lower, upper)` weight bounds; they override `tuple_bounds`
///     for those assets. Out-of-range indices are ignored.
/// tuple_bounds : tuple[float, float] | None, default None
///     `(lower, upper)` bounds for every asset not in `bounds`; None means `(0, 1)`. Upper
///     bounds above 1 are treated as 1; negative lower bounds allow shorting.
/// resample_by : str | None, default None
///     `"W"`/`"week"`/`"weekly"` keeps every 5th row, `"M"`/`"month"`/`"monthly"` every
///     21st (case-insensitive); anything else, or None, keeps every row.
/// returns_method : str | None, default None
///     `"mean"`/`"mean_historical"` (arithmetic mean, the default) or
///     `"exponential"`/`"exponential_historical"` (exponentially weighted mean, span 500);
///     case-insensitive.
///
/// Returns
/// -------
/// tuple[list[float], float, float, float]
///     `(weights, portfolio_risk, portfolio_return, portfolio_sharpe)`: weights per asset in
///     column order summing to 1, annualised volatility, annualised expected return, and
///     `(portfolio_return - risk_free_rate) / portfolio_risk` (0 when the risk is 0).
///
/// Raises
/// ------
/// ValueError
///     If `prices` is empty or ragged, `returns_method` is unknown, or the core rejects the
///     input (e.g. an unknown `solution`, fewer than two rows after resampling, a zero
///     price, bounds that cannot sum to 1, an unreachable `target_return`, no asset above
///     `risk_free_rate` for `"max_sharpe"`, or a zero variance for `"inverse_variance"`).
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
// Python keyword signature.
#[allow(clippy::too_many_arguments)]
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
        Some(name) => {
            openquant::portfolio_optimization::returns_method_from_str(&name).map_err(to_py_err)?
        }
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
    let out = openquant::portfolio_optimization::allocate_with_solution(&m, &solution, &opts)
        .map_err(to_py_err)?;
    Ok((out.weights, out.portfolio_risk, out.portfolio_return, out.portfolio_sharpe))
}

/// Allocate a mean-variance portfolio from given expected returns and covariance.
///
/// Same solutions as `allocate_with_solution` (`"inverse_variance"`, `"min_volatility"`,
/// `"max_sharpe"`, `"efficient_risk"`), but no returns are estimated and nothing is
/// annualised: the units are the caller's, and `expected_returns`, `covariance`,
/// `risk_free_rate` and `target_return` must agree (an annual `mu` with a daily covariance
/// overstates the Sharpe ratio by `sqrt(252)`). Markowitz (1952); AFML chapter 16.
///
/// Parameters
/// ----------
/// expected_returns : list[float]
///     Expected return per asset.
/// covariance : list[list[float]]
///     Square covariance matrix in the same asset order, one inner list per row.
/// solution : str
///     `"inverse_variance"`, `"min_volatility"`, `"max_sharpe"` or `"efficient_risk"`.
/// risk_free_rate : float, default 0.0
///     Risk-free rate for `"max_sharpe"` and the reported Sharpe ratio.
/// target_return : float, default 0.0
///     Minimum expected return for `"efficient_risk"`; ignored otherwise.
/// bounds : list[tuple[int, float, float]] | None, default None
///     Per-asset `(asset_index, lower, upper)` weight bounds; they override `tuple_bounds`
///     for those assets. Out-of-range indices are ignored.
/// tuple_bounds : tuple[float, float] | None, default None
///     `(lower, upper)` bounds for every asset not in `bounds`; None means `(0, 1)`. Upper
///     bounds above 1 are treated as 1; negative lower bounds allow shorting.
///
/// Returns
/// -------
/// tuple[list[float], float, float, float]
///     `(weights, portfolio_risk, portfolio_return, portfolio_sharpe)` in the caller's
///     units: weights summing to 1, `sqrt(w' Sigma w)`, `mu' w`, and
///     `(portfolio_return - risk_free_rate) / portfolio_risk` (0 when the risk is 0).
///
/// Raises
/// ------
/// ValueError
///     If `covariance` is empty or ragged, or the core rejects the input (e.g. a
///     non-square covariance or one whose size differs from `expected_returns`, an unknown
///     `solution`, bounds that cannot sum to 1, an unreachable `target_return`, or no
///     asset above `risk_free_rate` for `"max_sharpe"`).
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

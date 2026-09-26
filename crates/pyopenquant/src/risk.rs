use openquant::risk_metrics::RiskMetrics;
use pyo3::prelude::*;

use crate::helpers::{matrix_from_rows, to_py_err};

/// Historical value at risk: the "higher" `confidence_level`-quantile of `returns`.
///
/// No distribution is fitted. The quantile is the sorted element at index
/// `ceil(confidence_level * (n - 1))` (as `numpy.quantile(..., method="higher")`), never
/// interpolated, so with 20 returns the 5% VaR is the second-worst. `confidence_level` is the
/// lower-tail probability (0.05 for a 5% VaR, not 0.95). The result is a return with its sign
/// (typically negative: -0.0166 is a 1.66% loss) and is not annualised or horizon-scaled. NaNs
/// count toward `n` and shift the quantile; drop them first.
///
/// Parameters
/// ----------
/// returns : list[float]
///     Per-period returns, in any order.
/// confidence_level : float
///     Lower-tail probability in `[0, 1]`.
///
/// Returns
/// -------
/// float
///     The VaR as a signed return.
///
/// Raises
/// ------
/// ValueError
///     If `confidence_level` is outside `[0, 1]` (or NaN), or `returns` is empty.
#[pyfunction(name = "calculate_value_at_risk")]
fn risk_calculate_value_at_risk(returns: Vec<f64>, confidence_level: f64) -> PyResult<f64> {
    RiskMetrics.calculate_value_at_risk(&returns, confidence_level).map_err(to_py_err)
}

/// Historical expected shortfall (conditional VaR): the mean of the returns strictly below the VaR.
///
/// The VaR is `calculate_value_at_risk(returns, confidence_level)`, so `confidence_level` is
/// the lower-tail probability (0.05, not 0.95) and the result is a signed return (typically
/// negative). If no return lies strictly below the VaR (constant returns, or
/// `confidence_level=0`), the result is NaN rather than an error.
///
/// Parameters
/// ----------
/// returns : list[float]
///     Per-period returns, in any order.
/// confidence_level : float
///     Lower-tail probability in `[0, 1]`.
///
/// Returns
/// -------
/// float
///     The expected shortfall as a signed return, or NaN for an empty tail.
///
/// Raises
/// ------
/// ValueError
///     If `confidence_level` is outside `[0, 1]` (or NaN), or `returns` is empty.
#[pyfunction(name = "calculate_expected_shortfall")]
fn risk_calculate_expected_shortfall(returns: Vec<f64>, confidence_level: f64) -> PyResult<f64> {
    RiskMetrics.calculate_expected_shortfall(&returns, confidence_level).map_err(to_py_err)
}

/// Conditional drawdown at risk: the mean of the worst `1 - confidence_level` share of drawdowns.
///
/// Chekhlov, Uryasev and Zabarankin (2005). The drawdown is `running_max(returns) - returns`,
/// in the units of the input. The threshold is the "higher" `confidence_level`-quantile of the
/// drawdowns and every drawdown at or above it is averaged, so the tail is never empty.
///
/// Despite its name (kept so existing keyword calls work), `returns` must be a cumulative
/// series such as an equity curve, a price or a cumulative return, not per-period returns.
/// `confidence_level` is the upper-tail level (0.95 averages the worst 5%), the opposite of
/// `calculate_value_at_risk`, which takes the lower-tail probability (0.05).
///
/// Parameters
/// ----------
/// returns : list[float]
///     Cumulative series (equity curve, price or cumulative return), oldest first.
/// confidence_level : float
///     Upper-tail level in `[0, 1]`.
///
/// Returns
/// -------
/// float
///     The conditional drawdown at risk, in the units of `returns` (NaN only if the input
///     contains NaN).
///
/// Raises
/// ------
/// ValueError
///     If `confidence_level` is outside `[0, 1]` (or NaN), or `returns` is empty.
#[pyfunction(name = "calculate_conditional_drawdown_risk")]
fn risk_calculate_conditional_drawdown_risk(
    returns: Vec<f64>,
    confidence_level: f64,
) -> PyResult<f64> {
    RiskMetrics.calculate_conditional_drawdown_risk(&returns, confidence_level).map_err(to_py_err)
}

/// Portfolio variance `w' C w` for weights `weights` and covariance matrix `covariance`.
///
/// The result is in the units of the covariance (e.g. per-period variance of returns).
/// Weights are used as given, not normalised.
///
/// Parameters
/// ----------
/// covariance : list[list[float]]
///     Square covariance matrix, one inner list per row, assets in the same order as `weights`.
/// weights : list[float]
///     Portfolio weights.
///
/// Returns
/// -------
/// float
///     The portfolio variance.
///
/// Raises
/// ------
/// ValueError
///     If `covariance` is empty or ragged, is not square, or its size differs from
///     `len(weights)`.
#[pyfunction(name = "calculate_variance")]
fn risk_calculate_variance(covariance: Vec<Vec<f64>>, weights: Vec<f64>) -> PyResult<f64> {
    let cov = matrix_from_rows(covariance)?;
    RiskMetrics.calculate_variance(&cov, &weights).map_err(to_py_err)
}

/// `calculate_value_at_risk` on the first column of a matrix of returns.
///
/// Only the first column is read; other columns are silently ignored. See
/// `calculate_value_at_risk` for the quantile rule and sign conventions (`confidence_level` is
/// the lower-tail probability, e.g. 0.05).
///
/// Parameters
/// ----------
/// returns : list[list[float]]
///     Returns matrix, one inner list per row (period); column 0 is used.
/// confidence_level : float
///     Lower-tail probability in `[0, 1]`.
///
/// Returns
/// -------
/// float
///     The VaR of the first column as a signed return.
///
/// Raises
/// ------
/// ValueError
///     If `returns` has no rows or no columns or is ragged, or `confidence_level` is outside
///     `[0, 1]` (or NaN).
#[pyfunction(name = "calculate_value_at_risk_from_matrix")]
fn risk_calculate_value_at_risk_from_matrix(
    returns: Vec<Vec<f64>>,
    confidence_level: f64,
) -> PyResult<f64> {
    let m = matrix_from_rows(returns)?;
    RiskMetrics.calculate_value_at_risk_from_matrix(&m, confidence_level).map_err(to_py_err)
}

/// `calculate_expected_shortfall` on the first column of a matrix of returns.
///
/// Only the first column is read; other columns are silently ignored. `confidence_level` is the
/// lower-tail probability (e.g. 0.05); the result is NaN if no return lies strictly below the
/// VaR.
///
/// Parameters
/// ----------
/// returns : list[list[float]]
///     Returns matrix, one inner list per row (period); column 0 is used.
/// confidence_level : float
///     Lower-tail probability in `[0, 1]`.
///
/// Returns
/// -------
/// float
///     The expected shortfall of the first column as a signed return, or NaN for an empty tail.
///
/// Raises
/// ------
/// ValueError
///     If `returns` has no rows or no columns or is ragged, or `confidence_level` is outside
///     `[0, 1]` (or NaN).
#[pyfunction(name = "calculate_expected_shortfall_from_matrix")]
fn risk_calculate_expected_shortfall_from_matrix(
    returns: Vec<Vec<f64>>,
    confidence_level: f64,
) -> PyResult<f64> {
    let m = matrix_from_rows(returns)?;
    RiskMetrics.calculate_expected_shortfall_from_matrix(&m, confidence_level).map_err(to_py_err)
}

/// `calculate_conditional_drawdown_risk` on the first column of a matrix.
///
/// Only the first column is read; other columns are silently ignored. That column must be a
/// cumulative series (equity curve, price or cumulative return), not per-period returns, and
/// `confidence_level` is the upper-tail level (0.95 averages the worst 5% of drawdowns).
///
/// Parameters
/// ----------
/// returns : list[list[float]]
///     Matrix whose first column is a cumulative series, one inner list per row, oldest first.
/// confidence_level : float
///     Upper-tail level in `[0, 1]`.
///
/// Returns
/// -------
/// float
///     The conditional drawdown at risk of the first column, in its units.
///
/// Raises
/// ------
/// ValueError
///     If `returns` has no rows or no columns or is ragged, or `confidence_level` is outside
///     `[0, 1]` (or NaN).
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

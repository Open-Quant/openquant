use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::helpers::{matrix_from_rows, to_py_err};

/// Critical Line Algorithm portfolios (Markowitz; Bailey and López de Prado, 2013; AFML
/// chapter 16).
///
/// Computes every turning point of the efficient frontier exactly under one box bound for
/// all assets and the budget constraint (weights sum to 1), walking from the maximum-return
/// portfolio down to the minimum-variance portfolio, then returns the requested `solution`.
/// Supply either `asset_prices` (expected returns and covariance are estimated from their
/// simple returns, annualised with `252 / step` periods per year) or both
/// `expected_returns` and `covariance_matrix` (used as given, not annualised). If every
/// expected return is identical, `1e-5` is added to the last one so the walk has a starting
/// asset.
///
/// Parameters
/// ----------
/// asset_prices : list[list[float]] | None, default None
///     Prices, one inner list per observation (oldest first) and one column per asset.
///     Required unless both `expected_returns` and `covariance_matrix` are given.
/// expected_returns : list[float] | None, default None
///     Expected return per asset. When given it is used instead of an estimate from prices.
/// covariance_matrix : list[list[float]] | None, default None
///     `N x N` covariance. When given it is used instead of an estimate from prices.
/// weight_bounds_lower : float | None, default None
///     Lower bound on every weight; None means 0.
/// weight_bounds_upper : float | None, default None
///     Upper bound on every weight; None means 1.
/// resample_by : str | None, default None
///     For prices only, positional resampling (case-insensitive): `"W"`/`"week"`/`"weekly"`
///     keeps every 5th row (`step = 5`), `"M"`/`"month"`/`"monthly"` every 21st
///     (`step = 21`); anything else, or None, keeps every row (`step = 1`).
/// solution : str | None, default None
///     `"cla_turning_points"` (the default when None), `"min_volatility"`, `"max_sharpe"`
///     (maximises `mu'w / sigma`, i.e. a zero risk-free rate) or `"efficient_frontier"`
///     (about 100 points along the frontier).
/// calculate_expected_returns : str, default "mean"
///     How expected returns are estimated from prices: `"mean"` (mean periodic return) or
///     `"exponential"` (exponentially weighted mean, span 500, seeded at the first return).
///     Ignored when `expected_returns` is given.
///
/// Returns
/// -------
/// dict[str, list]
///     `weights`: the requested portfolios, one list of `N` weights each, in input column
///     order (every turning point, maximum return first, for `"cla_turning_points"`; one
///     portfolio for `"min_volatility"` and `"max_sharpe"`; the frontier points for
///     `"efficient_frontier"`). `lambdas`: the risk-aversion parameter at each turning point,
///     infinite first and falling to 0. `efficient_frontier_means` and
///     `efficient_frontier_sigma`: the expected return and volatility of each frontier point
///     (empty unless `solution="efficient_frontier"`).
///
/// Raises
/// ------
/// ValueError
///     If a matrix is empty or ragged; no prices are given and either expected returns or
///     covariance is missing; prices have fewer than two (resampled) rows or a zero price;
///     `calculate_expected_returns` or `solution` is unknown; the inputs disagree on the
///     number of assets; the bounds are infeasible (not finite, lower above upper, or unable
///     to sum to 1); or the covariance of the free assets is singular at some step of the
///     walk (e.g. two perfectly correlated assets, or more assets than observations).
#[pyfunction(name = "allocate_cla")]
#[pyo3(signature = (
    asset_prices=None,
    expected_returns=None,
    covariance_matrix=None,
    weight_bounds_lower=None,
    weight_bounds_upper=None,
    resample_by=None,
    solution=None,
    calculate_expected_returns="mean"
))]
// Python keyword signature.
#[allow(clippy::too_many_arguments)]
fn cla_allocate(
    py: Python<'_>,
    asset_prices: Option<Vec<Vec<f64>>>,
    expected_returns: Option<Vec<f64>>,
    covariance_matrix: Option<Vec<Vec<f64>>>,
    weight_bounds_lower: Option<f64>,
    weight_bounds_upper: Option<f64>,
    resample_by: Option<String>,
    solution: Option<String>,
    calculate_expected_returns: &str,
) -> PyResult<PyObject> {
    let lb = weight_bounds_lower.unwrap_or(0.0);
    let ub = weight_bounds_upper.unwrap_or(1.0);
    let wb = openquant::cla::WeightBounds::Tuple(lb, ub);

    let mut cla = openquant::cla::CLA::new(wb, calculate_expected_returns);

    let prices_m = asset_prices.map(matrix_from_rows).transpose()?;
    let cov_m = covariance_matrix.map(matrix_from_rows).transpose()?;
    let expected_ret_m = expected_returns.map(|v| nalgebra::DMatrix::from_vec(v.len(), 1, v));

    cla.allocate(
        prices_m.as_ref().map(openquant::cla::AssetPricesInput::RawMatrix),
        expected_ret_m.as_ref(),
        cov_m.as_ref(),
        resample_by.as_deref(),
        solution.as_deref(),
    )
    .map_err(to_py_err)?;

    let d = PyDict::new(py);
    d.set_item("weights", &cla.weights)?;
    d.set_item("lambdas", &cla.lambdas)?;
    d.set_item("efficient_frontier_means", &cla.efficient_frontier_means)?;
    d.set_item("efficient_frontier_sigma", &cla.efficient_frontier_sigma)?;
    Ok(d.into_pyobject(py).unwrap().into_any().unbind())
}

pub fn register(py: Python<'_>, parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(py, "cla")?;
    m.add_function(wrap_pyfunction!(cla_allocate, &m)?)?;
    parent.add_submodule(&m)?;
    parent.add("cla", m)?;
    Ok(())
}

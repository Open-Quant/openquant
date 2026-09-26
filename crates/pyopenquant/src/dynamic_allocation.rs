use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use openquant::dynamic_allocation::{self as da, DynamicAllocationConfig, HorizonForecast};

use nalgebra::DMatrix;

use crate::helpers::to_py_err;

/// One `HorizonForecast` per horizon from parallel per-horizon lists.
fn forecasts(
    means: Vec<Vec<f64>>,
    covariances: Vec<Vec<Vec<f64>>>,
    costs: Vec<Vec<f64>>,
) -> PyResult<Vec<HorizonForecast>> {
    if covariances.len() != means.len() || costs.len() != means.len() {
        return Err(PyValueError::new_err(format!(
            "means, covariances and costs must have one entry per horizon (got {}, {} and {})",
            means.len(),
            covariances.len(),
            costs.len()
        )));
    }
    means
        .into_iter()
        .zip(covariances)
        .zip(costs)
        .enumerate()
        .map(|(h, ((mean, cov), cost))| {
            let n = cov.len();
            if cov.iter().any(|row| row.len() != n) {
                return Err(PyValueError::new_err(format!(
                    "the covariance matrix of horizon {h} must be square"
                )));
            }
            let flat: Vec<f64> = cov.into_iter().flatten().collect();
            Ok(HorizonForecast { mean, covariance: DMatrix::from_row_slice(n, n, &flat), cost })
        })
        .collect()
}

fn initial_or_zeros(initial_weights: Option<Vec<f64>>, means: &[Vec<f64>]) -> Vec<f64> {
    initial_weights.unwrap_or_else(|| vec![0.0; means.first().map_or(0, Vec::len)])
}

/// Every way to place `k` indivisible units of capital into `n` assets (AFML Snippet 21.1).
///
/// Each partition is a list of `n` non-negative counts summing to `k`, in the book's order
/// (the order `itertools.combinations_with_replacement(range(n), k)` implies):
/// `[k, 0, ..., 0]` first and `[0, ..., 0, k]` last. There are `C(k + n - 1, n - 1)` of
/// them, so large inputs produce very long lists. With `n = 0` there is one (empty)
/// partition of `k = 0` and none of any larger `k`.
///
/// Parameters
/// ----------
/// k : int
///     Number of units of capital.
/// n : int
///     Number of assets (slots).
///
/// Returns
/// -------
/// list[list[int]]
///     One list of `n` counts per partition.
#[pyfunction(name = "pigeonhole_partitions")]
fn dynamic_allocation_pigeonhole_partitions(k: usize, n: usize) -> Vec<Vec<usize>> {
    da::pigeonhole_partitions(k, n)
}

/// The set of signed weight vectors searched by the dynamic allocation (AFML Snippet 21.2).
///
/// Every partition from `pigeonhole_partitions(k, n)` is divided by `k`, so its absolute
/// weights sum to 1 (gross exposure 1), and given every combination of signs. Unlike the
/// book, which repeats vectors whose zero entries are sign-flipped, each vector appears
/// once, at its first occurrence in the book's order.
///
/// Parameters
/// ----------
/// k : int
///     Number of units of capital; weights are multiples of `1 / k`. Must be positive.
/// n : int
///     Number of assets. Must be positive.
///
/// Returns
/// -------
/// list[list[float]]
///     One weight vector of length `n` per element of the set.
///
/// Raises
/// ------
/// ValueError
///     If `k` or `n` is 0.
#[pyfunction(name = "all_weights")]
fn dynamic_allocation_all_weights(k: usize, n: usize) -> PyResult<Vec<Vec<f64>>> {
    da::all_weights(k, n).map_err(to_py_err)
}

/// Transaction cost of each horizon of a weight trajectory (AFML Snippet 21.3, `evalTCosts`).
///
/// `tau_h = sum_n c[h][n] * sqrt(|w[h][n] - w[h-1][n]|)`, where `w[-1]` is
/// `initial_weights`. The square root makes the cost concave in the trade size. The
/// forecasts are validated as in `dynamic_optimal_portfolio`, including the covariances.
///
/// Parameters
/// ----------
/// trajectory : list[list[float]]
///     One weight vector of length `N` per horizon, in horizon order.
/// means : list[list[float]]
///     Expected return of each asset, one list of length `N` per horizon.
/// covariances : list[list[list[float]]]
///     Symmetric positive definite `N x N` covariance matrix per horizon (row lists).
/// costs : list[list[float]]
///     Non-negative cost coefficient of each asset, one list of length `N` per horizon.
/// initial_weights : list[float] | None, default None
///     Portfolio held before the first horizon; None starts from cash (all zeros).
///
/// Returns
/// -------
/// list[float]
///     The cost `tau_h` of each horizon.
///
/// Raises
/// ------
/// ValueError
///     If `means`, `covariances` and `costs` differ in length, a covariance is not square,
///     or the core rejects the input (e.g. no horizons or assets, a length that does not
///     match `N` or the number of horizons, a NaN or infinity, a negative cost, or a
///     covariance that is not symmetric positive definite).
#[pyfunction(name = "transaction_costs")]
#[pyo3(signature = (trajectory, means, covariances, costs, initial_weights=None))]
fn dynamic_allocation_transaction_costs(
    trajectory: Vec<Vec<f64>>,
    means: Vec<Vec<f64>>,
    covariances: Vec<Vec<Vec<f64>>>,
    costs: Vec<Vec<f64>>,
    initial_weights: Option<Vec<f64>>,
) -> PyResult<Vec<f64>> {
    let initial = initial_or_zeros(initial_weights, &means);
    let horizons = forecasts(means, covariances, costs)?;
    da::transaction_costs(&trajectory, &horizons, &initial).map_err(to_py_err)
}

/// Sharpe ratio of a weight trajectory net of transaction costs (AFML Snippet 21.3, `evalSR`).
///
/// `SR = sum_h (mu_h' w_h - tau_h) / sqrt(sum_h w_h' V_h w_h)`, with `tau_h` from
/// `transaction_costs`. Costs are subtracted from the mean only; the variance is that of
/// the gross returns, as in the book.
///
/// Parameters
/// ----------
/// trajectory : list[list[float]]
///     One weight vector of length `N` per horizon, in horizon order.
/// means : list[list[float]]
///     Expected return of each asset, one list of length `N` per horizon.
/// covariances : list[list[list[float]]]
///     Symmetric positive definite `N x N` covariance matrix per horizon (row lists).
/// costs : list[list[float]]
///     Non-negative cost coefficient of each asset, one list of length `N` per horizon.
/// initial_weights : list[float] | None, default None
///     Portfolio held before the first horizon; None starts from cash (all zeros).
///
/// Returns
/// -------
/// float
///     The trajectory's net Sharpe ratio.
///
/// Raises
/// ------
/// ValueError
///     If `means`, `covariances` and `costs` differ in length, a covariance is not square,
///     or the core rejects the input (e.g. no horizons or assets, a length that does not
///     match `N` or the number of horizons, a NaN or infinity, a negative cost, a
///     covariance that is not symmetric positive definite, or a trajectory whose total
///     variance is not positive, such as all zeros).
#[pyfunction(name = "trajectory_sharpe_ratio")]
#[pyo3(signature = (trajectory, means, covariances, costs, initial_weights=None))]
fn dynamic_allocation_trajectory_sharpe_ratio(
    trajectory: Vec<Vec<f64>>,
    means: Vec<Vec<f64>>,
    covariances: Vec<Vec<Vec<f64>>>,
    costs: Vec<Vec<f64>>,
    initial_weights: Option<Vec<f64>>,
) -> PyResult<f64> {
    let initial = initial_or_zeros(initial_weights, &means);
    let horizons = forecasts(means, covariances, costs)?;
    da::trajectory_sharpe_ratio(&trajectory, &horizons, &initial).map_err(to_py_err)
}

/// Best weight trajectory by exhaustive integer search (AFML Snippet 21.3, `dynOptPort`).
///
/// Enumerates every trajectory in `Omega^H`, where `Omega` is `all_weights(k, N)` and `H`
/// the number of horizons, scores each with `trajectory_sharpe_ratio` and keeps the first
/// trajectory with the highest score (the book's order and tie-breaking). The count
/// `|Omega|^H` grows fast (38 vectors for `N = k = 3`, so 54,872 trajectories over three
/// horizons but 23 million over five); a problem above `max_trajectories` is rejected
/// before any work is done. The search releases the GIL.
///
/// Parameters
/// ----------
/// means : list[list[float]]
///     Expected return of each asset, one list of length `N` per horizon.
/// covariances : list[list[list[float]]]
///     Symmetric positive definite `N x N` covariance matrix per horizon (row lists).
/// costs : list[list[float]]
///     Non-negative cost coefficient of each asset, one list of length `N` per horizon;
///     trading `x` of weight costs `c * sqrt(|x|)`.
/// k : int | None, default None
///     Units of capital; weights are multiples of `1 / k`. None uses the book's default
///     `k = N`.
/// initial_weights : list[float] | None, default None
///     Portfolio held before the first horizon; None starts from cash (all zeros).
/// max_trajectories : int, default 1000000
///     Largest number of trajectories the search may evaluate.
///
/// Returns
/// -------
/// dict[str, Any]
///     `weights` (list[list[float]], one vector of length `N` per horizon; the book returns
///     the transpose), `sharpe_ratio` (float, net of costs), `transaction_costs`
///     (list[float], one per horizon) and `trajectories_evaluated` (int, `|Omega|^H`).
///
/// Raises
/// ------
/// ValueError
///     If `means`, `covariances` and `costs` differ in length, a covariance is not square,
///     or the core rejects the input (e.g. no horizons or assets, `k` of 0, a length that
///     does not match `N`, a NaN or infinity, a negative cost, a covariance that is not
///     symmetric positive definite, or more than `max_trajectories` trajectories).
#[pyfunction(name = "dynamic_optimal_portfolio")]
#[pyo3(signature = (
    means,
    covariances,
    costs,
    k=None,
    initial_weights=None,
    max_trajectories=da::DEFAULT_MAX_TRAJECTORIES,
))]
fn dynamic_allocation_dynamic_optimal_portfolio(
    py: Python<'_>,
    means: Vec<Vec<f64>>,
    covariances: Vec<Vec<Vec<f64>>>,
    costs: Vec<Vec<f64>>,
    k: Option<usize>,
    initial_weights: Option<Vec<f64>>,
    max_trajectories: usize,
) -> PyResult<PyObject> {
    // The book's default: as many units of capital as assets.
    let units = k.unwrap_or_else(|| means.first().map_or(0, Vec::len));
    let horizons = forecasts(means, covariances, costs)?;
    let config = DynamicAllocationConfig { units, initial_weights, max_trajectories };
    let result = py
        .allow_threads(|| da::dynamic_optimal_portfolio(&horizons, &config))
        .map_err(to_py_err)?;

    let d = PyDict::new(py);
    d.set_item("weights", result.weights)?;
    d.set_item("sharpe_ratio", result.sharpe_ratio)?;
    d.set_item("transaction_costs", result.transaction_costs)?;
    d.set_item("trajectories_evaluated", result.trajectories_evaluated)?;
    Ok(d.into_pyobject(py).unwrap().into_any().unbind())
}

pub fn register(py: Python<'_>, parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(py, "dynamic_allocation")?;
    m.add_function(wrap_pyfunction!(dynamic_allocation_pigeonhole_partitions, &m)?)?;
    m.add_function(wrap_pyfunction!(dynamic_allocation_all_weights, &m)?)?;
    m.add_function(wrap_pyfunction!(dynamic_allocation_transaction_costs, &m)?)?;
    m.add_function(wrap_pyfunction!(dynamic_allocation_trajectory_sharpe_ratio, &m)?)?;
    m.add_function(wrap_pyfunction!(dynamic_allocation_dynamic_optimal_portfolio, &m)?)?;
    parent.add_submodule(&m)?;
    parent.add("dynamic_allocation", m)?;
    Ok(())
}

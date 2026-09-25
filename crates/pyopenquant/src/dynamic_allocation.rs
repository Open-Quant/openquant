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

#[pyfunction(name = "pigeonhole_partitions")]
fn dynamic_allocation_pigeonhole_partitions(k: usize, n: usize) -> Vec<Vec<usize>> {
    da::pigeonhole_partitions(k, n)
}

#[pyfunction(name = "all_weights")]
fn dynamic_allocation_all_weights(k: usize, n: usize) -> PyResult<Vec<Vec<f64>>> {
    da::all_weights(k, n).map_err(to_py_err)
}

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

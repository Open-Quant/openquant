use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::helpers::{matrix_from_rows, to_py_err};

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
    let expected_ret_m = expected_returns.map(|v| {
        nalgebra::DMatrix::from_vec(v.len(), 1, v)
    });

    cla.allocate(
        prices_m.as_ref().map(|m| openquant::cla::AssetPricesInput::RawMatrix(m)),
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

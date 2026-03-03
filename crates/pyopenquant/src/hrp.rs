use pyo3::prelude::*;

use crate::helpers::{matrix_from_rows, to_py_err};

#[pyfunction(name = "allocate_hrp")]
#[pyo3(signature = (
    asset_names,
    asset_prices=None,
    asset_returns=None,
    covariance_matrix=None,
    resample_by=None,
    use_shrinkage=false
))]
fn hrp_allocate(
    asset_names: Vec<String>,
    asset_prices: Option<Vec<Vec<f64>>>,
    asset_returns: Option<Vec<Vec<f64>>>,
    covariance_matrix: Option<Vec<Vec<f64>>>,
    resample_by: Option<String>,
    use_shrinkage: bool,
) -> PyResult<(Vec<f64>, Vec<usize>)> {
    let prices_m = asset_prices.map(matrix_from_rows).transpose()?;
    let returns_m = asset_returns.map(matrix_from_rows).transpose()?;
    let cov_m = covariance_matrix.map(matrix_from_rows).transpose()?;

    let mut hrp = openquant::hrp::HierarchicalRiskParity::new();
    hrp.allocate(
        &asset_names,
        prices_m.as_ref(),
        returns_m.as_ref(),
        cov_m.as_ref(),
        resample_by.as_deref(),
        use_shrinkage,
    )
    .map_err(to_py_err)?;

    Ok((hrp.weights, hrp.ordered_indices))
}

pub fn register(py: Python<'_>, parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(py, "hrp")?;
    m.add_function(wrap_pyfunction!(hrp_allocate, &m)?)?;
    parent.add_submodule(&m)?;
    parent.add("hrp", m)?;
    Ok(())
}

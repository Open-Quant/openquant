use pyo3::prelude::*;

use crate::helpers::{matrix_from_rows, to_py_err};

#[pyfunction(name = "allocate_hcaa")]
#[pyo3(signature = (
    asset_names,
    asset_prices=None,
    asset_returns=None,
    covariance_matrix=None,
    expected_asset_returns=None,
    allocation_metric="equal_weighting",
    confidence_level=0.05,
    optimal_num_clusters=None,
    resample_by=None,
    calculate_expected_returns="mean"
))]
fn hcaa_allocate(
    asset_names: Vec<String>,
    asset_prices: Option<Vec<Vec<f64>>>,
    asset_returns: Option<Vec<Vec<f64>>>,
    covariance_matrix: Option<Vec<Vec<f64>>>,
    expected_asset_returns: Option<Vec<f64>>,
    allocation_metric: &str,
    confidence_level: f64,
    optimal_num_clusters: Option<usize>,
    resample_by: Option<String>,
    calculate_expected_returns: &str,
) -> PyResult<(Vec<f64>, Vec<usize>)> {
    let prices_m = asset_prices.map(matrix_from_rows).transpose()?;
    let returns_m = asset_returns.map(matrix_from_rows).transpose()?;
    let cov_m = covariance_matrix.map(matrix_from_rows).transpose()?;

    let mut hcaa = openquant::hcaa::HierarchicalClusteringAssetAllocation::new(calculate_expected_returns);
    hcaa.allocate(
        &asset_names,
        prices_m.as_ref(),
        returns_m.as_ref(),
        cov_m.as_ref(),
        expected_asset_returns.as_deref(),
        allocation_metric,
        confidence_level,
        optimal_num_clusters,
        resample_by.as_deref(),
    )
    .map_err(to_py_err)?;

    Ok((hcaa.weights, hcaa.ordered_indices))
}

pub fn register(py: Python<'_>, parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(py, "hcaa")?;
    m.add_function(wrap_pyfunction!(hcaa_allocate, &m)?)?;
    parent.add_submodule(&m)?;
    parent.add("hcaa", m)?;
    Ok(())
}

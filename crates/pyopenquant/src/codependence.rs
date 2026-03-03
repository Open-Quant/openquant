use pyo3::prelude::*;

use crate::helpers::to_py_err;

#[pyfunction(name = "angular_distance")]
fn codependence_angular_distance(x: Vec<f64>, y: Vec<f64>) -> PyResult<f64> {
    openquant::codependence::angular_distance(&x, &y).map_err(to_py_err)
}

#[pyfunction(name = "absolute_angular_distance")]
fn codependence_absolute_angular_distance(x: Vec<f64>, y: Vec<f64>) -> PyResult<f64> {
    openquant::codependence::absolute_angular_distance(&x, &y).map_err(to_py_err)
}

#[pyfunction(name = "squared_angular_distance")]
fn codependence_squared_angular_distance(x: Vec<f64>, y: Vec<f64>) -> PyResult<f64> {
    openquant::codependence::squared_angular_distance(&x, &y).map_err(to_py_err)
}

#[pyfunction(name = "distance_correlation")]
fn codependence_distance_correlation(x: Vec<f64>, y: Vec<f64>) -> PyResult<f64> {
    openquant::codependence::distance_correlation(&x, &y).map_err(to_py_err)
}

#[pyfunction(name = "get_optimal_number_of_bins")]
#[pyo3(signature = (num_obs, corr_coef=None))]
fn codependence_get_optimal_number_of_bins(
    num_obs: usize,
    corr_coef: Option<f64>,
) -> PyResult<usize> {
    openquant::codependence::get_optimal_number_of_bins(num_obs, corr_coef).map_err(to_py_err)
}

#[pyfunction(name = "get_mutual_info")]
#[pyo3(signature = (x, y, n_bins=None, normalize=false))]
fn codependence_get_mutual_info(
    x: Vec<f64>,
    y: Vec<f64>,
    n_bins: Option<usize>,
    normalize: bool,
) -> PyResult<f64> {
    openquant::codependence::get_mutual_info(&x, &y, n_bins, normalize).map_err(to_py_err)
}

#[pyfunction(name = "variation_of_information_score")]
#[pyo3(signature = (x, y, n_bins=None, normalize=false))]
fn codependence_variation_of_information_score(
    x: Vec<f64>,
    y: Vec<f64>,
    n_bins: Option<usize>,
    normalize: bool,
) -> PyResult<f64> {
    openquant::codependence::variation_of_information_score(&x, &y, n_bins, normalize)
        .map_err(to_py_err)
}

pub fn register(py: Python<'_>, parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(py, "codependence")?;
    m.add_function(wrap_pyfunction!(codependence_angular_distance, &m)?)?;
    m.add_function(wrap_pyfunction!(codependence_absolute_angular_distance, &m)?)?;
    m.add_function(wrap_pyfunction!(codependence_squared_angular_distance, &m)?)?;
    m.add_function(wrap_pyfunction!(codependence_distance_correlation, &m)?)?;
    m.add_function(wrap_pyfunction!(codependence_get_optimal_number_of_bins, &m)?)?;
    m.add_function(wrap_pyfunction!(codependence_get_mutual_info, &m)?)?;
    m.add_function(wrap_pyfunction!(codependence_variation_of_information_score, &m)?)?;
    parent.add_submodule(&m)?;
    parent.add("codependence", m)?;
    Ok(())
}

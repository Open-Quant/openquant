use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::helpers::{matrix_from_rows, to_py_err};

#[pyfunction(name = "get_onc_clusters")]
fn onc_get_onc_clusters(
    py: Python<'_>,
    corr_mat: Vec<Vec<f64>>,
    repeat: usize,
) -> PyResult<PyObject> {
    let m = matrix_from_rows(corr_mat)?;
    let result = openquant::onc::get_onc_clusters(&m, repeat).map_err(to_py_err)?;

    let d = PyDict::new(py);

    // Convert ordered correlation matrix to Vec<Vec<f64>>
    let nrows = result.ordered_correlation.nrows();
    let ncols = result.ordered_correlation.ncols();
    let ordered: Vec<Vec<f64>> = (0..nrows)
        .map(|r| (0..ncols).map(|c| result.ordered_correlation[(r, c)]).collect())
        .collect();
    d.set_item("ordered_correlation", ordered)?;

    // Convert BTreeMap<usize, Vec<usize>> to PyDict
    let clusters = PyDict::new(py);
    for (k, v) in &result.clusters {
        clusters.set_item(*k, v.clone())?;
    }
    d.set_item("clusters", clusters)?;

    d.set_item("silhouette_scores", result.silhouette_scores)?;

    Ok(d.into_pyobject(py).unwrap().into_any().unbind())
}

pub fn register(py: Python<'_>, parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(py, "onc")?;
    m.add_function(wrap_pyfunction!(onc_get_onc_clusters, &m)?)?;
    parent.add_submodule(&m)?;
    parent.add("onc", m)?;
    Ok(())
}

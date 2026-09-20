use pyo3::prelude::*;

use crate::helpers::to_py_err;

#[pyfunction(name = "ewma")]
fn fast_ewma_ewma(arr: Vec<f64>, window: usize) -> PyResult<Vec<f64>> {
    openquant::util::fast_ewma::ewma(&arr, window).map_err(to_py_err)
}

pub fn register(py: Python<'_>, parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(py, "fast_ewma")?;
    m.add_function(wrap_pyfunction!(fast_ewma_ewma, &m)?)?;
    parent.add_submodule(&m)?;
    parent.add("fast_ewma", m)?;
    Ok(())
}

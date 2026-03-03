use pyo3::prelude::*;

#[pyfunction(name = "ewma")]
fn fast_ewma_ewma(arr: Vec<f64>, window: usize) -> Vec<f64> {
    openquant::util::fast_ewma::ewma(&arr, window)
}

pub fn register(py: Python<'_>, parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(py, "fast_ewma")?;
    m.add_function(wrap_pyfunction!(fast_ewma_ewma, &m)?)?;
    parent.add_submodule(&m)?;
    parent.add("fast_ewma", m)?;
    Ok(())
}

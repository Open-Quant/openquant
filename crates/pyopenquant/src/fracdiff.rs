use pyo3::prelude::*;

#[pyfunction(name = "get_weights")]
fn fracdiff_get_weights(diff_amt: f64, size: usize) -> Vec<f64> {
    openquant::fracdiff::get_weights(diff_amt, size)
}

#[pyfunction(name = "get_weights_ffd")]
fn fracdiff_get_weights_ffd(diff_amt: f64, thresh: f64, lim: usize) -> Vec<f64> {
    openquant::fracdiff::get_weights_ffd(diff_amt, thresh, lim)
}

#[pyfunction(name = "frac_diff")]
fn fracdiff_frac_diff(series: Vec<f64>, diff_amt: f64, thresh: f64) -> Vec<f64> {
    openquant::fracdiff::frac_diff(&series, diff_amt, thresh)
}

#[pyfunction(name = "frac_diff_ffd")]
fn fracdiff_frac_diff_ffd(series: Vec<f64>, diff_amt: f64, thresh: f64) -> Vec<f64> {
    openquant::fracdiff::frac_diff_ffd(&series, diff_amt, thresh)
}

pub fn register(py: Python<'_>, parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(py, "fracdiff")?;
    m.add_function(wrap_pyfunction!(fracdiff_get_weights, &m)?)?;
    m.add_function(wrap_pyfunction!(fracdiff_get_weights_ffd, &m)?)?;
    m.add_function(wrap_pyfunction!(fracdiff_frac_diff, &m)?)?;
    m.add_function(wrap_pyfunction!(fracdiff_frac_diff_ffd, &m)?)?;
    parent.add_submodule(&m)?;
    parent.add("fracdiff", m)?;
    Ok(())
}

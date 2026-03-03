use pyo3::prelude::*;

use crate::helpers::to_py_err;

#[pyfunction(name = "get_chow_type_stat")]
fn sb_get_chow_type_stat(log_prices: Vec<f64>, min_length: usize) -> PyResult<Vec<f64>> {
    openquant::structural_breaks::get_chow_type_stat(&log_prices, min_length).map_err(to_py_err)
}

#[pyfunction(name = "get_chu_stinchcombe_white_statistics")]
fn sb_get_chu_stinchcombe_white_statistics(
    log_prices: Vec<f64>,
    test_type: String,
) -> PyResult<(Vec<f64>, Vec<f64>)> {
    let result =
        openquant::structural_breaks::get_chu_stinchcombe_white_statistics(&log_prices, &test_type)
            .map_err(to_py_err)?;
    Ok((result.critical_value, result.stat))
}

#[pyfunction(name = "get_sadf")]
#[pyo3(signature = (series, model, add_const, min_length, lags))]
fn sb_get_sadf(
    series: Vec<f64>,
    model: String,
    add_const: bool,
    min_length: usize,
    lags: usize,
) -> PyResult<Vec<f64>> {
    openquant::structural_breaks::get_sadf(
        &series,
        &model,
        add_const,
        min_length,
        openquant::structural_breaks::SadfLags::Fixed(lags),
    )
    .map_err(to_py_err)
}

pub fn register(py: Python<'_>, parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(py, "structural_breaks")?;
    m.add_function(wrap_pyfunction!(sb_get_chow_type_stat, &m)?)?;
    m.add_function(wrap_pyfunction!(sb_get_chu_stinchcombe_white_statistics, &m)?)?;
    m.add_function(wrap_pyfunction!(sb_get_sadf, &m)?)?;
    parent.add_submodule(&m)?;
    parent.add("structural_breaks", m)?;
    Ok(())
}

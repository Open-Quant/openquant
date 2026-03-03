use pyo3::prelude::*;
use pyo3::types::PyDict;


#[pyfunction(name = "centered_moment")]
fn ef3m_centered_moment(moments: Vec<f64>, order: usize) -> f64 {
    openquant::ef3m::centered_moment(&moments, order)
}

#[pyfunction(name = "raw_moment")]
fn ef3m_raw_moment(central_moments: Vec<f64>, dist_mean: f64) -> Vec<f64> {
    openquant::ef3m::raw_moment(&central_moments, dist_mean)
}

#[pyfunction(name = "most_likely_parameters")]
fn ef3m_most_likely_parameters(
    py: Python<'_>,
    data: Vec<(f64, f64, f64, f64, f64, f64)>,
    res: usize,
) -> PyResult<PyObject> {
    let rows: Vec<openquant::ef3m::FitResultRow> = data
        .into_iter()
        .map(|(mu_1, mu_2, sigma_1, sigma_2, p_1, error)| openquant::ef3m::FitResultRow {
            mu_1, mu_2, sigma_1, sigma_2, p_1, error,
        })
        .collect();
    let result = openquant::ef3m::most_likely_parameters(&rows, None, res);
    let d = PyDict::new(py);
    for (k, v) in result {
        d.set_item(k, v)?;
    }
    Ok(d.into_pyobject(py).unwrap().into_any().unbind())
}

#[pyfunction(name = "fit_m2n")]
#[pyo3(signature = (
    moments,
    epsilon=1e-5,
    factor=5.0,
    n_runs=1,
    variant=4,
    max_iter=100_000
))]
fn ef3m_fit_m2n(
    moments: Vec<f64>,
    epsilon: f64,
    factor: f64,
    n_runs: usize,
    variant: usize,
    max_iter: usize,
) -> PyResult<Vec<(f64, f64, f64, f64, f64, f64)>> {
    let mut m2n = openquant::ef3m::M2N::new(moments, epsilon, factor, n_runs, variant, max_iter, 1);
    let results = m2n.single_fit_loop(None);
    Ok(results
        .into_iter()
        .map(|r| (r.mu_1, r.mu_2, r.sigma_1, r.sigma_2, r.p_1, r.error))
        .collect())
}

pub fn register(py: Python<'_>, parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(py, "ef3m")?;
    m.add_function(wrap_pyfunction!(ef3m_centered_moment, &m)?)?;
    m.add_function(wrap_pyfunction!(ef3m_raw_moment, &m)?)?;
    m.add_function(wrap_pyfunction!(ef3m_most_likely_parameters, &m)?)?;
    m.add_function(wrap_pyfunction!(ef3m_fit_m2n, &m)?)?;
    parent.add_submodule(&m)?;
    parent.add("ef3m", m)?;
    Ok(())
}

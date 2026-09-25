use pyo3::prelude::*;

use crate::helpers::to_py_err;

#[pyfunction(name = "get_ind_matrix")]
fn sampling_get_ind_matrix(
    label_endtime: Vec<(usize, usize)>,
    bar_index: Vec<usize>,
) -> PyResult<Vec<Vec<u32>>> {
    // Widened so each row reaches Python as a list of ints; PyO3 turns a Vec<u8> into `bytes`.
    let ind_mat =
        openquant::sampling::get_ind_matrix(&label_endtime, &bar_index).map_err(to_py_err)?;
    Ok(ind_mat.into_iter().map(|row| row.into_iter().map(u32::from).collect()).collect())
}

#[pyfunction(name = "get_ind_mat_average_uniqueness")]
fn sampling_get_ind_mat_average_uniqueness(ind_mat: Vec<Vec<u8>>) -> PyResult<f64> {
    openquant::sampling::get_ind_mat_average_uniqueness(&ind_mat).map_err(to_py_err)
}

#[pyfunction(name = "get_ind_mat_label_uniqueness")]
fn sampling_get_ind_mat_label_uniqueness(ind_mat: Vec<Vec<u8>>) -> PyResult<Vec<Vec<f64>>> {
    openquant::sampling::get_ind_mat_label_uniqueness(&ind_mat).map_err(to_py_err)
}

#[pyfunction(name = "bootstrap_loop_run")]
fn sampling_bootstrap_loop_run(
    ind_mat: Vec<Vec<u8>>,
    prev_concurrency: Vec<f64>,
) -> PyResult<Vec<f64>> {
    openquant::sampling::bootstrap_loop_run(&ind_mat, &prev_concurrency).map_err(to_py_err)
}

#[pyfunction(name = "seq_bootstrap")]
#[pyo3(signature = (ind_mat, sample_length=None, warmup_samples=None, random_state=None))]
fn sampling_seq_bootstrap(
    ind_mat: Vec<Vec<u8>>,
    sample_length: Option<usize>,
    warmup_samples: Option<Vec<usize>>,
    random_state: Option<u64>,
) -> PyResult<Vec<usize>> {
    match random_state {
        Some(seed) => {
            let mut rng = <rand::rngs::StdRng as rand::SeedableRng>::seed_from_u64(seed);
            openquant::sampling::seq_bootstrap_with_rng(
                &ind_mat,
                sample_length,
                warmup_samples,
                &mut rng,
            )
        }
        None => openquant::sampling::seq_bootstrap(&ind_mat, sample_length, warmup_samples),
    }
    .map_err(to_py_err)
}

#[pyfunction(name = "get_av_uniqueness_from_triple_barrier")]
fn sampling_get_av_uniqueness_from_triple_barrier(
    samples_info: Vec<(usize, usize)>,
    price_bars_len: usize,
) -> PyResult<Vec<f64>> {
    openquant::sampling::get_av_uniqueness_from_triple_barrier(&samples_info, price_bars_len)
        .map_err(to_py_err)
}

#[pyfunction(name = "num_concurrent_events")]
fn sampling_num_concurrent_events(
    price_index_len: usize,
    t1: Vec<(usize, usize)>,
    t_events: Vec<usize>,
) -> Vec<usize> {
    openquant::sampling::num_concurrent_events(price_index_len, &t1, &t_events)
}

pub fn register(py: Python<'_>, parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(py, "sampling")?;
    m.add_function(wrap_pyfunction!(sampling_get_ind_matrix, &m)?)?;
    m.add_function(wrap_pyfunction!(sampling_get_ind_mat_average_uniqueness, &m)?)?;
    m.add_function(wrap_pyfunction!(sampling_get_ind_mat_label_uniqueness, &m)?)?;
    m.add_function(wrap_pyfunction!(sampling_bootstrap_loop_run, &m)?)?;
    m.add_function(wrap_pyfunction!(sampling_seq_bootstrap, &m)?)?;
    m.add_function(wrap_pyfunction!(sampling_get_av_uniqueness_from_triple_barrier, &m)?)?;
    m.add_function(wrap_pyfunction!(sampling_num_concurrent_events, &m)?)?;
    parent.add_submodule(&m)?;
    parent.add("sampling", m)?;
    Ok(())
}

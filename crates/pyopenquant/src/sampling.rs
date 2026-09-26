use pyo3::prelude::*;

use crate::helpers::to_py_err;

/// Indicator matrix of which labels span which bars (AFML Snippet 4.3).
///
/// Works in bar positions, not timestamps: `label_endtime[i] = (start, end)` is label `i`'s
/// inclusive span. Entry `[r][i]` is 1 when `start <= bar_index[r] <= end` and 0
/// otherwise. A span reaching past the bars listed is truncated silently, and `bar_index`
/// need not be contiguous. The matrix is dense (bars x labels).
///
/// Parameters
/// ----------
/// label_endtime : list[tuple[int, int]]
///     Inclusive `(start, end)` bar positions of each label.
/// bar_index : list[int]
///     Bar positions to report, in the order the rows should appear.
///
/// Returns
/// -------
/// list[list[int]]
///     A 0/1 matrix with one row per entry of `bar_index` and one column per label.
///
/// Raises
/// ------
/// ValueError
///     If any label has `start > end`.
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

/// Mean over labels of each label's average uniqueness (AFML Snippet 4.4, whole sample).
///
/// For each label `i` the average uniqueness is `sum_t u[t][i] / sum_t ind_mat[t][i]`,
/// with `u[t][i] = ind_mat[t][i] / c_t` and `c_t` the row sum at bar `t`. The result is the
/// plain mean over labels that span at least one bar; all-zero columns are left out. An
/// empty matrix, or one where no label spans any bar, gives 0.0. Pass a 0/1 matrix.
/// Repeated columns (a bootstrap sample) are allowed and lower each other's uniqueness.
///
/// Parameters
/// ----------
/// ind_mat : list[list[int]]
///     Indicator matrix, one row per bar and one 0/1 column per label (see
///     `get_ind_matrix`).
///
/// Returns
/// -------
/// float
///     The mean average uniqueness, in `[0, 1]`.
///
/// Raises
/// ------
/// ValueError
///     If the rows of `ind_mat` differ in length.
#[pyfunction(name = "get_ind_mat_average_uniqueness")]
fn sampling_get_ind_mat_average_uniqueness(ind_mat: Vec<Vec<u8>>) -> PyResult<f64> {
    openquant::sampling::get_ind_mat_average_uniqueness(&ind_mat).map_err(to_py_err)
}

/// Per-bar uniqueness `u[t][i] = ind_mat[t][i] / c_t` of every label (AFML section 4.4).
///
/// `c_t` is the row sum of `ind_mat` at bar `t`, the number of labels alive there. The
/// result is indexed label first. An empty matrix gives an empty result.
///
/// Parameters
/// ----------
/// ind_mat : list[list[int]]
///     Indicator matrix, one row per bar and one 0/1 column per label (see
///     `get_ind_matrix`).
///
/// Returns
/// -------
/// list[list[float]]
///     `result[i][t]` is label `i`'s uniqueness at bar (row) `t`, and 0 at bars the label
///     does not span.
///
/// Raises
/// ------
/// ValueError
///     If the rows of `ind_mat` differ in length.
#[pyfunction(name = "get_ind_mat_label_uniqueness")]
fn sampling_get_ind_mat_label_uniqueness(ind_mat: Vec<Vec<u8>>) -> PyResult<Vec<Vec<f64>>> {
    openquant::sampling::get_ind_mat_label_uniqueness(&ind_mat).map_err(to_py_err)
}

/// One step of the sequential bootstrap (the inner loop of AFML Snippet 4.5).
///
/// Returns the average uniqueness each label would have if it were drawn next: for label
/// `j`, the mean over the bars `t` it spans of `1 / (1 + prev_concurrency[t])`. A label
/// that spans no bar gets 0. Normalising the result to sum to one gives the draw
/// probabilities.
///
/// Parameters
/// ----------
/// ind_mat : list[list[int]]
///     Indicator matrix, one row per bar and one 0/1 column per label (see
///     `get_ind_matrix`).
/// prev_concurrency : list[float]
///     How many times each bar is already covered by the labels drawn so far, one entry
///     per row of `ind_mat`.
///
/// Returns
/// -------
/// list[float]
///     One average uniqueness per label.
///
/// Raises
/// ------
/// ValueError
///     If the rows of `ind_mat` differ in length, or `prev_concurrency` does not have one
///     entry per row.
#[pyfunction(name = "bootstrap_loop_run")]
fn sampling_bootstrap_loop_run(
    ind_mat: Vec<Vec<u8>>,
    prev_concurrency: Vec<f64>,
) -> PyResult<Vec<f64>> {
    openquant::sampling::bootstrap_loop_run(&ind_mat, &prev_concurrency).map_err(to_py_err)
}

/// Sequential bootstrap of label indices (AFML Snippet 4.5).
///
/// Labels are drawn one at a time, each with probability proportional to the average
/// uniqueness it would have given the draws so far (see `bootstrap_loop_run`). Repeats are
/// possible, only less likely than under a uniform bootstrap. `warmup_samples` forces the
/// first draws and is consumed from the end of the list (so `[2, 0]` draws label 0
/// first). If no label spans any bar, the later draws are uniform. The work is of order
/// bars x labels^2, so bootstrap within blocks for large label sets.
///
/// Parameters
/// ----------
/// ind_mat : list[list[int]]
///     Indicator matrix, one row per bar and one 0/1 column per label (see
///     `get_ind_matrix`).
/// sample_length : int | None, default None
///     Number of draws; None draws as many as there are labels. 0 returns an empty list.
/// warmup_samples : list[int] | None, default None
///     Label indices forced as the first draws, taken from the end of the list.
/// random_state : int | None, default None
///     Seed for a reproducible sample; None draws from the thread-local generator, so
///     repeated calls differ.
///
/// Returns
/// -------
/// list[int]
///     Drawn label (column) indices, in draw order.
///
/// Raises
/// ------
/// ValueError
///     If the rows of `ind_mat` differ in length, `ind_mat` has no labels while a non-zero
///     `sample_length` is requested, or a `warmup_samples` index is not a valid label.
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

/// Average uniqueness of each label over its lifespan, from label spans (AFML Snippet 4.4).
///
/// `samples_info[i] = (start, end)` is label `i`'s inclusive span in bar positions, over
/// bars `0..price_bars_len`. Each label's value is `sum_t u[t][i] / sum_t 1[t][i]` with
/// `u[t][i] = 1 / c_t` on the bars it spans and `c_t` the number of labels alive at bar
/// `t`. Spans running past the last bar are truncated.
///
/// Parameters
/// ----------
/// samples_info : list[tuple[int, int]]
///     Inclusive `(start, end)` bar positions of each label.
/// price_bars_len : int
///     Number of bars.
///
/// Returns
/// -------
/// list[float]
///     One value per label: 1.0 for a label that overlaps no other, 0.0 for one that
///     spans no bar in range.
///
/// Raises
/// ------
/// ValueError
///     If any label has `start > end`.
#[pyfunction(name = "get_av_uniqueness_from_triple_barrier")]
fn sampling_get_av_uniqueness_from_triple_barrier(
    samples_info: Vec<(usize, usize)>,
    price_bars_len: usize,
) -> PyResult<Vec<f64>> {
    openquant::sampling::get_av_uniqueness_from_triple_barrier(&samples_info, price_bars_len)
        .map_err(to_py_err)
}

/// Number of labels alive at each bar, `c_t` (AFML Snippet 4.1).
///
/// `t1[i] = (start, end)` is label `i`'s inclusive span in bar positions. Spans are
/// truncated at the last bar, and a span with `start > end` is skipped silently (where
/// `get_ind_matrix` rejects it).
///
/// Parameters
/// ----------
/// price_index_len : int
///     Number of bars.
/// t1 : list[tuple[int, int]]
///     Inclusive `(start, end)` bar positions of each label.
/// t_events : list[int]
///     Ignored; kept for signature compatibility with the snippet. Pass an empty list.
///
/// Returns
/// -------
/// list[int]
///     The concurrency count of each of the `price_index_len` bars.
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

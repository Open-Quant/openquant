//! `openquant._core.cross_validation`: purged k-fold and CPCV splits as index lists.
//!
//! Nothing here takes a Python callable. The splitters return indices, so the caller fits
//! whatever model it likes (scikit-learn or anything else) on them. The pure-Python
//! `openquant.cross_validation` turns the lists into numpy arrays and converts timestamps.

use chrono::NaiveDateTime;
use openquant::cross_validation::{
    count_train_test_overlaps, naive_kfold_splits, PurgedKFold, PurgedSplit,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::helpers::to_py_err;

/// One `(t0, t1)` label span per sample, from two equal-length columns of int64 nanoseconds.
///
/// The Python wrapper converts datetimes to nanoseconds since the epoch; plain integers (bar
/// positions, say) pass through unchanged. Only the order of the values matters to purging.
pub(crate) fn label_spans(
    t0: Vec<i64>,
    t1: Vec<i64>,
) -> PyResult<Vec<(NaiveDateTime, NaiveDateTime)>> {
    if t0.len() != t1.len() {
        return Err(PyValueError::new_err(format!(
            "t0/t1 length mismatch: {} vs {}",
            t0.len(),
            t1.len()
        )));
    }
    Ok(t0
        .into_iter()
        .zip(t1)
        .map(|(start, end)| {
            (
                chrono::DateTime::from_timestamp_nanos(start).naive_utc(),
                chrono::DateTime::from_timestamp_nanos(end).naive_utc(),
            )
        })
        .collect())
}

/// A `PurgedKFold` over the spans, with its errors raised as `ValueError`.
pub(crate) fn purged_kfold(
    t0: Vec<i64>,
    t1: Vec<i64>,
    n_splits: usize,
    pct_embargo: f64,
) -> PyResult<(PurgedKFold, usize)> {
    let spans = label_spans(t0, t1)?;
    let n = spans.len();
    Ok((PurgedKFold::new(n_splits, spans, pct_embargo).map_err(to_py_err)?, n))
}

fn split_to_dict<'py>(py: Python<'py>, split: PurgedSplit) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    let diag = split.diagnostics;
    d.set_item("split_id", diag.split_id)?;
    d.set_item("train_indices", split.train_indices)?;
    d.set_item("test_indices", split.test_indices)?;
    d.set_item("test_ranges", diag.test_ranges)?;
    d.set_item("purged_indices", diag.purged_indices)?;
    d.set_item("embargo_indices", diag.embargo_indices)?;
    d.set_item("overlap_count_after_purge", diag.overlap_count_after_purge)?;
    Ok(d)
}

/// Purged k-fold splits with an embargo (AFML Snippet 7.3), as index lists.
///
/// Folds are blocks of consecutive samples, never shuffled; the first `n % n_splits` folds hold
/// one extra sample. Purging (AFML Snippet 7.1) drops every non-test sample whose span
/// intersects, as closed intervals, the window from a test block's first start to its latest
/// end. The embargo (Snippet 7.3) then drops `ceil(pct_embargo * n)` more samples after each
/// test block, counting from where the purge ends; nothing before a block is embargoed. The
/// purge compares timestamps but the embargo counts positions, so the samples must be sorted by
/// start (this is not checked).
///
/// Purging can empty a training set when labels are long relative to the folds; check the
/// length of each training list.
///
/// Parameters
/// ----------
/// t0 : list[int]
///     Start of each sample's label span, as int64 nanoseconds since the epoch (plain integers
///     such as bar positions also work; only the order of the values matters). One per sample,
///     in time order.
/// t1 : list[int]
///     End of each sample's label span, in the same units as `t0`. Must be `>= t0`.
/// n_splits : int
///     Number of contiguous test folds; `2 <= n_splits <= len(t0)`.
/// pct_embargo : float
///     Embargo as a fraction of the whole sample count, in `[0, 1)`; rounded up to a number of
///     samples, so any positive value embargoes at least one. AFML suggests about 0.01.
///
/// Returns
/// -------
/// list[tuple[list[int], list[int]]]
///     One `(train_indices, test_indices)` pair per fold, in fold order; both lists sorted.
///
/// Raises
/// ------
/// ValueError
///     If `t0` and `t1` differ in length, or if the core rejects the input (e.g. no samples,
///     `n_splits` below 2 or above the sample count, `pct_embargo` not a finite number in
///     `[0, 1)`, or a span that ends before it starts).
#[pyfunction(name = "purged_kfold_splits")]
fn cv_purged_kfold_splits(
    t0: Vec<i64>,
    t1: Vec<i64>,
    n_splits: usize,
    pct_embargo: f64,
) -> PyResult<Vec<(Vec<usize>, Vec<usize>)>> {
    let (cv, n) = purged_kfold(t0, t1, n_splits, pct_embargo)?;
    cv.split(n).map_err(to_py_err)
}

/// Purged k-fold splits together with the indices purging and the embargo removed.
///
/// The folds are those of `purged_kfold_splits` (AFML Snippet 7.3); each also reports why
/// every non-training, non-test sample was left out.
///
/// Parameters
/// ----------
/// t0 : list[int]
///     Start of each sample's label span, as int64 nanoseconds since the epoch (plain integers
///     such as bar positions also work; only the order of the values matters). One per sample,
///     in time order.
/// t1 : list[int]
///     End of each sample's label span, in the same units as `t0`. Must be `>= t0`.
/// n_splits : int
///     Number of contiguous test folds; `2 <= n_splits <= len(t0)`.
/// pct_embargo : float
///     Embargo as a fraction of the whole sample count, in `[0, 1)`; rounded up to a number of
///     samples, so any positive value embargoes at least one. AFML suggests about 0.01.
///
/// Returns
/// -------
/// list[dict[str, Any]]
///     One dict per fold, in fold order, with keys:
///
///     - `split_id` (int): position of the split in the list.
///     - `train_indices` (list[int]): sorted training indices after purging and embargo.
///     - `test_indices` (list[int]): sorted test indices.
///     - `test_ranges` (list[tuple[int, int]]): the test set as half-open `[start, stop)` blocks.
///     - `purged_indices` (list[int]): non-test samples removed by purging.
///     - `embargo_indices` (list[int]): non-test samples inside an embargo window (a sample can
///       be both purged and embargoed).
///     - `overlap_count_after_purge` (int): training samples whose span still overlaps a test
///       span; always 0, reported so callers can assert it.
///
/// Raises
/// ------
/// ValueError
///     If `t0` and `t1` differ in length, or if the core rejects the input (e.g. no samples,
///     `n_splits` below 2 or above the sample count, `pct_embargo` not a finite number in
///     `[0, 1)`, or a span that ends before it starts).
#[pyfunction(name = "split_with_diagnostics")]
fn cv_split_with_diagnostics<'py>(
    py: Python<'py>,
    t0: Vec<i64>,
    t1: Vec<i64>,
    n_splits: usize,
    pct_embargo: f64,
) -> PyResult<Vec<Bound<'py, PyDict>>> {
    let (cv, n) = purged_kfold(t0, t1, n_splits, pct_embargo)?;
    cv.split_with_diagnostics(n)
        .map_err(to_py_err)?
        .into_iter()
        .map(|s| split_to_dict(py, s))
        .collect()
}

/// Combinatorial purged cross-validation splits (AFML section 12.4).
///
/// One split is built for each of the C(`n_splits`, `n_test_splits`) ways to choose the test
/// folds, in lexicographic order of the chosen folds. Each contiguous run of test folds is
/// purged and embargoed exactly like a `purged_kfold_splits` fold. All splits are materialised
/// at once: `n_splits=10, n_test_splits=5` is already 252 splits. The `split_id` numbering
/// matches `openquant.backtesting_engine.run_cpcv`, which expects one return array per split
/// in this order.
///
/// Parameters
/// ----------
/// t0 : list[int]
///     Start of each sample's label span, as int64 nanoseconds since the epoch (plain integers
///     such as bar positions also work; only the order of the values matters). One per sample,
///     in time order.
/// t1 : list[int]
///     End of each sample's label span, in the same units as `t0`. Must be `>= t0`.
/// n_splits : int
///     Number of contiguous test folds; `2 <= n_splits <= len(t0)`.
/// pct_embargo : float
///     Embargo as a fraction of the whole sample count, in `[0, 1)`; rounded up to a number of
///     samples, so any positive value embargoes at least one. AFML suggests about 0.01.
/// n_test_splits : int
///     Number of folds tested together in each split; `1 <= n_test_splits < n_splits`.
///
/// Returns
/// -------
/// list[dict[str, Any]]
///     One dict per split, in `split_id` order, with the keys of `split_with_diagnostics`:
///
///     - `split_id` (int): position of the split in the list.
///     - `train_indices` (list[int]): sorted training indices after purging and embargo.
///     - `test_indices` (list[int]): sorted test indices.
///     - `test_ranges` (list[tuple[int, int]]): the test set as half-open `[start, stop)` blocks.
///     - `purged_indices` (list[int]): non-test samples removed by purging.
///     - `embargo_indices` (list[int]): non-test samples inside an embargo window (a sample can
///       be both purged and embargoed).
///     - `overlap_count_after_purge` (int): training samples whose span still overlaps a test
///       span; always 0, reported so callers can assert it.
///
///     plus `test_fold_ids` (list[int]): the folds tested in this split, ascending.
///
/// Raises
/// ------
/// ValueError
///     If `t0` and `t1` differ in length, or if the core rejects the input (e.g. no samples,
///     `n_splits` below 2 or above the sample count, `pct_embargo` not a finite number in
///     `[0, 1)`, a span that ends before it starts, `n_test_splits` not in `[1, n_splits)`,
///     or a split count too large for the platform's integers).
#[pyfunction(name = "cpcv_splits")]
fn cv_cpcv_splits<'py>(
    py: Python<'py>,
    t0: Vec<i64>,
    t1: Vec<i64>,
    n_splits: usize,
    n_test_splits: usize,
    pct_embargo: f64,
) -> PyResult<Vec<Bound<'py, PyDict>>> {
    let (cv, n) = purged_kfold(t0, t1, n_splits, pct_embargo)?;
    cv.cpcv_splits(n, n_test_splits)
        .map_err(to_py_err)?
        .into_iter()
        .map(|c| {
            let d = split_to_dict(py, c.split)?;
            d.set_item("test_fold_ids", c.test_fold_ids)?;
            Ok(d)
        })
        .collect()
}

/// The CPCV backtest paths for `n_splits` folds tested `n_test_splits` at a time (AFML 12.4).
///
/// There are phi = `n_test_splits / n_splits * C(n_splits, n_test_splits)` paths, and each fold
/// is tested in exactly phi splits of `cpcv_splits`. Path `j` takes, for every fold, the `j`-th
/// split (in `split_id` order) that tests it; stitching those predictions together gives one
/// out-of-sample prediction for every sample. Paths depend on the fold counts alone, so no
/// label spans are needed.
///
/// Parameters
/// ----------
/// n_splits : int
///     Number of contiguous folds; at least 2.
/// n_test_splits : int
///     Number of folds tested together in each split; `1 <= n_test_splits < n_splits`.
///
/// Returns
/// -------
/// list[list[int]]
///     One list per path, in path order; element `g` is the `split_id` of the `cpcv_splits`
///     split whose predictions for fold `g` the path uses.
///
/// Raises
/// ------
/// ValueError
///     If the core rejects the input (e.g. `n_splits` below 2, `n_test_splits` not in
///     `[1, n_splits)`, or a split count too large for the platform's integers).
#[pyfunction(name = "cpcv_paths")]
fn cv_cpcv_paths(n_splits: usize, n_test_splits: usize) -> PyResult<Vec<Vec<usize>>> {
    // Paths depend on the fold count alone. `cpcv_paths` is a method of `PurgedKFold`, so build
    // one over `n_splits` placeholder labels; they never reach the result.
    let placeholder = vec![(NaiveDateTime::default(), NaiveDateTime::default()); n_splits.max(1)];
    let cv = PurgedKFold::new(n_splits, placeholder, 0.0).map_err(to_py_err)?;
    Ok(cv
        .cpcv_paths(n_test_splits)
        .map_err(to_py_err)?
        .into_iter()
        .map(|p| p.split_for_fold)
        .collect())
}

/// Unpurged k-fold splits: contiguous test folds, every other sample trains.
///
/// This is the baseline AFML section 7.3 warns against. It exists to measure leakage (for
/// example with `count_train_test_overlaps`), not to validate models. Folds are contiguous and
/// never shuffled; the first `n_samples % n_splits` folds hold one extra sample.
///
/// Parameters
/// ----------
/// n_samples : int
///     Number of samples.
/// n_splits : int
///     Number of folds; `2 <= n_splits <= n_samples`.
///
/// Returns
/// -------
/// list[tuple[list[int], list[int]]]
///     One `(train_indices, test_indices)` pair per fold, in fold order; both lists sorted.
///
/// Raises
/// ------
/// ValueError
///     If the core rejects the input (`n_splits` below 2 or above `n_samples`).
#[pyfunction(name = "naive_kfold_splits")]
fn cv_naive_kfold_splits(
    n_samples: usize,
    n_splits: usize,
) -> PyResult<Vec<(Vec<usize>, Vec<usize>)>> {
    naive_kfold_splits(n_samples, n_splits).map_err(to_py_err)
}

/// Count the training samples whose label span overlaps some test sample's span.
///
/// Spans are compared as closed intervals, so a label that ends exactly when a test label
/// starts counts as an overlap. Use it with `naive_kfold_splits` to measure the leak that
/// purging removes (AFML section 7.3); on a purged split it is always 0.
///
/// Parameters
/// ----------
/// t0 : list[int]
///     Start of each sample's label span, as int64 nanoseconds since the epoch (plain integers
///     such as bar positions also work; only the order of the values matters). One per sample,
///     in time order.
/// t1 : list[int]
///     End of each sample's label span, in the same units as `t0`.
/// train_indices : list[int]
///     Positions of the training samples in `t0`/`t1`.
/// test_indices : list[int]
///     Positions of the test samples in `t0`/`t1`.
///
/// Returns
/// -------
/// int
///     Number of entries of `train_indices` that overlap at least one test span.
///
/// Raises
/// ------
/// ValueError
///     If `t0` and `t1` differ in length, or if an index in `train_indices` or `test_indices`
///     is not a valid position in them.
#[pyfunction(name = "count_train_test_overlaps")]
fn cv_count_train_test_overlaps(
    t0: Vec<i64>,
    t1: Vec<i64>,
    train_indices: Vec<usize>,
    test_indices: Vec<usize>,
) -> PyResult<usize> {
    let spans = label_spans(t0, t1)?;
    count_train_test_overlaps(&spans, &train_indices, &test_indices).map_err(to_py_err)
}

pub fn register(py: Python<'_>, parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(py, "cross_validation")?;
    m.add_function(wrap_pyfunction!(cv_purged_kfold_splits, &m)?)?;
    m.add_function(wrap_pyfunction!(cv_split_with_diagnostics, &m)?)?;
    m.add_function(wrap_pyfunction!(cv_cpcv_splits, &m)?)?;
    m.add_function(wrap_pyfunction!(cv_cpcv_paths, &m)?)?;
    m.add_function(wrap_pyfunction!(cv_naive_kfold_splits, &m)?)?;
    m.add_function(wrap_pyfunction!(cv_count_train_test_overlaps, &m)?)?;
    parent.add_submodule(&m)?;
    parent.add("cross_validation", m)?;
    Ok(())
}

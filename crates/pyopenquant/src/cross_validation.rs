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

#[pyfunction(name = "naive_kfold_splits")]
fn cv_naive_kfold_splits(
    n_samples: usize,
    n_splits: usize,
) -> PyResult<Vec<(Vec<usize>, Vec<usize>)>> {
    naive_kfold_splits(n_samples, n_splits).map_err(to_py_err)
}

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

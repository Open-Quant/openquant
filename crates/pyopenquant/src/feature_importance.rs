//! `openquant._core.feature_importance`: MDI from per-tree importances, and MDA / SFI scored
//! from out-of-sample probabilities the caller computed on purged folds.
//!
//! The Rust MDA and SFI fit a `SimpleClassifier` themselves. A Python model cannot be one
//! without passing a Python callable into Rust, so these bindings hand the Rust functions a
//! `Replay` classifier instead: `fit` does nothing and `predict_proba` returns, in call order,
//! the probabilities the caller already produced. The scoring, the MDA normalisation and the
//! mean / standard-error aggregation are therefore the Rust code's own.
//!
//! The folds are always rebuilt here from the label spans with `PurgedKFold`, so there is no
//! way to score importance on unpurged folds from Python (issue #27 made spans mandatory).

use std::cell::{Cell, RefCell};
use std::collections::BTreeMap;

use openquant::cross_validation::{Scoring, SimpleClassifier};
use openquant::feature_importance::{
    mean_decrease_accuracy, mean_decrease_impurity, single_feature_importance, ImportanceStats,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::cross_validation::purged_kfold;
use crate::helpers::to_py_err;

/// `{feature: (mean, std)}`.
type ImportanceMap = BTreeMap<String, (f64, f64)>;

fn scoring_from(name: &str) -> PyResult<Scoring> {
    match name {
        "accuracy" => Ok(Scoring::Accuracy),
        "neg_log_loss" => Ok(Scoring::NegLogLoss),
        "f1" => Ok(Scoring::F1),
        other => Err(PyValueError::new_err(format!(
            "scoring must be one of 'neg_log_loss', 'accuracy', 'f1', got '{other}'"
        ))),
    }
}

fn to_map(stats: BTreeMap<String, ImportanceStats>) -> ImportanceMap {
    stats.into_iter().map(|(k, v)| (k, (v.mean, v.std))).collect()
}

/// Plays back precomputed probabilities, one vector per `predict_proba` call.
struct Replay {
    queue: Vec<Vec<f64>>,
    next: Cell<usize>,
    error: RefCell<Option<String>>,
}

impl Replay {
    fn new(queue: Vec<Vec<f64>>) -> Self {
        Self { queue, next: Cell::new(0), error: RefCell::new(None) }
    }

    /// Fails unless every queued vector was consumed, in order, with the expected length.
    fn finish(self) -> PyResult<()> {
        if let Some(msg) = self.error.into_inner() {
            return Err(PyValueError::new_err(msg));
        }
        if self.next.get() != self.queue.len() {
            return Err(PyValueError::new_err(format!(
                "internal error: {} of {} prediction blocks were scored",
                self.next.get(),
                self.queue.len()
            )));
        }
        Ok(())
    }
}

impl SimpleClassifier for Replay {
    fn fit(&mut self, _x: &[Vec<f64>], _y: &[f64], _sample_weight: Option<&[f64]>) {}

    fn predict_proba(&self, x: &[Vec<f64>]) -> Vec<f64> {
        let i = self.next.get();
        self.next.set(i + 1);
        match self.queue.get(i) {
            Some(p) if p.len() == x.len() => p.clone(),
            _ => {
                self.error.borrow_mut().get_or_insert_with(|| {
                    format!("internal error: prediction block {i} does not match its test fold")
                });
                vec![0.5; x.len()]
            }
        }
    }
}

/// Checks a probability column: one finite value in [0, 1] per sample.
fn check_proba(name: &str, values: &[f64], n: usize) -> PyResult<()> {
    if values.len() != n {
        return Err(PyValueError::new_err(format!(
            "{name} has {} values, expected one per sample ({n})",
            values.len()
        )));
    }
    if let Some(v) = values.iter().find(|v| !v.is_finite() || **v < 0.0 || **v > 1.0) {
        return Err(PyValueError::new_err(format!(
            "{name} must hold probabilities in [0, 1], got {v}"
        )));
    }
    Ok(())
}

fn check_common(
    y: &[f64],
    feature_names: &[String],
    sample_weight: Option<&[f64]>,
    n: usize,
) -> PyResult<()> {
    if y.len() != n {
        return Err(PyValueError::new_err(format!(
            "y has {} values but t0/t1 describe {n} samples",
            y.len()
        )));
    }
    if feature_names.is_empty() {
        return Err(PyValueError::new_err("feature_names cannot be empty"));
    }
    if let Some(sw) = sample_weight {
        if sw.len() != n {
            return Err(PyValueError::new_err(format!(
                "sample_weight has {} values, expected {n}",
                sw.len()
            )));
        }
    }
    Ok(())
}

#[pyfunction(name = "mean_decrease_impurity")]
fn fi_mean_decrease_impurity(
    per_tree_importances: Vec<Vec<f64>>,
    feature_names: Vec<String>,
) -> PyResult<ImportanceMap> {
    mean_decrease_impurity(&per_tree_importances, &feature_names).map(to_map).map_err(to_py_err)
}

// The spans, the fold settings and the caller's predictions are all independent inputs.
#[allow(clippy::too_many_arguments)]
#[pyfunction(name = "mda_from_probabilities")]
#[pyo3(signature = (
    y,
    t0,
    t1,
    base_proba,
    permuted_proba,
    feature_names,
    *,
    n_splits,
    pct_embargo,
    scoring,
    sample_weight=None
))]
fn fi_mda_from_probabilities(
    y: Vec<f64>,
    t0: Vec<i64>,
    t1: Vec<i64>,
    base_proba: Vec<f64>,
    permuted_proba: Vec<Vec<f64>>,
    feature_names: Vec<String>,
    n_splits: usize,
    pct_embargo: f64,
    scoring: &str,
    sample_weight: Option<Vec<f64>>,
) -> PyResult<ImportanceMap> {
    let scoring = scoring_from(scoring)?;
    let (cv, n) = purged_kfold(t0, t1, n_splits, pct_embargo)?;
    check_common(&y, &feature_names, sample_weight.as_deref(), n)?;
    check_proba("base_proba", &base_proba, n)?;
    if permuted_proba.len() != feature_names.len() {
        return Err(PyValueError::new_err(format!(
            "permuted_proba has {} columns, expected one per feature ({})",
            permuted_proba.len(),
            feature_names.len()
        )));
    }
    for (name, column) in feature_names.iter().zip(&permuted_proba) {
        check_proba(&format!("permuted_proba for '{name}'"), column, n)?;
    }
    let splits = cv.split(n).map_err(to_py_err)?;

    // `mean_decrease_accuracy` scores, per fold, the unpermuted test set and then each feature
    // permuted in turn.
    let mut queue = Vec::with_capacity(splits.len() * (feature_names.len() + 1));
    for (_, test) in &splits {
        queue.push(test.iter().map(|&i| base_proba[i]).collect());
        for column in &permuted_proba {
            queue.push(test.iter().map(|&i| column[i]).collect());
        }
    }
    // The model never reads its features, so a one-column placeholder per sample suffices;
    // it only needs one column per feature name.
    let x = vec![vec![0.0; feature_names.len()]; n];
    let mut replay = Replay::new(queue);
    let out = mean_decrease_accuracy(
        &mut replay,
        &x,
        &y,
        &feature_names,
        &splits,
        sample_weight.as_deref(),
        scoring,
    )
    .map_err(to_py_err)?;
    replay.finish()?;
    Ok(to_map(out))
}

#[allow(clippy::too_many_arguments)]
#[pyfunction(name = "sfi_from_probabilities")]
#[pyo3(signature = (
    y,
    t0,
    t1,
    proba,
    feature_names,
    *,
    n_splits,
    pct_embargo,
    scoring,
    sample_weight=None
))]
fn fi_sfi_from_probabilities(
    y: Vec<f64>,
    t0: Vec<i64>,
    t1: Vec<i64>,
    proba: Vec<Vec<f64>>,
    feature_names: Vec<String>,
    n_splits: usize,
    pct_embargo: f64,
    scoring: &str,
    sample_weight: Option<Vec<f64>>,
) -> PyResult<ImportanceMap> {
    let scoring = scoring_from(scoring)?;
    let (cv, n) = purged_kfold(t0, t1, n_splits, pct_embargo)?;
    check_common(&y, &feature_names, sample_weight.as_deref(), n)?;
    if proba.len() != feature_names.len() {
        return Err(PyValueError::new_err(format!(
            "proba has {} columns, expected one per feature ({})",
            proba.len(),
            feature_names.len()
        )));
    }
    for (name, column) in feature_names.iter().zip(&proba) {
        check_proba(&format!("proba for '{name}'"), column, n)?;
    }
    let splits = cv.split(n).map_err(to_py_err)?;

    // `single_feature_importance` cross-validates each feature in turn, fold by fold.
    let mut queue = Vec::with_capacity(splits.len() * feature_names.len());
    for column in &proba {
        for (_, test) in &splits {
            queue.push(test.iter().map(|&i| column[i]).collect());
        }
    }
    let x = vec![vec![0.0; feature_names.len()]; n];
    let mut replay = Replay::new(queue);
    let out = single_feature_importance(
        &mut replay,
        &x,
        &y,
        &feature_names,
        &splits,
        sample_weight.as_deref(),
        scoring,
    )
    .map_err(to_py_err)?;
    replay.finish()?;
    Ok(to_map(out))
}

pub fn register(py: Python<'_>, parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(py, "feature_importance")?;
    m.add_function(wrap_pyfunction!(fi_mean_decrease_impurity, &m)?)?;
    m.add_function(wrap_pyfunction!(fi_mda_from_probabilities, &m)?)?;
    m.add_function(wrap_pyfunction!(fi_sfi_from_probabilities, &m)?)?;
    parent.add_submodule(&m)?;
    parent.add("feature_importance", m)?;
    Ok(())
}

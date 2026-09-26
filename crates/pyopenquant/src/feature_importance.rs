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

/// Mean decrease impurity (MDI) aggregated over the trees of a forest (AFML Snippet 8.2).
///
/// Following the snippet, a zero importance is treated as missing: the snippet trains with
/// `max_features=1`, where 0 means "never offered to the tree". With other settings this
/// inflates the mean, so replace zeros with a tiny positive number if 0 means "useless". Per
/// feature the mean over non-missing trees is taken, and its standard error is the sample
/// deviation (ddof 1) times `n_trees ** -0.5`, with `n_trees` counting every tree. Means and
/// standard errors are then divided by the sum of the means, so the means sum to 1. A feature
/// that is zero in every tree gets 0; if no mean is positive, everything is 0.
///
/// Parameters
/// ----------
/// per_tree_importances : list[list[float]]
///     One row per tree, each with one impurity importance per feature in `feature_names` order
///     (e.g. `[t.feature_importances_ for t in forest.estimators_]` in scikit-learn).
/// feature_names : list[str]
///     Feature names, one per column of `per_tree_importances`.
///
/// Returns
/// -------
/// dict[str, tuple[float, float]]
///     `{feature: (mean, std)}`, keyed by feature name (in sorted order). The means sum to 1;
///     `std` is the normalised standard error of the mean, not the deviation.
///
/// Raises
/// ------
/// ValueError
///     If `per_tree_importances` or `feature_names` is empty, or a row does not have one entry
///     per feature name.
#[pyfunction(name = "mean_decrease_impurity")]
fn fi_mean_decrease_impurity(
    per_tree_importances: Vec<Vec<f64>>,
    feature_names: Vec<String>,
) -> PyResult<ImportanceMap> {
    mean_decrease_impurity(&per_tree_importances, &feature_names).map(to_map).map_err(to_py_err)
}

// The spans, the fold settings and the caller's predictions are all independent inputs.
#[allow(clippy::too_many_arguments)]
/// Mean decrease accuracy (MDA) from out-of-sample probabilities you computed (AFML 8.3).
///
/// The purged folds are rebuilt here with `PurgedKFold(n_splits, spans, pct_embargo)`; they are
/// the folds of `openquant.cross_validation.purged_kfold_splits` with the same arguments. For
/// each fold, fit your model on the training indices, predict the test indices to get
/// `base_proba`, then, for each feature `j`, shuffle column `j` within the test rows and predict
/// again to get `permuted_proba[j]`. Because the test folds partition the samples, each column
/// holds one out-of-sample value per sample. The Rust MDA then scores, per fold, the base and
/// each permuted prediction; the per-fold importance is `(base - perm) / (0 - perm)` for
/// `"neg_log_loss"` and `(base - perm) / (1 - perm)` for `"accuracy"` and `"f1"` (0 when the
/// denominator is 0 or the ratio is not finite), and the result is its mean over folds. Test-fold
/// scores are weighted by `sample_weight`.
///
/// 1 means shuffling the feature destroyed everything the model had, 0 that the model did not
/// need it, and a negative value that it did better without it.
///
/// Parameters
/// ----------
/// y : list[float]
///     Labels, each 0.0 or 1.0, one per sample.
/// t0 : list[int]
///     Start of each sample's label span, as int64 nanoseconds since the epoch (plain integers
///     also work). One per sample, in time order.
/// t1 : list[int]
///     End of each sample's label span, in the same units as `t0`. Must be `>= t0`.
/// base_proba : list[float]
///     Out-of-sample probability of class 1 per sample, from the unpermuted test fold; each in
///     `[0, 1]`.
/// permuted_proba : list[list[float]]
///     One column per feature, in `feature_names` order; column `j` holds, per sample, the
///     out-of-sample probability of class 1 with feature `j` shuffled within the test fold.
/// feature_names : list[str]
///     Feature names; must not be empty.
/// n_splits : int
///     Number of purged folds; `2 <= n_splits <= len(y)`. Keyword-only.
/// pct_embargo : float
///     Embargo as a fraction of the whole sample count, in `[0, 1)`, rounded up. Keyword-only.
/// scoring : str
///     One of `"neg_log_loss"`, `"accuracy"` or `"f1"` (F1 of the positive class). Accuracy
///     and F1 threshold the probabilities at 0.5. Keyword-only.
/// sample_weight : list[float] | None, default None
///     Weight per sample, applied to the test-fold scores. Keyword-only.
/// seed : int, default 42
///     Accepted for signature parity with the Rust MDA. It has no effect: the shuffles are
///     already in `permuted_proba`. Keyword-only.
///
/// Returns
/// -------
/// dict[str, tuple[float, float]]
///     `{feature: (mean, std)}`, keyed by feature name (in sorted order). `std` is the standard
///     error of the mean over folds, not the deviation.
///
/// Raises
/// ------
/// ValueError
///     If `scoring` is not one of the names above; if `t0` and `t1`, `y`, `base_proba`, a
///     column of `permuted_proba` or `sample_weight` differ in length from the sample count; if
///     `feature_names` is empty or `permuted_proba` does not have one column per feature; if a
///     probability is not finite or outside `[0, 1]`; or if the core rejects the folds (e.g. no
///     samples, `n_splits` below 2 or above the sample count, `pct_embargo` not a finite number in
///     `[0, 1)`, a span that ends before it starts).
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
    sample_weight=None,
    seed=42
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
    seed: u64,
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
    // The model never reads its features, so a placeholder row per sample suffices; it only
    // needs one column per feature name. For the same reason the permutation `seed` drives
    // (Rust shuffles each placeholder column) cannot change the result: the caller's
    // `permuted_proba` already holds the shuffled predictions.
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
        seed,
    )
    .map_err(to_py_err)?;
    replay.finish()?;
    Ok(to_map(out))
}

#[allow(clippy::too_many_arguments)]
/// Single feature importance (SFI) from out-of-sample probabilities you computed (AFML 8.4).
///
/// The purged folds are rebuilt here with `PurgedKFold(n_splits, spans, pct_embargo)`; they are
/// the folds of `openquant.cross_validation.purged_kfold_splits` with the same arguments. For
/// each feature `j` and each fold, fit your model on feature `j` alone over the training
/// indices and predict the test indices; `proba[j]` holds those out-of-sample probabilities,
/// one per sample. The value per feature is the raw cross-validated score, not a ratio: for
/// `"neg_log_loss"` compare it with `-ln 2` (about -0.693), a coin flip. The standard error is
/// the population deviation (ddof 0) over folds divided by `sqrt(n_splits)`.
///
/// Parameters
/// ----------
/// y : list[float]
///     Labels, each 0.0 or 1.0, one per sample.
/// t0 : list[int]
///     Start of each sample's label span, as int64 nanoseconds since the epoch (plain integers
///     also work). One per sample, in time order.
/// t1 : list[int]
///     End of each sample's label span, in the same units as `t0`. Must be `>= t0`.
/// proba : list[list[float]]
///     One column per feature, in `feature_names` order; column `j` holds, per sample, the
///     out-of-sample probability of class 1 from the model fitted on feature `j` alone.
/// feature_names : list[str]
///     Feature names; must not be empty.
/// n_splits : int
///     Number of purged folds; `2 <= n_splits <= len(y)`. Keyword-only.
/// pct_embargo : float
///     Embargo as a fraction of the whole sample count, in `[0, 1)`, rounded up. Keyword-only.
/// scoring : str
///     One of `"neg_log_loss"`, `"accuracy"` or `"f1"` (F1 of the positive class). Accuracy
///     and F1 threshold the probabilities at 0.5. Keyword-only.
/// sample_weight : list[float] | None, default None
///     Weight per sample. Its length is checked, but it does not affect the result: the Rust
///     SFI passes weights only to model fitting (which happened on your side) and scores the
///     test folds unweighted. Keyword-only.
///
/// Returns
/// -------
/// dict[str, tuple[float, float]]
///     `{feature: (mean, std)}`, keyed by feature name (in sorted order). `std` is the standard
///     error of the mean over folds, not the deviation.
///
/// Raises
/// ------
/// ValueError
///     If `scoring` is not one of the names above; if `t0` and `t1`, `y`, a column of `proba`
///     or `sample_weight` differ in length from the sample count; if `feature_names` is empty
///     or `proba` does not have one column per feature; if a probability is not finite or
///     outside `[0, 1]`; or if the core rejects the folds (e.g. no samples,
///     `n_splits` below 2 or above the sample count, `pct_embargo` not a finite number in
///     `[0, 1)`, a span that ends before it starts).
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

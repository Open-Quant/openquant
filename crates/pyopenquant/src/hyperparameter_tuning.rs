//! `openquant._core.hyperparameter_tuning`: the parts of the purged search that need no model.
//!
//! `grid_search` and `randomized_search` build and fit a `SimpleClassifier` per candidate, so
//! they are not bound. What is bound is everything around the model: the candidate parameter
//! sets (in the order the Rust searches evaluate them) and the sample-weighted fold score. The
//! pure-Python `openquant.hyperparameter_tuning` runs the fit loop on purged folds.

use std::collections::BTreeMap;

use openquant::hyperparameter_tuning::{
    classification_score, expand_param_grid, sample_param_sets, HyperParamValue, ParamSet,
    RandomParamDistribution, SearchScoring,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyFloat, PyInt};

use crate::helpers::to_py_err;

fn value_from(key: &str, v: &Bound<'_, PyAny>) -> PyResult<HyperParamValue> {
    // `bool` is a subclass of `int` in Python, so it is checked first.
    if v.is_instance_of::<PyBool>() {
        Ok(HyperParamValue::Bool(v.extract()?))
    } else if v.is_instance_of::<PyInt>() {
        Ok(HyperParamValue::Int(v.extract()?))
    } else if v.is_instance_of::<PyFloat>() {
        Ok(HyperParamValue::Float(v.extract()?))
    } else {
        Err(PyValueError::new_err(format!(
            "parameter '{key}': values must be int, float or bool, got {}",
            v.get_type().name()?
        )))
    }
}

fn value_to_py<'py>(py: Python<'py>, v: &HyperParamValue) -> PyResult<Bound<'py, PyAny>> {
    Ok(match v {
        HyperParamValue::Int(i) => i.into_pyobject(py)?.into_any(),
        HyperParamValue::Float(f) => f.into_pyobject(py)?.into_any(),
        HyperParamValue::Bool(b) => b.into_pyobject(py)?.to_owned().into_any(),
    })
}

fn param_sets_to_py<'py>(
    py: Python<'py>,
    sets: Vec<ParamSet>,
) -> PyResult<Vec<Bound<'py, PyDict>>> {
    sets.iter()
        .map(|set| {
            let d = PyDict::new(py);
            for (k, v) in set {
                d.set_item(k, value_to_py(py, v)?)?;
            }
            Ok(d)
        })
        .collect()
}

fn distribution_from(key: &str, spec: &Bound<'_, PyAny>) -> PyResult<RandomParamDistribution> {
    let bad = || {
        PyValueError::new_err(format!(
            "parameter '{key}': expected ('choice', [values]), ('uniform', low, high), \
             ('log_uniform', low, high) or ('int', low, high)"
        ))
    };
    let items: Vec<Bound<'_, PyAny>> = spec.extract().map_err(|_| bad())?;
    let kind: String = items.first().ok_or_else(bad)?.extract().map_err(|_| bad())?;
    match (kind.as_str(), items.len()) {
        ("choice", 2) => {
            let values: Vec<Bound<'_, PyAny>> = items[1].extract().map_err(|_| bad())?;
            Ok(RandomParamDistribution::Choice(
                values.iter().map(|v| value_from(key, v)).collect::<PyResult<_>>()?,
            ))
        }
        ("uniform", 3) => Ok(RandomParamDistribution::Uniform {
            low: items[1].extract()?,
            high: items[2].extract()?,
        }),
        ("log_uniform", 3) => Ok(RandomParamDistribution::LogUniform {
            low: items[1].extract()?,
            high: items[2].extract()?,
        }),
        ("int", 3) => Ok(RandomParamDistribution::IntRangeInclusive {
            low: items[1].extract()?,
            high: items[2].extract()?,
        }),
        _ => Err(bad()),
    }
}

/// Expand a parameter grid into every combination of its values (AFML section 9.2).
///
/// These are the candidates a purged grid search evaluates, in the order the Rust
/// `grid_search` evaluates them: keys are iterated in sorted name order, with the last key
/// varying fastest. Fit and score each candidate on purged folds (e.g.
/// `openquant.cross_validation.purged_kfold_splits`) with your own model.
///
/// Parameters
/// ----------
/// param_grid : dict[str, list[int | float | bool]]
///     Parameter name to the values to try. Values must be Python `int`, `float` or `bool`
///     (subclasses such as `numpy.float64` are accepted; `numpy.int64` and `numpy.bool_` are
///     not, so convert them with `int()` or `bool()`).
///
/// Returns
/// -------
/// list[dict[str, int | float | bool]]
///     One dict per combination, each with every key of `param_grid`; values keep their
///     Python type.
///
/// Raises
/// ------
/// ValueError
///     If a value is not an int, float or bool, or if the core rejects the grid (no keys, or a
///     key with no values).
#[pyfunction(name = "expand_param_grid")]
fn ht_expand_param_grid<'py>(
    py: Python<'py>,
    param_grid: BTreeMap<String, Vec<Bound<'py, PyAny>>>,
) -> PyResult<Vec<Bound<'py, PyDict>>> {
    let grid = param_grid
        .iter()
        .map(|(k, values)| {
            let parsed = values.iter().map(|v| value_from(k, v)).collect::<PyResult<Vec<_>>>()?;
            Ok((k.clone(), parsed))
        })
        .collect::<PyResult<BTreeMap<_, _>>>()?;
    param_sets_to_py(py, expand_param_grid(&grid).map_err(to_py_err)?)
}

/// Draw the `n_iter` parameter sets a randomized search evaluates (AFML section 9.3).
///
/// Each draw samples every key of `param_space` in sorted name order from one random generator
/// seeded with `seed`, so the same inputs always give the same sets, in the order the Rust
/// `randomized_search` evaluates them. Fit and score each set on purged folds with your own
/// model.
///
/// Parameters
/// ----------
/// param_space : dict[str, tuple]
///     Parameter name to a distribution spec, as a tuple or list:
///
///     - `("choice", [values])`: uniform choice among the values (each an `int`, `float` or
///       `bool`, with the same rules as `expand_param_grid`).
///     - `("uniform", low, high)`: float uniform in `[low, high)`; finite, `low < high`.
///     - `("log_uniform", low, high)`: float whose logarithm is uniform in
///       `[ln low, ln high)` (AFML section 9.3.1); `0 < low < high`.
///     - `("int", low, high)`: integer uniform in `[low, high]`, both inclusive; `low <= high`.
/// n_iter : int
///     Number of parameter sets to draw; must be positive.
/// seed : int
///     Seed of the random generator.
///
/// Returns
/// -------
/// list[dict[str, int | float | bool]]
///     `n_iter` dicts, in draw order, each with every key of `param_space`. `uniform` and
///     `log_uniform` give floats, `int` gives ints, and `choice` keeps the chosen value's type.
///
/// Raises
/// ------
/// ValueError
///     If a spec is malformed (unknown kind or wrong number of items) or a choice value is not
///     an int, float or bool, or if the core rejects the space (no keys, `n_iter == 0`, an
///     empty choice list, non-finite or unordered `uniform` bounds, non-positive or unordered
///     `log_uniform` bounds, or `int` with `low > high`).
#[pyfunction(name = "sample_param_sets")]
fn ht_sample_param_sets<'py>(
    py: Python<'py>,
    param_space: BTreeMap<String, Bound<'py, PyAny>>,
    n_iter: usize,
    seed: u64,
) -> PyResult<Vec<Bound<'py, PyDict>>> {
    let space = param_space
        .iter()
        .map(|(k, spec)| Ok((k.clone(), distribution_from(k, spec)?)))
        .collect::<PyResult<BTreeMap<_, _>>>()?;
    param_sets_to_py(py, sample_param_sets(&space, n_iter, seed).map_err(to_py_err)?)
}

/// Score binary predictions, optionally sample-weighted (AFML Snippet 9.1).
///
/// Scoring the test fold with its sample weights is the correction Snippet 9.1 makes to
/// scikit-learn. Every score is higher-is-better. Samples with weight 0 are skipped. Hard
/// predictions use `probability >= 0.5` for class 1. Log loss clips probabilities to
/// `[1e-15, 1 - 1e-15]`; AFML section 9.4 recommends it for strategies that size bets by
/// probability.
///
/// Parameters
/// ----------
/// y_true : list[float]
///     Labels, each 0.0 or 1.0.
/// probabilities : list[float]
///     Predicted probability of class 1 per sample, finite and in `[0, 1]`.
/// sample_weight : list[float] | None, default None
///     Non-negative weight per sample; None weights every sample 1.
/// scoring : str, default "neg_log_loss"
///     One of `"neg_log_loss"` (weighted mean log-likelihood of the true label, at most 0),
///     `"accuracy"` (weighted share of labels matched) or `"balanced_accuracy"` (mean of the
///     weighted per-class recalls, over the classes present).
///
/// Returns
/// -------
/// float
///     The score.
///
/// Raises
/// ------
/// ValueError
///     If `scoring` is not one of the names above, or if the core rejects the input (empty
///     `y_true`, `probabilities` or `sample_weight` of a different length, a negative weight,
///     a probability not finite or outside `[0, 1]`, a label other than 0 or 1, or weights
///     summing to 0).
#[pyfunction(name = "classification_score")]
#[pyo3(signature = (y_true, probabilities, sample_weight=None, scoring="neg_log_loss"))]
fn ht_classification_score(
    y_true: Vec<f64>,
    probabilities: Vec<f64>,
    sample_weight: Option<Vec<f64>>,
    scoring: &str,
) -> PyResult<f64> {
    let scoring = match scoring {
        "neg_log_loss" => SearchScoring::NegLogLoss,
        "accuracy" => SearchScoring::Accuracy,
        "balanced_accuracy" => SearchScoring::BalancedAccuracy,
        other => {
            return Err(PyValueError::new_err(format!(
                "scoring must be one of 'neg_log_loss', 'accuracy', 'balanced_accuracy', got \
                 '{other}'"
            )))
        }
    };
    classification_score(&y_true, &probabilities, sample_weight.as_deref(), scoring)
        .map_err(to_py_err)
}

pub fn register(py: Python<'_>, parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(py, "hyperparameter_tuning")?;
    m.add_function(wrap_pyfunction!(ht_expand_param_grid, &m)?)?;
    m.add_function(wrap_pyfunction!(ht_sample_param_sets, &m)?)?;
    m.add_function(wrap_pyfunction!(ht_classification_score, &m)?)?;
    parent.add_submodule(&m)?;
    parent.add("hyperparameter_tuning", m)?;
    Ok(())
}

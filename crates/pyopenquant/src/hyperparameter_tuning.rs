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

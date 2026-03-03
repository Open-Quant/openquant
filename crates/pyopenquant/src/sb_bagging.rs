use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::helpers::{matrix_from_rows, to_py_err};

#[pyfunction(name = "fit_predict_sb_classifier")]
#[pyo3(signature = (
    x,
    y,
    ind_mat,
    n_estimators=10,
    max_samples=1.0,
    max_features=1.0,
    random_state=42,
    sample_weight=None
))]
fn sb_fit_predict_classifier(
    py: Python<'_>,
    x: Vec<Vec<f64>>,
    y: Vec<u8>,
    ind_mat: Vec<Vec<u8>>,
    n_estimators: usize,
    max_samples: f64,
    max_features: f64,
    random_state: u64,
    sample_weight: Option<Vec<f64>>,
) -> PyResult<PyObject> {
    let x_mat = matrix_from_rows(x)?;

    let mut clf = openquant::sb_bagging::SequentiallyBootstrappedBaggingClassifier::new(random_state);
    clf.n_estimators = n_estimators;
    clf.max_samples = openquant::sb_bagging::MaxSamples::Float(max_samples);
    clf.max_features = openquant::sb_bagging::MaxFeatures::Float(max_features);
    clf.oob_score = true;

    clf.fit(&x_mat, &y, &ind_mat, sample_weight.as_deref()).map_err(to_py_err)?;
    let predictions = clf.predict(&x_mat).map_err(to_py_err)?;

    let d = PyDict::new(py);
    d.set_item("predictions", predictions)?;
    d.set_item("oob_score", clf.oob_score_value)?;
    Ok(d.into_pyobject(py).unwrap().into_any().unbind())
}

#[pyfunction(name = "fit_predict_sb_regressor")]
#[pyo3(signature = (
    x,
    y,
    ind_mat,
    n_estimators=10,
    max_samples=1.0,
    max_features=1.0,
    random_state=42,
    sample_weight=None
))]
fn sb_fit_predict_regressor(
    py: Python<'_>,
    x: Vec<Vec<f64>>,
    y: Vec<f64>,
    ind_mat: Vec<Vec<u8>>,
    n_estimators: usize,
    max_samples: f64,
    max_features: f64,
    random_state: u64,
    sample_weight: Option<Vec<f64>>,
) -> PyResult<PyObject> {
    let x_mat = matrix_from_rows(x)?;

    let mut reg = openquant::sb_bagging::SequentiallyBootstrappedBaggingRegressor::new(random_state);
    reg.n_estimators = n_estimators;
    reg.max_samples = openquant::sb_bagging::MaxSamples::Float(max_samples);
    reg.max_features = openquant::sb_bagging::MaxFeatures::Float(max_features);
    reg.oob_score = true;

    reg.fit(&x_mat, &y, &ind_mat, sample_weight.as_deref()).map_err(to_py_err)?;
    let predictions = reg.predict(&x_mat).map_err(to_py_err)?;

    let d = PyDict::new(py);
    d.set_item("predictions", predictions)?;
    d.set_item("oob_score", reg.oob_score_value)?;
    Ok(d.into_pyobject(py).unwrap().into_any().unbind())
}

pub fn register(py: Python<'_>, parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(py, "sb_bagging")?;
    m.add_function(wrap_pyfunction!(sb_fit_predict_classifier, &m)?)?;
    m.add_function(wrap_pyfunction!(sb_fit_predict_regressor, &m)?)?;
    parent.add_submodule(&m)?;
    parent.add("sb_bagging", m)?;
    Ok(())
}

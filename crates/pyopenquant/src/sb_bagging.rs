use nalgebra::DMatrix;
use openquant::sb_bagging::{MaxFeatures, MaxSamples, SequentiallyBootstrappedBaggingClassifier};
use pyo3::exceptions::PyValueError;
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
// Python keyword signature.
#[allow(clippy::too_many_arguments)]
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

    let mut clf =
        openquant::sb_bagging::SequentiallyBootstrappedBaggingClassifier::new(random_state);
    clf.n_estimators = n_estimators;
    clf.max_samples = openquant::sb_bagging::MaxSamples::Float(max_samples);
    clf.max_features = openquant::sb_bagging::MaxFeatures::Float(max_features);
    clf.oob_score = true;

    clf.fit(&x_mat, &y, &ind_mat, sample_weight.as_deref()).map_err(to_py_err)?;
    let predictions = clf.predict(&x_mat).map_err(to_py_err)?;

    // Widened so the classes reach Python as a list of ints, not `bytes`.
    let predictions: Vec<u32> = predictions.into_iter().map(u32::from).collect();

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
// Python keyword signature.
#[allow(clippy::too_many_arguments)]
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

    let mut reg =
        openquant::sb_bagging::SequentiallyBootstrappedBaggingRegressor::new(random_state);
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

/// `sb_bagging.SequentiallyBootstrappedBaggingClassifier`: the classifier as an object with
/// `fit`, `predict` and `predict_proba`, so it can be fitted on one set of rows and scored on
/// another (for example, out of fold under purged cross-validation).
///
/// `predict_proba` returns one `[P(y = 0), P(y = 1)]` row per input row, in the order of
/// `classes_` (`[0, 1]`), as scikit-learn classifiers do. The probability of class 1 is the
/// share of estimators voting 1, so `predict` is 1 exactly when it is at least 0.5.
#[pyclass(name = "SequentiallyBootstrappedBaggingClassifier", module = "openquant.sb_bagging")]
struct PySbClassifier {
    inner: SequentiallyBootstrappedBaggingClassifier,
    /// Column count of the matrix passed to the last successful `fit`; `None` until then.
    n_features: Option<usize>,
}

impl PySbClassifier {
    /// Checks the model is fitted and `x` has the training column count, so a mismatched
    /// matrix is a `ValueError` rather than a panic in the core.
    fn checked_matrix(&self, x: Vec<Vec<f64>>) -> PyResult<DMatrix<f64>> {
        let Some(n_features) = self.n_features else {
            return Err(PyValueError::new_err(
                "this SequentiallyBootstrappedBaggingClassifier is not fitted; call fit first",
            ));
        };
        let x_mat = matrix_from_rows(x)?;
        if x_mat.ncols() != n_features {
            return Err(PyValueError::new_err(format!(
                "x has {} feature columns but the model was fitted on {}",
                x_mat.ncols(),
                n_features
            )));
        }
        Ok(x_mat)
    }
}

fn py_bool(v: bool) -> &'static str {
    if v {
        "True"
    } else {
        "False"
    }
}

#[pymethods]
impl PySbClassifier {
    #[new]
    #[pyo3(signature = (
        n_estimators=10,
        max_samples=1.0,
        max_features=1.0,
        bootstrap_features=false,
        oob_score=false,
        random_state=42
    ))]
    fn new(
        n_estimators: usize,
        max_samples: f64,
        max_features: f64,
        bootstrap_features: bool,
        oob_score: bool,
        random_state: u64,
    ) -> Self {
        let mut inner = SequentiallyBootstrappedBaggingClassifier::new(random_state);
        inner.n_estimators = n_estimators;
        inner.max_samples = MaxSamples::Float(max_samples);
        inner.max_features = MaxFeatures::Float(max_features);
        inner.bootstrap_features = bootstrap_features;
        inner.oob_score = oob_score;
        Self { inner, n_features: None }
    }

    /// Fits the ensemble and returns the model, so calls can be chained.
    #[pyo3(signature = (x, y, ind_mat, sample_weight=None))]
    fn fit<'py>(
        mut slf: PyRefMut<'py, Self>,
        x: Vec<Vec<f64>>,
        y: Vec<u8>,
        ind_mat: Vec<Vec<u8>>,
        sample_weight: Option<Vec<f64>>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let x_mat = matrix_from_rows(x)?;
        slf.n_features = None;
        slf.inner.fit(&x_mat, &y, &ind_mat, sample_weight.as_deref()).map_err(to_py_err)?;
        slf.n_features = Some(x_mat.ncols());
        Ok(slf)
    }

    /// Predicted class (0 or 1) for each row of `x`, by majority vote; ties go to 1.
    fn predict(&self, x: Vec<Vec<f64>>) -> PyResult<Vec<u32>> {
        let x_mat = self.checked_matrix(x)?;
        let predictions = self.inner.predict(&x_mat).map_err(to_py_err)?;
        Ok(predictions.into_iter().map(u32::from).collect())
    }

    /// `[P(y = 0), P(y = 1)]` for each row of `x`; the columns follow `classes_`.
    fn predict_proba(&self, x: Vec<Vec<f64>>) -> PyResult<Vec<[f64; 2]>> {
        let x_mat = self.checked_matrix(x)?;
        let positive = self.inner.predict_proba(&x_mat).map_err(to_py_err)?;
        Ok(positive.into_iter().map(|p| [1.0 - p, p]).collect())
    }

    /// The class labels, `[0, 1]`: the column order of `predict_proba`.
    #[getter]
    fn classes_(&self) -> Vec<u32> {
        vec![0, 1]
    }

    /// Column count seen by the last successful `fit`; `None` before one.
    #[getter]
    fn n_features_in_(&self) -> Option<usize> {
        self.n_features
    }

    /// Out-of-bag accuracy from the last `fit` when `oob_score=True`, else `None`.
    #[getter]
    fn oob_score_(&self) -> Option<f64> {
        self.inner.oob_score_value
    }

    /// Row indices each estimator was trained on, one list per estimator.
    #[getter]
    fn estimators_samples_(&self) -> Vec<Vec<usize>> {
        self.inner.estimators_samples.clone()
    }

    #[getter]
    fn n_estimators(&self) -> usize {
        self.inner.n_estimators
    }

    #[getter]
    fn random_state(&self) -> u64 {
        self.inner.random_state
    }

    fn __repr__(&self) -> String {
        let max_samples = match self.inner.max_samples {
            MaxSamples::Float(v) => v.to_string(),
            MaxSamples::Int(v) => v.to_string(),
        };
        let max_features = match self.inner.max_features {
            MaxFeatures::Float(v) => v.to_string(),
            MaxFeatures::Int(v) => v.to_string(),
        };
        format!(
            "SequentiallyBootstrappedBaggingClassifier(n_estimators={}, max_samples={}, \
             max_features={}, bootstrap_features={}, oob_score={}, random_state={})",
            self.inner.n_estimators,
            max_samples,
            max_features,
            py_bool(self.inner.bootstrap_features),
            py_bool(self.inner.oob_score),
            self.inner.random_state
        )
    }
}

pub fn register(py: Python<'_>, parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(py, "sb_bagging")?;
    m.add_class::<PySbClassifier>()?;
    m.add_function(wrap_pyfunction!(sb_fit_predict_classifier, &m)?)?;
    m.add_function(wrap_pyfunction!(sb_fit_predict_regressor, &m)?)?;
    parent.add_submodule(&m)?;
    parent.add("sb_bagging", m)?;
    Ok(())
}

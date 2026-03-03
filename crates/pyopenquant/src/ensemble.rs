use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::helpers::to_py_err;

#[pyfunction(name = "bias_variance_noise")]
fn ens_bias_variance_noise(
    y_true: Vec<f64>,
    per_model_predictions: Vec<Vec<f64>>,
) -> PyResult<(f64, f64, f64, f64)> {
    let result = openquant::ensemble_methods::bias_variance_noise(&y_true, &per_model_predictions)
        .map_err(to_py_err)?;
    Ok((result.bias_sq, result.variance, result.noise, result.mse))
}

#[pyfunction(name = "bootstrap_sample_indices")]
fn ens_bootstrap_sample_indices(
    n_samples: usize,
    sample_size: usize,
    seed: u64,
) -> PyResult<Vec<usize>> {
    openquant::ensemble_methods::bootstrap_sample_indices(n_samples, sample_size, seed)
        .map_err(to_py_err)
}

#[pyfunction(name = "sequential_bootstrap_sample_indices")]
fn ens_sequential_bootstrap_sample_indices(
    ind_mat: Vec<Vec<u8>>,
    sample_size: usize,
    seed: u64,
) -> PyResult<Vec<usize>> {
    openquant::ensemble_methods::sequential_bootstrap_sample_indices(&ind_mat, sample_size, seed)
        .map_err(to_py_err)
}

#[pyfunction(name = "aggregate_regression_mean")]
fn ens_aggregate_regression_mean(per_model_predictions: Vec<Vec<f64>>) -> PyResult<Vec<f64>> {
    openquant::ensemble_methods::aggregate_regression_mean(&per_model_predictions)
        .map_err(to_py_err)
}

#[pyfunction(name = "aggregate_classification_vote")]
fn ens_aggregate_classification_vote(per_model_predictions: Vec<Vec<u8>>) -> PyResult<Vec<u8>> {
    openquant::ensemble_methods::aggregate_classification_vote(&per_model_predictions)
        .map_err(to_py_err)
}

#[pyfunction(name = "aggregate_classification_probability_mean")]
fn ens_aggregate_classification_probability_mean(
    per_model_probabilities: Vec<Vec<f64>>,
    threshold: f64,
) -> PyResult<(Vec<f64>, Vec<u8>)> {
    openquant::ensemble_methods::aggregate_classification_probability_mean(
        &per_model_probabilities,
        threshold,
    )
    .map_err(to_py_err)
}

#[pyfunction(name = "average_pairwise_prediction_correlation")]
fn ens_average_pairwise_prediction_correlation(
    per_model_predictions: Vec<Vec<f64>>,
) -> PyResult<f64> {
    openquant::ensemble_methods::average_pairwise_prediction_correlation(&per_model_predictions)
        .map_err(to_py_err)
}

#[pyfunction(name = "bagging_ensemble_variance")]
fn ens_bagging_ensemble_variance(
    single_estimator_variance: f64,
    average_correlation: f64,
    n_estimators: usize,
) -> PyResult<f64> {
    openquant::ensemble_methods::bagging_ensemble_variance(
        single_estimator_variance,
        average_correlation,
        n_estimators,
    )
    .map_err(to_py_err)
}

#[pyfunction(name = "recommend_bagging_vs_boosting")]
fn ens_recommend_bagging_vs_boosting(
    py: Python<'_>,
    base_estimator_accuracy: f64,
    average_prediction_correlation: f64,
    label_redundancy: f64,
    single_estimator_variance: f64,
    n_estimators: usize,
) -> PyResult<PyObject> {
    let result = openquant::ensemble_methods::recommend_bagging_vs_boosting(
        base_estimator_accuracy,
        average_prediction_correlation,
        label_redundancy,
        single_estimator_variance,
        n_estimators,
    )
    .map_err(to_py_err)?;

    let d = PyDict::new(py);
    d.set_item(
        "recommended",
        match result.recommended {
            openquant::ensemble_methods::EnsembleMethod::Bagging => "bagging",
            openquant::ensemble_methods::EnsembleMethod::Boosting => "boosting",
        },
    )?;
    d.set_item("expected_bagging_variance", result.expected_bagging_variance)?;
    d.set_item("expected_variance_reduction", result.expected_variance_reduction)?;
    Ok(d.into_pyobject(py).unwrap().into_any().unbind())
}

pub fn register(py: Python<'_>, parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(py, "ensemble")?;
    m.add_function(wrap_pyfunction!(ens_bias_variance_noise, &m)?)?;
    m.add_function(wrap_pyfunction!(ens_bootstrap_sample_indices, &m)?)?;
    m.add_function(wrap_pyfunction!(ens_sequential_bootstrap_sample_indices, &m)?)?;
    m.add_function(wrap_pyfunction!(ens_aggregate_regression_mean, &m)?)?;
    m.add_function(wrap_pyfunction!(ens_aggregate_classification_vote, &m)?)?;
    m.add_function(wrap_pyfunction!(ens_aggregate_classification_probability_mean, &m)?)?;
    m.add_function(wrap_pyfunction!(ens_average_pairwise_prediction_correlation, &m)?)?;
    m.add_function(wrap_pyfunction!(ens_bagging_ensemble_variance, &m)?)?;
    m.add_function(wrap_pyfunction!(ens_recommend_bagging_vs_boosting, &m)?)?;
    parent.add_submodule(&m)?;
    parent.add("ensemble", m)?;
    Ok(())
}

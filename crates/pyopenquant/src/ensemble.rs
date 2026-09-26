use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::helpers::to_py_err;

/// Bias-variance-noise decomposition of an ensemble's forecasts (AFML section 6.2).
///
/// All terms are averaged over observations. `variance` is the population variance of the
/// models' predictions around their mean, per observation, and `mse` is the error of the
/// individual models against `y_true`, averaged over models and observations. Without
/// `y_expected`, `bias_sq = mean((mean_pred - y_true)**2)` includes the irreducible noise,
/// `noise` is None and `bias_sq + variance == mse` exactly. With `y_expected` (the noiseless
/// target E[y|x], known in simulation studies), the bias is measured against it and
/// `noise = mean((y_true - y_expected)**2)`, so the three terms add up to `mse` in expectation.
///
/// Parameters
/// ----------
/// y_true : list[float]
///     Observed targets, one per observation.
/// per_model_predictions : list[list[float]]
///     One row per model, each as long as `y_true`.
/// y_expected : list[float] | None, default None
///     Noiseless target E[y|x] per observation, as long as `y_true`.
///
/// Returns
/// -------
/// tuple[float, float, float | None, float]
///     `(bias_sq, variance, noise, mse)`; `noise` is None unless `y_expected` is given.
///
/// Raises
/// ------
/// ValueError
///     If `y_true` or `per_model_predictions` is empty, or a model row or `y_expected` differs
///     in length from `y_true`.
#[pyfunction(name = "bias_variance_noise")]
#[pyo3(signature = (y_true, per_model_predictions, y_expected=None))]
fn ens_bias_variance_noise(
    y_true: Vec<f64>,
    per_model_predictions: Vec<Vec<f64>>,
    y_expected: Option<Vec<f64>>,
) -> PyResult<(f64, f64, Option<f64>, f64)> {
    let result = openquant::ensemble_methods::bias_variance_noise(
        &y_true,
        &per_model_predictions,
        y_expected.as_deref(),
    )
    .map_err(to_py_err)?;
    Ok((result.bias_sq, result.variance, result.noise, result.mse))
}

/// Draw a uniform bootstrap sample of indices, with replacement (AFML section 6.3).
///
/// This is the standard bagging bootstrap; it ignores label overlap (see
/// `sequential_bootstrap_sample_indices` for the uniqueness-aware alternative). The draw is
/// reproducible for a given `seed`.
///
/// Parameters
/// ----------
/// n_samples : int
///     Size of the population; indices are drawn from `range(n_samples)`.
/// sample_size : int
///     Number of indices to draw.
/// seed : int
///     Seed of the random generator.
///
/// Returns
/// -------
/// list[int]
///     `sample_size` indices in `[0, n_samples)`, in draw order, possibly repeated.
///
/// Raises
/// ------
/// ValueError
///     If `n_samples` or `sample_size` is 0.
#[pyfunction(name = "bootstrap_sample_indices")]
fn ens_bootstrap_sample_indices(
    n_samples: usize,
    sample_size: usize,
    seed: u64,
) -> PyResult<Vec<usize>> {
    openquant::ensemble_methods::bootstrap_sample_indices(n_samples, sample_size, seed)
        .map_err(to_py_err)
}

/// Draw label indices with the sequential bootstrap (AFML section 4.5.1, Snippets 4.5-4.6).
///
/// Labels are drawn one at a time, with replacement, each with probability proportional to its
/// average uniqueness given the draws so far, so overlapping labels are drawn less often than
/// under a uniform bootstrap. The draw is reproducible for a given `seed` (but differs from
/// `bootstrap_sample_indices` with the same seed).
///
/// Parameters
/// ----------
/// ind_mat : list[list[int]]
///     Bars x labels indicator matrix (e.g. from `openquant.sampling`): one row per bar, one
///     column per label, 1 where the label spans the bar and 0 elsewhere.
/// sample_size : int
///     Number of labels to draw; must be positive.
/// seed : int
///     Seed of the random generator.
///
/// Returns
/// -------
/// list[int]
///     `sample_size` column (label) indices of `ind_mat`, in draw order, possibly repeated.
///
/// Raises
/// ------
/// ValueError
///     If `sample_size` is 0, `ind_mat` has no rows, its first row is empty, or its rows
///     differ in length.
#[pyfunction(name = "sequential_bootstrap_sample_indices")]
fn ens_sequential_bootstrap_sample_indices(
    ind_mat: Vec<Vec<u8>>,
    sample_size: usize,
    seed: u64,
) -> PyResult<Vec<usize>> {
    openquant::ensemble_methods::sequential_bootstrap_sample_indices(&ind_mat, sample_size, seed)
        .map_err(to_py_err)
}

/// Element-wise mean of the models' predictions: the bagged regression forecast (AFML 6.3).
///
/// Parameters
/// ----------
/// per_model_predictions : list[list[float]]
///     One row per model, one column per observation; every row the same length.
///
/// Returns
/// -------
/// list[float]
///     The mean prediction per observation.
///
/// Raises
/// ------
/// ValueError
///     If there are no model rows, the first row is empty, or the rows differ in length.
#[pyfunction(name = "aggregate_regression_mean")]
fn ens_aggregate_regression_mean(per_model_predictions: Vec<Vec<f64>>) -> PyResult<Vec<f64>> {
    openquant::ensemble_methods::aggregate_regression_mean(&per_model_predictions)
        .map_err(to_py_err)
}

/// Majority vote over binary class predictions (AFML section 6.3.2).
///
/// An observation is labelled 1 when at least half of the models vote 1, so a tie goes to 1.
/// The vote discards confidence; `aggregate_classification_probability_mean` keeps it.
///
/// Parameters
/// ----------
/// per_model_predictions : list[list[int]]
///     One row per model, one 0/1 label per observation; every row the same length.
///
/// Returns
/// -------
/// list[int]
///     The voted 0/1 label per observation.
///
/// Raises
/// ------
/// ValueError
///     If there are no model rows, the first row is empty, the rows differ in length, or a
///     label is not 0 or 1.
#[pyfunction(name = "aggregate_classification_vote")]
fn ens_aggregate_classification_vote(per_model_predictions: Vec<Vec<u8>>) -> PyResult<Vec<u32>> {
    // Widened so the labels reach Python as a list of ints; PyO3 turns a Vec<u8> into `bytes`.
    let votes = openquant::ensemble_methods::aggregate_classification_vote(&per_model_predictions)
        .map_err(to_py_err)?;
    Ok(votes.into_iter().map(u32::from).collect())
}

/// Mean predicted probability per observation, and the label it implies at `threshold`.
///
/// The averaged probability is what bet sizing (AFML Chapter 10) takes as input. Every input
/// probability must lie in `[0, 1]`.
///
/// Parameters
/// ----------
/// per_model_probabilities : list[list[float]]
///     One row per model, one probability of class 1 per observation; every row the same
///     length.
/// threshold : float
///     Cut-off in `[0, 1]`; a label is 1 when the mean probability is `>= threshold`.
///
/// Returns
/// -------
/// tuple[list[float], list[int]]
///     `(probabilities, labels)`: the mean probability and the 0/1 label per observation.
///
/// Raises
/// ------
/// ValueError
///     If there are no model rows, the first row is empty, the rows differ in length,
///     `threshold` is outside `[0, 1]` or NaN, or any input probability is outside `[0, 1]`
///     or NaN.
#[pyfunction(name = "aggregate_classification_probability_mean")]
fn ens_aggregate_classification_probability_mean(
    per_model_probabilities: Vec<Vec<f64>>,
    threshold: f64,
) -> PyResult<(Vec<f64>, Vec<u32>)> {
    let (probabilities, labels) =
        openquant::ensemble_methods::aggregate_classification_probability_mean(
            &per_model_probabilities,
            threshold,
        )
        .map_err(to_py_err)?;
    Ok((probabilities, labels.into_iter().map(u32::from).collect()))
}

/// Mean Pearson correlation over all pairs of model rows: the rho-bar of AFML section 6.3.1.
///
/// Pass the models' errors (residuals) to get the rho-bar that `bagging_ensemble_variance`
/// expects; on raw predictions of a common target the correlation mostly measures that every
/// model tracks the target. A pair involving a constant row contributes a correlation of 0.
///
/// Parameters
/// ----------
/// per_model_predictions : list[list[float]]
///     One row per model, one column per observation; every row the same length.
///
/// Returns
/// -------
/// float
///     The average of the pairwise correlations.
///
/// Raises
/// ------
/// ValueError
///     If there are fewer than two model rows, the first row has fewer than two entries, or the
///     rows differ in length.
#[pyfunction(name = "average_pairwise_prediction_correlation")]
fn ens_average_pairwise_prediction_correlation(
    per_model_predictions: Vec<Vec<f64>>,
) -> PyResult<f64> {
    openquant::ensemble_methods::average_pairwise_prediction_correlation(&per_model_predictions)
        .map_err(to_py_err)
}

/// Variance of the average of N estimators, `sigma2 * (rho + (1 - rho) / N)` (AFML 6.3.1).
///
/// As N grows the result falls to the floor `sigma2 * rho`, which no number of estimators
/// moves: redundant, correlated estimators limit what bagging can achieve.
///
/// Parameters
/// ----------
/// single_estimator_variance : float
///     Common variance `sigma2` of one estimator's predictions; must be finite and
///     non-negative.
/// average_correlation : float
///     Average pairwise correlation `rho` between estimators, in `[-1, 1]`, e.g. from
///     `average_pairwise_prediction_correlation`.
/// n_estimators : int
///     Number of estimators N; must be positive.
///
/// Returns
/// -------
/// float
///     The variance of the ensemble average.
///
/// Raises
/// ------
/// ValueError
///     If `single_estimator_variance` is negative, NaN or infinite, `average_correlation` is
///     outside `[-1, 1]`
///     or NaN, or `n_estimators` is 0.
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

/// Expected bagging variance plus a rule-of-thumb bagging-or-boosting label (AFML 6.6).
///
/// The label is "boosting" if `base_estimator_accuracy < 0.55`, or
/// `average_prediction_correlation >= 0.75`, or `label_redundancy >= 0.70`, and "bagging"
/// otherwise. The reasoning follows AFML section 6.6 (bagging reduces variance, boosting
/// reduces bias, and bagging cannot rescue a learner barely better than chance), but the three
/// cut-offs are this library's, not the book's; AFML's own conclusion is that bagging is
/// generally preferable in finance. Treat the label as a prompt to look at the numbers.
///
/// Parameters
/// ----------
/// base_estimator_accuracy : float
///     Accuracy of a single base estimator, in `[0, 1]`.
/// average_prediction_correlation : float
///     Average pairwise correlation between estimators, in `[-1, 1]`.
/// label_redundancy : float
///     Label redundancy in `[0, 1]`, e.g. one minus the average uniqueness of the labels.
/// single_estimator_variance : float
///     Variance of one estimator's predictions; must be non-negative.
/// n_estimators : int
///     Number of estimators in the ensemble; must be positive.
///
/// Returns
/// -------
/// dict[str, Any]
///     A dict with keys `recommended` (str, "bagging" or "boosting"),
///     `expected_bagging_variance` (float, from `bagging_ensemble_variance`) and
///     `expected_variance_reduction` (float, `single_estimator_variance` minus the expected
///     bagging variance, floored at 0).
///
/// Raises
/// ------
/// ValueError
///     If `base_estimator_accuracy` or `label_redundancy` is outside `[0, 1]` or NaN, or if the
///     core rejects the variance inputs (negative `single_estimator_variance`,
///     `average_prediction_correlation` outside `[-1, 1]`, or `n_estimators == 0`).
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

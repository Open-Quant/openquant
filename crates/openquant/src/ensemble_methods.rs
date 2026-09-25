//! Ensemble-method diagnostics aligned to AFML Chapter 6.
//!
//! This module does not train ensembles. It answers the question that comes before training
//! one: *will averaging help here?* The central result is the variance of an average of `N`
//! estimators with common variance `σ̄²` and average pairwise correlation `ρ̄` (AFML §6.3.1):
//!
//! ```text
//! V[mean of N estimators] = σ̄² (ρ̄ + (1 − ρ̄) / N)
//! ```
//!
//! The second term vanishes as `N` grows; the first is a floor that no number of estimators
//! moves, which is why redundant (overlapping) labels hurt bagging in finance (§6.3.3).
//!
//! The module provides:
//! - [`bias_variance_noise`]: the error decomposition of §6.2.
//! - [`bootstrap_sample_indices`] and [`sequential_bootstrap_sample_indices`]: seeded uniform
//!   and sequential (AFML §4.5.1) bootstrap draws for bagging (§6.3).
//! - [`aggregate_regression_mean`], [`aggregate_classification_vote`] and
//!   [`aggregate_classification_probability_mean`]: combining the models' outputs.
//! - [`average_pairwise_prediction_correlation`] and [`bagging_ensemble_variance`]: `ρ̄` and
//!   the formula above.
//! - [`recommend_bagging_vs_boosting`]: the formula plus a rule-of-thumb label (§6.5–6.6).
//!   **Its cut-offs are this library's, not the book's.**
//!
//! Conventions:
//!
//! - Per-model inputs are `&[Vec<_>]` with **one row per model** and one column per
//!   observation; every row must have the same length.
//! - Classification labels are `u8` in `{0, 1}`; probabilities are in `[0, 1]`.
//! - Correlations are Pearson correlations of whatever rows are passed. For the `ρ̄` the
//!   formula means, pass the models' **errors** (residuals), not raw predictions of a common
//!   target, which mostly measure that every model tracks the target.
//! - The formula assumes equal variances and a single average correlation; clusters of
//!   near-identical models or a few strong models among weak ones break it.
//! - Randomised functions take an explicit `seed` and are reproducible for a given seed.
//!
//! ```
//! use openquant::ensemble_methods::{
//!     aggregate_classification_vote, bagging_ensemble_variance, recommend_bagging_vs_boosting,
//!     EnsembleError, EnsembleMethod,
//! };
//!
//! # fn main() -> Result<(), EnsembleError> {
//! // The floor: at rho = 0.9, a thousand estimators remove under a tenth of the variance.
//! let v = bagging_ensemble_variance(1.0, 0.9, 1_000)?;
//! assert!((v - 0.9001).abs() < 1e-12);
//! // Independent estimators: 1/N.
//! assert!((bagging_ensemble_variance(2.0, 0.0, 50)? - 0.04).abs() < 1e-12);
//!
//! // Two of three vote 1; a 1-1 tie also resolves to 1.
//! assert_eq!(aggregate_classification_vote(&[vec![1, 0], vec![1, 0], vec![0, 0]])?, vec![1, 0]);
//! assert_eq!(aggregate_classification_vote(&[vec![1], vec![0]])?, vec![1]);
//! assert_eq!(aggregate_classification_vote(&[vec![2]]), Err(EnsembleError::NonBinaryLabels));
//!
//! let decision = recommend_bagging_vs_boosting(0.62, 0.30, 0.40, 1.0, 25)?;
//! assert_eq!(decision.recommended, EnsembleMethod::Bagging);
//! assert!((decision.expected_bagging_variance - 0.328).abs() < 1e-12);
//! # Ok(())
//! # }
//! ```
#![deny(missing_docs)]

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use crate::sampling::seq_bootstrap_with_rng;

/// Errors returned by the ensemble-method functions.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum EnsembleError {
    /// A malformed indicator matrix passed through to the sequential bootstrap (for example,
    /// rows of different lengths).
    #[error(transparent)]
    Input(#[from] crate::util::InputError),
    /// The named input is empty.
    #[error("{0} cannot be empty")]
    Empty(&'static str),
    /// A row of the named input does not have the expected length.
    #[error("{0} length mismatch")]
    LengthMismatch(&'static str),
    /// [`bootstrap_sample_indices`] was given `n_samples == 0` or `sample_size == 0`.
    #[error("n_samples and sample_size must be > 0")]
    ZeroSampleCount,
    /// A scalar argument is outside its domain.
    #[error("{name} must be {requirement}")]
    Invalid {
        /// Name of the offending argument.
        name: &'static str,
        /// The condition it must satisfy.
        requirement: &'static str,
    },
    /// The indicator matrix's first row has no label columns.
    #[error("ind_mat must include at least one label column")]
    NoLabelColumns,
    /// The model rows have no observations.
    #[error("prediction rows cannot be empty")]
    EmptyPredictionRows,
    /// A classification vote received a label other than 0 or 1.
    #[error("classification vote expects binary labels in {{0,1}}")]
    NonBinaryLabels,
    /// A pairwise statistic needs at least two model rows.
    #[error("at least two model prediction rows are required")]
    TooFewModels,
    /// A correlation needs at least two observations per model row.
    #[error("prediction rows must have at least two samples")]
    TooFewPredictionSamples,
}

/// Ensemble method recommended by [`recommend_bagging_vs_boosting`] (AFML §6.6).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EnsembleMethod {
    /// Bootstrap aggregation: averages independently fitted estimators to reduce variance
    /// (§6.3).
    Bagging,
    /// Sequential reweighting of weak learners to reduce bias (§6.5).
    Boosting,
}

/// Error decomposition of an ensemble's forecasts (AFML §6.2), averaged over observations.
///
/// `mse` is always measured against the observed labels `y_true`. The other terms depend on
/// whether the noiseless target `y_expected` = E\[y|x\] was supplied to [`bias_variance_noise`]:
///
/// - **With `y_expected`:** `bias_sq` = mean((mean_pred − y_expected)²), `noise` =
///   `Some(mean((y_true − y_expected)²))`, and `bias_sq + variance + noise` equals `mse` in
///   expectation (exactly only when the label noise is uncorrelated with the predictions in
///   the sample at hand).
/// - **Without it:** `bias_sq` = mean((mean_pred − y_true)²), which *includes* the irreducible
///   noise, and `noise` is `None`. In that case `bias_sq + variance == mse` identically.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BiasVarianceNoise {
    /// Mean squared bias of the ensemble-mean prediction, against `y_expected` if given and
    /// against `y_true` (so including the noise) otherwise.
    pub bias_sq: f64,
    /// Mean over observations of the population variance of the models' predictions.
    pub variance: f64,
    /// Mean squared label noise `(y_true − y_expected)²`; `None` without `y_expected`.
    pub noise: Option<f64>,
    /// Mean squared error of the individual models against `y_true`, averaged over models and
    /// observations.
    pub mse: f64,
}

/// Output of [`recommend_bagging_vs_boosting`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BaggingBoostingDecision {
    /// Rule-of-thumb label; treat it as a prompt to look at the inputs, not a verdict.
    pub recommended: EnsembleMethod,
    /// `σ̄² (ρ̄ + (1 − ρ̄)/N)` from [`bagging_ensemble_variance`] (AFML §6.3.1).
    pub expected_bagging_variance: f64,
    /// `single_estimator_variance − expected_bagging_variance`, floored at zero.
    pub expected_variance_reduction: f64,
}

/// Bias², variance, noise and MSE of an ensemble's predictions, averaged over observations.
///
/// `per_model_predictions` holds one row per model, each as long as `y_true`. `variance` is
/// the population variance of the models' predictions around their mean, per observation.
///
/// Pass `y_expected` (the noiseless target E\[y|x\], known in simulation studies) to split
/// the irreducible noise out of the bias; see [`BiasVarianceNoise`] for the formulas. Without
/// it `noise` is `None`, because noise cannot be separated from bias using observed labels
/// alone.
///
/// ```
/// use openquant::ensemble_methods::bias_variance_noise;
///
/// # fn main() -> Result<(), openquant::ensemble_methods::EnsembleError> {
/// // Two models, two observations; the ensemble mean is [2, 2].
/// let preds = [vec![1.0, 3.0], vec![3.0, 1.0]];
/// let d = bias_variance_noise(&[1.0, 2.0], &preds, None)?;
/// assert_eq!((d.bias_sq, d.variance, d.noise, d.mse), (0.5, 1.0, None, 1.5));
///
/// // With the noiseless target the noise is split out of the bias.
/// let d = bias_variance_noise(&[1.0, 2.0], &preds, Some(&[1.5, 2.5]))?;
/// assert_eq!((d.bias_sq, d.variance, d.noise), (0.25, 1.0, Some(0.25)));
/// assert_eq!(d.bias_sq + d.variance + d.noise.unwrap(), d.mse);
/// # Ok(())
/// # }
/// ```
///
/// # Errors
///
/// - [`EnsembleError::Empty`] for an empty `y_true` or no model rows.
/// - [`EnsembleError::LengthMismatch`]`("prediction")` if a model row's length differs from
///   `y_true`; `LengthMismatch("y_expected")` if `y_expected`'s does.
pub fn bias_variance_noise(
    y_true: &[f64],
    per_model_predictions: &[Vec<f64>],
    y_expected: Option<&[f64]>,
) -> Result<BiasVarianceNoise, EnsembleError> {
    if y_true.is_empty() {
        return Err(EnsembleError::Empty("y_true"));
    }
    if per_model_predictions.is_empty() {
        return Err(EnsembleError::Empty("per_model_predictions"));
    }
    if per_model_predictions.iter().any(|row| row.len() != y_true.len()) {
        return Err(EnsembleError::LengthMismatch("prediction"));
    }
    if y_expected.is_some_and(|target| target.len() != y_true.len()) {
        return Err(EnsembleError::LengthMismatch("y_expected"));
    }

    let n_models = per_model_predictions.len() as f64;
    let n_samples = y_true.len() as f64;

    let mut bias_sq_sum = 0.0;
    let mut var_sum = 0.0;
    let mut mse_sum = 0.0;
    let mut noise_sum = 0.0;

    for i in 0..y_true.len() {
        let mut mean_pred = 0.0;
        for model in per_model_predictions {
            mean_pred += model[i];
            let err = model[i] - y_true[i];
            mse_sum += err * err;
        }
        mean_pred /= n_models;

        // Bias is measured against E[y|x] when it is known, else against the observed label.
        let reference = match y_expected {
            Some(target) => {
                let eps = y_true[i] - target[i];
                noise_sum += eps * eps;
                target[i]
            }
            None => y_true[i],
        };
        let bias = mean_pred - reference;
        bias_sq_sum += bias * bias;

        let mut local_var = 0.0;
        for model in per_model_predictions {
            let d = model[i] - mean_pred;
            local_var += d * d;
        }
        local_var /= n_models;
        var_sum += local_var;
    }

    let bias_sq = bias_sq_sum / n_samples;
    let variance = var_sum / n_samples;
    let mse = mse_sum / (n_samples * n_models);
    let noise = y_expected.map(|_| noise_sum / n_samples);

    Ok(BiasVarianceNoise { bias_sq, variance, noise, mse })
}

/// Draws `sample_size` indices uniformly with replacement from `0..n_samples` (the standard
/// bootstrap used by bagging, AFML §6.3).
///
/// The draw is reproducible for a given `seed`. It ignores label overlap; see
/// [`sequential_bootstrap_sample_indices`] for the uniqueness-aware alternative.
///
/// ```
/// use openquant::ensemble_methods::bootstrap_sample_indices;
///
/// # fn main() -> Result<(), openquant::ensemble_methods::EnsembleError> {
/// let draw = bootstrap_sample_indices(10, 25, 42)?;
/// assert_eq!(draw.len(), 25);
/// assert!(draw.iter().all(|&i| i < 10));
/// assert_eq!(draw, bootstrap_sample_indices(10, 25, 42)?);
/// # Ok(())
/// # }
/// ```
///
/// # Errors
///
/// [`EnsembleError::ZeroSampleCount`] if `n_samples` or `sample_size` is zero.
pub fn bootstrap_sample_indices(
    n_samples: usize,
    sample_size: usize,
    seed: u64,
) -> Result<Vec<usize>, EnsembleError> {
    if n_samples == 0 || sample_size == 0 {
        return Err(EnsembleError::ZeroSampleCount);
    }
    let mut rng = StdRng::seed_from_u64(seed);
    Ok((0..sample_size).map(|_| rng.gen_range(0..n_samples)).collect())
}

/// Draws `sample_size` label indices with the sequential bootstrap (AFML §4.5.1, Snippets
/// 4.5–4.6), seeded for reproducibility.
///
/// `ind_mat` is the bars × labels indicator matrix from
/// [`get_ind_matrix`](crate::sampling::get_ind_matrix): one row per bar, one column per
/// label, `1` where the label spans the bar. The returned values are **column** (label)
/// indices. This is [`seq_bootstrap_with_rng`] with
/// a [`StdRng`] seeded from `seed`; the same `seed` gives the same indices, but not the
/// indices [`bootstrap_sample_indices`] gives for that seed, because the two draw from
/// different distributions.
///
/// ```
/// use openquant::ensemble_methods::sequential_bootstrap_sample_indices;
///
/// # fn main() -> Result<(), openquant::ensemble_methods::EnsembleError> {
/// // 3 labels over 4 bars: label 0 spans bars 0-1, label 1 bars 1-2, label 2 bars 2-3.
/// let ind_mat = vec![vec![1, 0, 0], vec![1, 1, 0], vec![0, 1, 1], vec![0, 0, 1]];
/// let draw = sequential_bootstrap_sample_indices(&ind_mat, 6, 7)?;
/// assert_eq!(draw.len(), 6);
/// assert!(draw.iter().all(|&i| i < 3));
/// assert_eq!(draw, sequential_bootstrap_sample_indices(&ind_mat, 6, 7)?);
/// # Ok(())
/// # }
/// ```
///
/// # Errors
///
/// - [`EnsembleError::Invalid`] if `sample_size == 0`.
/// - [`EnsembleError::Empty`]`("ind_mat")` if `ind_mat` has no rows.
/// - [`EnsembleError::NoLabelColumns`] if the first row of `ind_mat` is empty.
/// - [`EnsembleError::Input`] if the rows of `ind_mat` differ in length.
pub fn sequential_bootstrap_sample_indices(
    ind_mat: &[Vec<u8>],
    sample_size: usize,
    seed: u64,
) -> Result<Vec<usize>, EnsembleError> {
    if sample_size == 0 {
        return Err(EnsembleError::Invalid { name: "sample_size", requirement: "> 0" });
    }
    if ind_mat.is_empty() {
        return Err(EnsembleError::Empty("ind_mat"));
    }
    let n_labels = ind_mat.first().map(|r| r.len()).unwrap_or(0);
    if n_labels == 0 {
        return Err(EnsembleError::NoLabelColumns);
    }

    let mut rng = StdRng::seed_from_u64(seed);
    Ok(seq_bootstrap_with_rng(ind_mat, Some(sample_size), None, &mut rng)?)
}

/// Element-wise mean of the models' predictions: the bagged regression forecast (AFML §6.3).
///
/// ```
/// use openquant::ensemble_methods::aggregate_regression_mean;
///
/// # fn main() -> Result<(), openquant::ensemble_methods::EnsembleError> {
/// assert_eq!(aggregate_regression_mean(&[vec![1.0, 2.0], vec![3.0, 4.0]])?, vec![2.0, 3.0]);
/// # Ok(())
/// # }
/// ```
///
/// # Errors
///
/// - [`EnsembleError::Empty`] if there are no model rows.
/// - [`EnsembleError::EmptyPredictionRows`] if the first row is empty.
/// - [`EnsembleError::LengthMismatch`]`("prediction")` if the rows differ in length.
pub fn aggregate_regression_mean(
    per_model_predictions: &[Vec<f64>],
) -> Result<Vec<f64>, EnsembleError> {
    if per_model_predictions.is_empty() {
        return Err(EnsembleError::Empty("per_model_predictions"));
    }
    let n = per_model_predictions[0].len();
    if n == 0 {
        return Err(EnsembleError::EmptyPredictionRows);
    }
    if per_model_predictions.iter().any(|row| row.len() != n) {
        return Err(EnsembleError::LengthMismatch("prediction"));
    }

    let mut out = vec![0.0; n];
    for row in per_model_predictions {
        for (i, v) in row.iter().enumerate() {
            out[i] += *v;
        }
    }
    let denom = per_model_predictions.len() as f64;
    for v in &mut out {
        *v /= denom;
    }
    Ok(out)
}

/// Majority vote over binary `{0, 1}` class predictions (AFML §6.3.2).
///
/// An observation is labelled 1 when at least half of the models vote 1, so **a tie goes to
/// 1**. Majority vote discards confidence; [`aggregate_classification_probability_mean`]
/// keeps it. See the module example.
///
/// # Errors
///
/// - [`EnsembleError::Empty`] if there are no model rows.
/// - [`EnsembleError::EmptyPredictionRows`] if the first row is empty.
/// - [`EnsembleError::LengthMismatch`]`("prediction")` if the rows differ in length.
/// - [`EnsembleError::NonBinaryLabels`] if any label is greater than 1.
pub fn aggregate_classification_vote(
    per_model_predictions: &[Vec<u8>],
) -> Result<Vec<u8>, EnsembleError> {
    if per_model_predictions.is_empty() {
        return Err(EnsembleError::Empty("per_model_predictions"));
    }
    let n = per_model_predictions[0].len();
    if n == 0 {
        return Err(EnsembleError::EmptyPredictionRows);
    }
    if per_model_predictions.iter().any(|row| row.len() != n) {
        return Err(EnsembleError::LengthMismatch("prediction"));
    }
    if per_model_predictions.iter().flat_map(|row| row.iter()).any(|label| *label > 1) {
        return Err(EnsembleError::NonBinaryLabels);
    }

    let mut out = vec![0u8; n];
    for i in 0..n {
        let votes = per_model_predictions.iter().map(|row| row[i] as usize).sum::<usize>();
        out[i] = if votes * 2 >= per_model_predictions.len() { 1 } else { 0 };
    }
    Ok(out)
}

/// Mean predicted probability per observation and the label it implies at `threshold`.
///
/// Returns `(probabilities, labels)`, where a label is 1 when the mean probability is
/// `>= threshold`. The averaged probability is what bet sizing (AFML Chapter 10) wants as
/// input. Only the *averaged* probabilities are range-checked: individual entries outside
/// `[0, 1]` are accepted as long as their mean lies inside it.
///
/// ```
/// use openquant::ensemble_methods::aggregate_classification_probability_mean;
///
/// # fn main() -> Result<(), openquant::ensemble_methods::EnsembleError> {
/// let (p, labels) =
///     aggregate_classification_probability_mean(&[vec![0.2, 0.9], vec![0.6, 0.5]], 0.5)?;
/// assert!((p[0] - 0.4).abs() < 1e-12 && (p[1] - 0.7).abs() < 1e-12);
/// assert_eq!(labels, vec![0, 1]);
/// # Ok(())
/// # }
/// ```
///
/// # Errors
///
/// - [`EnsembleError::Invalid`] if `threshold` is outside `[0, 1]` (or `NaN`), or if a mean
///   probability is outside `[0, 1]` (or `NaN`).
/// - Any error of [`aggregate_regression_mean`].
pub fn aggregate_classification_probability_mean(
    per_model_probabilities: &[Vec<f64>],
    threshold: f64,
) -> Result<(Vec<f64>, Vec<u8>), EnsembleError> {
    if !(0.0..=1.0).contains(&threshold) {
        return Err(EnsembleError::Invalid { name: "threshold", requirement: "in [0,1]" });
    }
    let probs = aggregate_regression_mean(per_model_probabilities)?;
    if probs.iter().any(|p| !(0.0..=1.0).contains(p)) {
        return Err(EnsembleError::Invalid { name: "probabilities", requirement: "in [0,1]" });
    }
    let labels = probs.iter().map(|p| if *p >= threshold { 1 } else { 0 }).collect();
    Ok((probs, labels))
}

/// Mean Pearson correlation over all pairs of model rows: the `ρ̄` of AFML §6.3.1.
///
/// Pass the models' **errors** (residuals), or predictions of a de-meaned target, to get the
/// `ρ̄` that [`bagging_ensemble_variance`] means; on raw predictions of a common target this
/// mostly measures that every model tracks the target. A pair involving a constant row
/// contributes a correlation of 0.
///
/// ```
/// use openquant::ensemble_methods::average_pairwise_prediction_correlation;
///
/// # fn main() -> Result<(), openquant::ensemble_methods::EnsembleError> {
/// // Pair correlations are +1, -1 and -1.
/// let rows = [vec![1.0, 2.0, 3.0], vec![2.0, 4.0, 6.0], vec![3.0, 2.0, 1.0]];
/// let rho = average_pairwise_prediction_correlation(&rows)?;
/// assert!((rho + 1.0 / 3.0).abs() < 1e-12);
/// # Ok(())
/// # }
/// ```
///
/// # Errors
///
/// - [`EnsembleError::TooFewModels`] if there are fewer than two rows.
/// - [`EnsembleError::TooFewPredictionSamples`] if the first row has fewer than two entries.
/// - [`EnsembleError::LengthMismatch`]`("prediction")` if the rows differ in length.
pub fn average_pairwise_prediction_correlation(
    per_model_predictions: &[Vec<f64>],
) -> Result<f64, EnsembleError> {
    if per_model_predictions.len() < 2 {
        return Err(EnsembleError::TooFewModels);
    }
    let n = per_model_predictions[0].len();
    if n < 2 {
        return Err(EnsembleError::TooFewPredictionSamples);
    }
    if per_model_predictions.iter().any(|row| row.len() != n) {
        return Err(EnsembleError::LengthMismatch("prediction"));
    }

    let mut corr_sum = 0.0;
    let mut pairs = 0usize;
    for i in 0..per_model_predictions.len() {
        for j in (i + 1)..per_model_predictions.len() {
            corr_sum += pearson_corr(&per_model_predictions[i], &per_model_predictions[j]);
            pairs += 1;
        }
    }
    Ok(corr_sum / pairs as f64)
}

/// Variance of the average of `n_estimators` estimators: `σ̄² (ρ̄ + (1 − ρ̄)/N)` (AFML
/// §6.3.1).
///
/// `single_estimator_variance` is `σ̄²`, the (common) variance of one estimator's
/// predictions; `average_correlation` is `ρ̄`, e.g. from
/// [`average_pairwise_prediction_correlation`]. As `N` grows the result falls to the floor
/// `σ̄² ρ̄`. See the module example.
///
/// # Errors
///
/// [`EnsembleError::Invalid`] if `single_estimator_variance` is negative, if
/// `average_correlation` is outside `[-1, 1]` (or `NaN`), or if `n_estimators == 0`. A `NaN`
/// variance is not rejected and yields `NaN`.
pub fn bagging_ensemble_variance(
    single_estimator_variance: f64,
    average_correlation: f64,
    n_estimators: usize,
) -> Result<f64, EnsembleError> {
    if single_estimator_variance < 0.0 {
        return Err(EnsembleError::Invalid {
            name: "single_estimator_variance",
            requirement: "non-negative",
        });
    }
    if !(-1.0..=1.0).contains(&average_correlation) {
        return Err(EnsembleError::Invalid {
            name: "average_correlation",
            requirement: "in [-1,1]",
        });
    }
    if n_estimators == 0 {
        return Err(EnsembleError::Invalid { name: "n_estimators", requirement: "> 0" });
    }

    let n = n_estimators as f64;
    let rho = average_correlation;
    Ok(single_estimator_variance * (rho + (1.0 - rho) / n))
}

/// Expected bagging variance plus a rule-of-thumb bagging-or-boosting label (AFML §6.6).
///
/// Returns [`EnsembleMethod::Boosting`] if `base_estimator_accuracy < 0.55`, *or*
/// `average_prediction_correlation >= 0.75`, *or* `label_redundancy >= 0.70`, and
/// [`EnsembleMethod::Bagging`] otherwise. The reasoning follows §6.6 (bagging addresses
/// variance and overfitting, boosting addresses bias, and bagging cannot rescue a learner
/// barely better than chance, §6.3.2), but **the three cut-offs are this library's, not the
/// book's**; AFML's own conclusion is that bagging is generally preferable in finance. The
/// numbers that carry information are
/// [`expected_bagging_variance`](BaggingBoostingDecision::expected_bagging_variance) and
/// [`expected_variance_reduction`](BaggingBoostingDecision::expected_variance_reduction).
///
/// Arguments: `base_estimator_accuracy` and `label_redundancy` (e.g. one minus average
/// uniqueness) are in `[0, 1]`; the other three are passed to [`bagging_ensemble_variance`].
/// See the module example.
///
/// # Errors
///
/// - [`EnsembleError::Invalid`] if `base_estimator_accuracy` or `label_redundancy` is outside
///   `[0, 1]` (or `NaN`).
/// - Any error of [`bagging_ensemble_variance`].
pub fn recommend_bagging_vs_boosting(
    base_estimator_accuracy: f64,
    average_prediction_correlation: f64,
    label_redundancy: f64,
    single_estimator_variance: f64,
    n_estimators: usize,
) -> Result<BaggingBoostingDecision, EnsembleError> {
    if !(0.0..=1.0).contains(&base_estimator_accuracy) {
        return Err(EnsembleError::Invalid {
            name: "base_estimator_accuracy",
            requirement: "in [0,1]",
        });
    }
    if !(0.0..=1.0).contains(&label_redundancy) {
        return Err(EnsembleError::Invalid { name: "label_redundancy", requirement: "in [0,1]" });
    }
    let bag_var = bagging_ensemble_variance(
        single_estimator_variance,
        average_prediction_correlation,
        n_estimators,
    )?;
    let expected_reduction = (single_estimator_variance - bag_var).max(0.0);

    // Heuristic criteria:
    // - weak learners (accuracy near random) favor boosting for bias reduction.
    // - highly correlated learners or high label redundancy reduce bagging gains.
    let weak_learner = base_estimator_accuracy < 0.55;
    let highly_correlated = average_prediction_correlation >= 0.75;
    let redundant_labels = label_redundancy >= 0.70;

    let recommended = if weak_learner || highly_correlated || redundant_labels {
        EnsembleMethod::Boosting
    } else {
        EnsembleMethod::Bagging
    };

    Ok(BaggingBoostingDecision {
        recommended,
        expected_bagging_variance: bag_var,
        expected_variance_reduction: expected_reduction,
    })
}

fn pearson_corr(x: &[f64], y: &[f64]) -> f64 {
    let mx = x.iter().sum::<f64>() / x.len() as f64;
    let my = y.iter().sum::<f64>() / y.len() as f64;

    let mut num = 0.0;
    let mut den_x = 0.0;
    let mut den_y = 0.0;
    for (a, b) in x.iter().zip(y.iter()) {
        let dx = *a - mx;
        let dy = *b - my;
        num += dx * dy;
        den_x += dx * dx;
        den_y += dy * dy;
    }
    if den_x == 0.0 || den_y == 0.0 {
        0.0
    } else {
        num / (den_x.sqrt() * den_y.sqrt())
    }
}

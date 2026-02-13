//! Ensemble-method utilities aligned to AFML Chapter 6.
//!
//! This module provides:
//! - Bias/variance/noise diagnostics for ensemble forecasts.
//! - Bagging mechanics (bootstrap + sequential-bootstrap wrappers).
//! - Aggregation helpers (majority vote and mean probability).
//! - Dependency-aware diagnostics for when bagging is likely to underperform.
//! - A practical bagging-vs-boosting recommendation heuristic.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use crate::sampling::seq_bootstrap;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EnsembleMethod {
    Bagging,
    Boosting,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BiasVarianceNoise {
    pub bias_sq: f64,
    pub variance: f64,
    pub noise: f64,
    pub mse: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BaggingBoostingDecision {
    pub recommended: EnsembleMethod,
    pub expected_bagging_variance: f64,
    pub expected_variance_reduction: f64,
}

pub fn bias_variance_noise(
    y_true: &[f64],
    per_model_predictions: &[Vec<f64>],
) -> Result<BiasVarianceNoise, String> {
    if y_true.is_empty() {
        return Err("y_true cannot be empty".to_string());
    }
    if per_model_predictions.is_empty() {
        return Err("per_model_predictions cannot be empty".to_string());
    }
    if per_model_predictions.iter().any(|row| row.len() != y_true.len()) {
        return Err("prediction length mismatch".to_string());
    }

    let n_models = per_model_predictions.len() as f64;
    let n_samples = y_true.len() as f64;

    let mut bias_sq_sum = 0.0;
    let mut var_sum = 0.0;
    let mut mse_sum = 0.0;

    for i in 0..y_true.len() {
        let mut mean_pred = 0.0;
        for model in per_model_predictions {
            mean_pred += model[i];
            let err = model[i] - y_true[i];
            mse_sum += err * err;
        }
        mean_pred /= n_models;

        let bias = mean_pred - y_true[i];
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
    let noise = (mse - bias_sq - variance).max(0.0);

    Ok(BiasVarianceNoise {
        bias_sq,
        variance,
        noise,
        mse,
    })
}

pub fn bootstrap_sample_indices(
    n_samples: usize,
    sample_size: usize,
    seed: u64,
) -> Result<Vec<usize>, String> {
    if n_samples == 0 || sample_size == 0 {
        return Err("n_samples and sample_size must be > 0".to_string());
    }
    let mut rng = StdRng::seed_from_u64(seed);
    Ok((0..sample_size).map(|_| rng.gen_range(0..n_samples)).collect())
}

pub fn sequential_bootstrap_sample_indices(
    ind_mat: &[Vec<u8>],
    sample_size: usize,
    seed: u64,
) -> Result<Vec<usize>, String> {
    if sample_size == 0 {
        return Err("sample_size must be > 0".to_string());
    }
    if ind_mat.is_empty() {
        return Err("ind_mat cannot be empty".to_string());
    }
    let n_labels = ind_mat.first().map(|r| r.len()).unwrap_or(0);
    if n_labels == 0 {
        return Err("ind_mat must include at least one label column".to_string());
    }

    let mut rng = StdRng::seed_from_u64(seed);
    let warmup: Vec<usize> = (0..sample_size).map(|_| rng.gen_range(0..n_labels)).collect();
    Ok(seq_bootstrap(ind_mat, Some(sample_size), Some(warmup)))
}

pub fn aggregate_regression_mean(per_model_predictions: &[Vec<f64>]) -> Result<Vec<f64>, String> {
    if per_model_predictions.is_empty() {
        return Err("per_model_predictions cannot be empty".to_string());
    }
    let n = per_model_predictions[0].len();
    if n == 0 {
        return Err("prediction rows cannot be empty".to_string());
    }
    if per_model_predictions.iter().any(|row| row.len() != n) {
        return Err("prediction length mismatch".to_string());
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

pub fn aggregate_classification_vote(per_model_predictions: &[Vec<u8>]) -> Result<Vec<u8>, String> {
    if per_model_predictions.is_empty() {
        return Err("per_model_predictions cannot be empty".to_string());
    }
    let n = per_model_predictions[0].len();
    if n == 0 {
        return Err("prediction rows cannot be empty".to_string());
    }
    if per_model_predictions.iter().any(|row| row.len() != n) {
        return Err("prediction length mismatch".to_string());
    }
    if per_model_predictions
        .iter()
        .flat_map(|row| row.iter())
        .any(|label| *label > 1)
    {
        return Err("classification vote expects binary labels in {0,1}".to_string());
    }

    let mut out = vec![0u8; n];
    for i in 0..n {
        let votes = per_model_predictions.iter().map(|row| row[i] as usize).sum::<usize>();
        out[i] = if votes * 2 >= per_model_predictions.len() {
            1
        } else {
            0
        };
    }
    Ok(out)
}

pub fn aggregate_classification_probability_mean(
    per_model_probabilities: &[Vec<f64>],
    threshold: f64,
) -> Result<(Vec<f64>, Vec<u8>), String> {
    if !(0.0..=1.0).contains(&threshold) {
        return Err("threshold must be in [0,1]".to_string());
    }
    let probs = aggregate_regression_mean(per_model_probabilities)?;
    if probs.iter().any(|p| !(0.0..=1.0).contains(p)) {
        return Err("probabilities must be in [0,1]".to_string());
    }
    let labels = probs.iter().map(|p| if *p >= threshold { 1 } else { 0 }).collect();
    Ok((probs, labels))
}

pub fn average_pairwise_prediction_correlation(per_model_predictions: &[Vec<f64>]) -> Result<f64, String> {
    if per_model_predictions.len() < 2 {
        return Err("at least two model prediction rows are required".to_string());
    }
    let n = per_model_predictions[0].len();
    if n < 2 {
        return Err("prediction rows must have at least two samples".to_string());
    }
    if per_model_predictions.iter().any(|row| row.len() != n) {
        return Err("prediction length mismatch".to_string());
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

pub fn bagging_ensemble_variance(
    single_estimator_variance: f64,
    average_correlation: f64,
    n_estimators: usize,
) -> Result<f64, String> {
    if single_estimator_variance < 0.0 {
        return Err("single_estimator_variance must be non-negative".to_string());
    }
    if !(-1.0..=1.0).contains(&average_correlation) {
        return Err("average_correlation must be in [-1,1]".to_string());
    }
    if n_estimators == 0 {
        return Err("n_estimators must be > 0".to_string());
    }

    let n = n_estimators as f64;
    let rho = average_correlation;
    Ok(single_estimator_variance * (rho + (1.0 - rho) / n))
}

pub fn recommend_bagging_vs_boosting(
    base_estimator_accuracy: f64,
    average_prediction_correlation: f64,
    label_redundancy: f64,
    single_estimator_variance: f64,
    n_estimators: usize,
) -> Result<BaggingBoostingDecision, String> {
    if !(0.0..=1.0).contains(&base_estimator_accuracy) {
        return Err("base_estimator_accuracy must be in [0,1]".to_string());
    }
    if !(0.0..=1.0).contains(&label_redundancy) {
        return Err("label_redundancy must be in [0,1]".to_string());
    }
    let bag_var =
        bagging_ensemble_variance(single_estimator_variance, average_prediction_correlation, n_estimators)?;
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

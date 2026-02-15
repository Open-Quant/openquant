use chrono::NaiveDateTime;
use nalgebra::DMatrix;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use crate::cross_validation::{ml_cross_val_score, PurgedKFold, Scoring, SimpleClassifier};
use crate::sampling::{get_ind_mat_average_uniqueness, seq_bootstrap};
use crate::sb_bagging::{
    MaxFeatures, MaxSamples, SbBaggingError, SequentiallyBootstrappedBaggingClassifier,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelSelectionMetric {
    CvAccuracy,
    CvNegLogLoss,
    OobScore,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PurgedCvConfig {
    pub n_splits: usize,
    pub pct_embargo: f64,
}

impl Default for PurgedCvConfig {
    fn default() -> Self {
        Self { n_splits: 5, pct_embargo: 0.01 }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct PrimaryModelConfig {
    pub cv: PurgedCvConfig,
}

impl Default for PrimaryModelConfig {
    fn default() -> Self {
        Self { cv: PurgedCvConfig::default() }
    }
}

#[derive(Debug, Clone)]
pub struct BaggedModelConfig {
    pub n_estimators: usize,
    pub max_samples: MaxSamples,
    pub max_features: MaxFeatures,
    pub bootstrap_features: bool,
    pub random_state: u64,
    pub cv: PurgedCvConfig,
    pub overlap_risk: bool,
    pub selection_metric: ModelSelectionMetric,
    pub uniqueness_trials: usize,
}

impl Default for BaggedModelConfig {
    fn default() -> Self {
        Self {
            n_estimators: 64,
            max_samples: MaxSamples::Float(1.0),
            max_features: MaxFeatures::Float(1.0),
            bootstrap_features: false,
            random_state: 42,
            cv: PurgedCvConfig::default(),
            overlap_risk: true,
            selection_metric: ModelSelectionMetric::CvAccuracy,
            uniqueness_trials: 64,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct CvMetrics {
    pub accuracy_scores: Vec<f64>,
    pub neg_log_loss_scores: Vec<f64>,
    pub accuracy_mean: f64,
    pub neg_log_loss_mean: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BootstrapDiversitySummary {
    pub vanilla_uniqueness_mean: f64,
    pub sequential_uniqueness_mean: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PrimaryModel {
    pub feature_idx: usize,
    pub threshold: f64,
    pub positive_on_ge: bool,
    pub scale: f64,
    pub cv_metrics: CvMetrics,
}

impl PrimaryModel {
    pub fn predict_proba(&self, x: &[Vec<f64>]) -> Result<Vec<f64>, ModelingLabError> {
        validate_feature_matrix(x)?;
        if x[0].len() <= self.feature_idx {
            return Err(ModelingLabError::DimensionMismatch(
                "feature count mismatch for primary model".to_string(),
            ));
        }

        Ok(x.iter()
            .map(|row| {
                let signed = if self.positive_on_ge {
                    row[self.feature_idx] - self.threshold
                } else {
                    self.threshold - row[self.feature_idx]
                };
                sigmoid(signed / self.scale)
            })
            .collect())
    }

    pub fn predict(&self, x: &[Vec<f64>]) -> Result<Vec<u8>, ModelingLabError> {
        let probs = self.predict_proba(x)?;
        Ok(probs.into_iter().map(|p| if p >= 0.5 { 1 } else { 0 }).collect())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct MetaModel {
    pub model: PrimaryModel,
    pub primary_probability_feature_index: usize,
}

impl MetaModel {
    pub fn predict_proba(
        &self,
        x: &[Vec<f64>],
        primary_probabilities: &[f64],
    ) -> Result<Vec<f64>, ModelingLabError> {
        let augmented = augment_with_primary_probabilities(x, primary_probabilities)?;
        self.model.predict_proba(&augmented)
    }

    pub fn predict(
        &self,
        x: &[Vec<f64>],
        primary_probabilities: &[f64],
    ) -> Result<Vec<u8>, ModelingLabError> {
        let augmented = augment_with_primary_probabilities(x, primary_probabilities)?;
        self.model.predict(&augmented)
    }
}

#[derive(Debug, Clone)]
pub struct BaggedModelResult {
    pub model: SequentiallyBootstrappedBaggingClassifier,
    pub cv_metrics: CvMetrics,
    pub selected_score: f64,
    pub resolved_max_samples: usize,
    pub average_uniqueness: f64,
    pub diversity: BootstrapDiversitySummary,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ModelingLabError {
    EmptyInput,
    DimensionMismatch(String),
    InvalidLabel(String),
    InvalidConfig(String),
    CvRequiredUnderOverlapRisk,
    CrossValidation(String),
    Bagging(SbBaggingError),
}

impl From<SbBaggingError> for ModelingLabError {
    fn from(value: SbBaggingError) -> Self {
        Self::Bagging(value)
    }
}

pub fn fit_primary_model(
    x: &[Vec<f64>],
    y: &[u8],
    samples_info_sets: &[(NaiveDateTime, NaiveDateTime)],
    config: &PrimaryModelConfig,
) -> Result<PrimaryModel, ModelingLabError> {
    validate_xy_info_sets(x, y, samples_info_sets)?;

    let splits = build_purged_cv_splits(samples_info_sets, &config.cv)?;
    let cv_metrics = threshold_cv_metrics(x, y, &splits)?;
    let model = fit_threshold_classifier(x, y, None)?;

    Ok(PrimaryModel {
        feature_idx: model.feature_idx,
        threshold: model.threshold,
        positive_on_ge: model.positive_on_ge,
        scale: model.scale,
        cv_metrics,
    })
}

pub fn fit_meta_model(
    x: &[Vec<f64>],
    primary_probabilities: &[f64],
    y: &[u8],
    samples_info_sets: &[(NaiveDateTime, NaiveDateTime)],
    config: &PrimaryModelConfig,
) -> Result<MetaModel, ModelingLabError> {
    let augmented = augment_with_primary_probabilities(x, primary_probabilities)?;
    let model = fit_primary_model(&augmented, y, samples_info_sets, config)?;

    Ok(MetaModel {
        model,
        primary_probability_feature_index: augmented
            .first()
            .map(|row| row.len().saturating_sub(1))
            .unwrap_or(0),
    })
}

pub fn fit_bagged_model(
    x: &DMatrix<f64>,
    y: &[u8],
    samples_info_sets: &[(NaiveDateTime, NaiveDateTime)],
    ind_mat: &[Vec<u8>],
    config: &BaggedModelConfig,
) -> Result<BaggedModelResult, ModelingLabError> {
    validate_bagged_inputs(x, y, samples_info_sets, ind_mat)?;
    if config.uniqueness_trials == 0 {
        return Err(ModelingLabError::InvalidConfig("uniqueness_trials must be > 0".to_string()));
    }
    if config.overlap_risk && matches!(config.selection_metric, ModelSelectionMetric::OobScore) {
        return Err(ModelingLabError::CvRequiredUnderOverlapRisk);
    }

    let base_max_samples = resolve_max_samples(config.max_samples, x.nrows())?;
    let average_uniqueness = get_ind_mat_average_uniqueness(ind_mat);
    let resolved_max_samples = uniqueness_capped_max_samples(base_max_samples, average_uniqueness);

    let mut model = SequentiallyBootstrappedBaggingClassifier::new(config.random_state);
    model.n_estimators = config.n_estimators;
    model.max_samples = MaxSamples::Int(resolved_max_samples);
    model.max_features = config.max_features;
    model.bootstrap_features = config.bootstrap_features;
    model.oob_score = true;
    model.fit(x, y, ind_mat, None)?;

    let splits = build_purged_cv_splits(samples_info_sets, &config.cv)?;
    let mut acc_scores = Vec::with_capacity(splits.len());
    let mut nll_scores = Vec::with_capacity(splits.len());

    for (train_idx, test_idx) in &splits {
        let x_train = select_matrix_rows(x, train_idx);
        let y_train = select_labels(y, train_idx);
        let train_ind = subset_indicator_matrix(ind_mat, train_idx)?;

        let train_base_max = resolve_max_samples(config.max_samples, x_train.nrows())?;
        let train_avg_uniqueness = get_ind_mat_average_uniqueness(&train_ind);
        let train_resolved_max =
            uniqueness_capped_max_samples(train_base_max, train_avg_uniqueness);

        let mut fold_model = SequentiallyBootstrappedBaggingClassifier::new(config.random_state);
        fold_model.n_estimators = config.n_estimators;
        fold_model.max_samples = MaxSamples::Int(train_resolved_max);
        fold_model.max_features = config.max_features;
        fold_model.bootstrap_features = config.bootstrap_features;
        fold_model.oob_score = false;
        fold_model.fit(&x_train, &y_train, &train_ind, None)?;

        let x_test = select_matrix_rows(x, test_idx);
        let y_test = select_labels(y, test_idx);
        let probs = fold_model.predict_proba(&x_test)?;
        let preds: Vec<u8> = probs.iter().map(|p| if *p >= 0.5 { 1 } else { 0 }).collect();

        let acc = preds.iter().zip(y_test.iter()).filter(|(p, t)| **p == **t).count() as f64
            / y_test.len() as f64;
        let nll = neg_log_loss_u8(&probs, &y_test);

        acc_scores.push(acc);
        nll_scores.push(nll);
    }

    let cv_metrics = CvMetrics {
        accuracy_mean: mean(&acc_scores),
        neg_log_loss_mean: mean(&nll_scores),
        accuracy_scores: acc_scores,
        neg_log_loss_scores: nll_scores,
    };

    let diversity = bootstrap_diversity_benchmark(
        ind_mat,
        resolved_max_samples,
        config.uniqueness_trials,
        config.random_state,
    )?;

    let selected_score = match config.selection_metric {
        ModelSelectionMetric::CvAccuracy => cv_metrics.accuracy_mean,
        ModelSelectionMetric::CvNegLogLoss => -cv_metrics.neg_log_loss_mean,
        ModelSelectionMetric::OobScore => model.oob_score_value.ok_or_else(|| {
            ModelingLabError::InvalidConfig("oob score was not computed".to_string())
        })?,
    };

    Ok(BaggedModelResult {
        model,
        cv_metrics,
        selected_score,
        resolved_max_samples,
        average_uniqueness,
        diversity,
    })
}

#[derive(Debug, Clone)]
struct ThresholdClassifier {
    feature_idx: usize,
    threshold: f64,
    positive_on_ge: bool,
    scale: f64,
}

impl SimpleClassifier for ThresholdClassifier {
    fn fit(&mut self, x: &[Vec<f64>], y: &[f64], sample_weight: Option<&[f64]>) {
        let y_u8: Vec<u8> = y.iter().map(|v| if *v >= 0.5 { 1 } else { 0 }).collect();
        let fitted = fit_threshold_classifier(x, &y_u8, sample_weight)
            .expect("threshold classifier fit should not fail for validated inputs");
        self.feature_idx = fitted.feature_idx;
        self.threshold = fitted.threshold;
        self.positive_on_ge = fitted.positive_on_ge;
        self.scale = fitted.scale;
    }

    fn predict_proba(&self, x: &[Vec<f64>]) -> Vec<f64> {
        x.iter()
            .map(|row| {
                let signed = if self.positive_on_ge {
                    row[self.feature_idx] - self.threshold
                } else {
                    self.threshold - row[self.feature_idx]
                };
                sigmoid(signed / self.scale)
            })
            .collect()
    }
}

fn threshold_cv_metrics(
    x: &[Vec<f64>],
    y: &[u8],
    splits: &[(Vec<usize>, Vec<usize>)],
) -> Result<CvMetrics, ModelingLabError> {
    let mut clf =
        ThresholdClassifier { feature_idx: 0, threshold: 0.0, positive_on_ge: true, scale: 1.0 };

    let y_f64: Vec<f64> = y.iter().map(|v| *v as f64).collect();
    let acc_scores = ml_cross_val_score(&mut clf, x, &y_f64, None, splits, Scoring::Accuracy);
    let nll_scores: Vec<f64> =
        ml_cross_val_score(&mut clf, x, &y_f64, None, splits, Scoring::NegLogLoss)
            .into_iter()
            .map(|s| -s)
            .collect();

    Ok(CvMetrics {
        accuracy_mean: mean(&acc_scores),
        neg_log_loss_mean: mean(&nll_scores),
        accuracy_scores: acc_scores,
        neg_log_loss_scores: nll_scores,
    })
}

fn fit_threshold_classifier(
    x: &[Vec<f64>],
    y: &[u8],
    sample_weight: Option<&[f64]>,
) -> Result<ThresholdClassifier, ModelingLabError> {
    validate_feature_matrix(x)?;
    if y.len() != x.len() {
        return Err(ModelingLabError::DimensionMismatch("y length must equal x rows".to_string()));
    }

    let n_features = x[0].len();
    let mut best = None;

    for feature_idx in 0..n_features {
        for i in 0..x.len() {
            let thr = x[i][feature_idx];
            for positive_on_ge in [true, false] {
                let mut weighted_correct = 0.0;
                let mut total_weight = 0.0;
                for row_idx in 0..x.len() {
                    let pred = if positive_on_ge {
                        (x[row_idx][feature_idx] >= thr) as u8
                    } else {
                        (x[row_idx][feature_idx] < thr) as u8
                    };
                    let w = sample_weight.map(|sw| sw[row_idx]).unwrap_or(1.0);
                    total_weight += w;
                    if pred == y[row_idx] {
                        weighted_correct += w;
                    }
                }
                let score = if total_weight > 0.0 { weighted_correct / total_weight } else { 0.0 };

                if best.map(|(_, _, _, s)| score > s).unwrap_or(true) {
                    best = Some((feature_idx, thr, positive_on_ge, score));
                }
            }
        }
    }

    let (feature_idx, threshold, positive_on_ge, _) = best.ok_or(ModelingLabError::EmptyInput)?;
    let scale = feature_stddev(x, feature_idx).max(1e-6);

    Ok(ThresholdClassifier { feature_idx, threshold, positive_on_ge, scale })
}

fn feature_stddev(x: &[Vec<f64>], feature_idx: usize) -> f64 {
    let mean = x.iter().map(|row| row[feature_idx]).sum::<f64>() / x.len() as f64;
    let var = x
        .iter()
        .map(|row| {
            let d = row[feature_idx] - mean;
            d * d
        })
        .sum::<f64>()
        / x.len() as f64;
    var.sqrt()
}

fn build_purged_cv_splits(
    samples_info_sets: &[(NaiveDateTime, NaiveDateTime)],
    config: &PurgedCvConfig,
) -> Result<Vec<(Vec<usize>, Vec<usize>)>, ModelingLabError> {
    let cv = PurgedKFold::new(config.n_splits, samples_info_sets.to_vec(), config.pct_embargo)
        .map_err(ModelingLabError::CrossValidation)?;
    cv.split(samples_info_sets.len()).map_err(ModelingLabError::CrossValidation)
}

fn augment_with_primary_probabilities(
    x: &[Vec<f64>],
    primary_probabilities: &[f64],
) -> Result<Vec<Vec<f64>>, ModelingLabError> {
    validate_feature_matrix(x)?;
    if x.len() != primary_probabilities.len() {
        return Err(ModelingLabError::DimensionMismatch(
            "primary_probabilities length must match x rows".to_string(),
        ));
    }
    if primary_probabilities.iter().any(|p| !(0.0..=1.0).contains(p)) {
        return Err(ModelingLabError::InvalidLabel(
            "primary probabilities must be in [0,1]".to_string(),
        ));
    }

    let mut out = Vec::with_capacity(x.len());
    for (row, p) in x.iter().zip(primary_probabilities.iter()) {
        let mut augmented = row.clone();
        augmented.push(*p);
        out.push(augmented);
    }
    Ok(out)
}

fn resolve_max_samples(
    max_samples: MaxSamples,
    n_samples: usize,
) -> Result<usize, ModelingLabError> {
    let resolved = match max_samples {
        MaxSamples::Int(v) => v,
        MaxSamples::Float(v) => {
            if v <= 0.0 {
                return Err(ModelingLabError::InvalidConfig(
                    "max_samples float must be > 0".to_string(),
                ));
            }
            (v * n_samples as f64).floor() as usize
        }
    };

    if resolved == 0 || resolved > n_samples {
        return Err(ModelingLabError::InvalidConfig(
            "max_samples resolved out of range".to_string(),
        ));
    }

    Ok(resolved)
}

fn uniqueness_capped_max_samples(base_max_samples: usize, average_uniqueness: f64) -> usize {
    let multiplier = average_uniqueness.clamp(0.0, 1.0);
    let capped = ((base_max_samples as f64) * multiplier).ceil() as usize;
    capped.clamp(1, base_max_samples)
}

fn bootstrap_diversity_benchmark(
    ind_mat: &[Vec<u8>],
    sample_length: usize,
    trials: usize,
    seed: u64,
) -> Result<BootstrapDiversitySummary, ModelingLabError> {
    let n_labels = ind_mat.first().map(|row| row.len()).ok_or(ModelingLabError::EmptyInput)?;
    if n_labels == 0 {
        return Err(ModelingLabError::EmptyInput);
    }

    let mut rng = StdRng::seed_from_u64(seed);
    let mut vanilla_uniqueness = Vec::with_capacity(trials);
    let mut sequential_uniqueness = Vec::with_capacity(trials);

    for _ in 0..trials {
        let vanilla_indices: Vec<usize> =
            (0..sample_length).map(|_| rng.gen_range(0..n_labels)).collect();
        let warmup: Vec<usize> = (0..3.min(sample_length))
            .map(|_| rng.gen_range(0..n_labels))
            .collect();
        let seq_indices = seq_bootstrap(ind_mat, Some(sample_length), Some(warmup));

        vanilla_uniqueness.push(sampled_label_uniqueness(ind_mat, &vanilla_indices)?);
        sequential_uniqueness.push(sampled_label_uniqueness(ind_mat, &seq_indices)?);
    }

    Ok(BootstrapDiversitySummary {
        vanilla_uniqueness_mean: mean(&vanilla_uniqueness),
        sequential_uniqueness_mean: mean(&sequential_uniqueness),
    })
}

fn sampled_label_uniqueness(
    ind_mat: &[Vec<u8>],
    sampled_indices: &[usize],
) -> Result<f64, ModelingLabError> {
    if sampled_indices.is_empty() {
        return Err(ModelingLabError::EmptyInput);
    }

    let mut sampled = Vec::with_capacity(ind_mat.len());
    for row in ind_mat {
        let mut projected = Vec::with_capacity(sampled_indices.len());
        for idx in sampled_indices {
            let value = row.get(*idx).ok_or_else(|| {
                ModelingLabError::DimensionMismatch(
                    "sampled index out of indicator matrix bounds".to_string(),
                )
            })?;
            projected.push(*value);
        }
        sampled.push(projected);
    }

    Ok(get_ind_mat_average_uniqueness(&sampled))
}

fn subset_indicator_matrix(
    ind_mat: &[Vec<u8>],
    indices: &[usize],
) -> Result<Vec<Vec<u8>>, ModelingLabError> {
    let mut subset = vec![vec![0u8; indices.len()]; indices.len()];
    for (r_sub, r) in indices.iter().enumerate() {
        for (c_sub, c) in indices.iter().enumerate() {
            let row = ind_mat.get(*r).ok_or_else(|| {
                ModelingLabError::DimensionMismatch(
                    "indicator matrix row out of bounds".to_string(),
                )
            })?;
            subset[r_sub][c_sub] = *row.get(*c).ok_or_else(|| {
                ModelingLabError::DimensionMismatch(
                    "indicator matrix column out of bounds".to_string(),
                )
            })?;
        }
    }
    Ok(subset)
}

fn select_matrix_rows(x: &DMatrix<f64>, indices: &[usize]) -> DMatrix<f64> {
    DMatrix::from_fn(indices.len(), x.ncols(), |r, c| x[(indices[r], c)])
}

fn select_labels(y: &[u8], indices: &[usize]) -> Vec<u8> {
    indices.iter().map(|idx| y[*idx]).collect()
}

fn validate_xy_info_sets(
    x: &[Vec<f64>],
    y: &[u8],
    samples_info_sets: &[(NaiveDateTime, NaiveDateTime)],
) -> Result<(), ModelingLabError> {
    validate_feature_matrix(x)?;
    if y.len() != x.len() {
        return Err(ModelingLabError::DimensionMismatch("y length must equal x rows".to_string()));
    }
    validate_binary_labels(y)?;
    if samples_info_sets.len() != x.len() {
        return Err(ModelingLabError::DimensionMismatch(
            "samples_info_sets length must equal x rows".to_string(),
        ));
    }
    Ok(())
}

fn validate_bagged_inputs(
    x: &DMatrix<f64>,
    y: &[u8],
    samples_info_sets: &[(NaiveDateTime, NaiveDateTime)],
    ind_mat: &[Vec<u8>],
) -> Result<(), ModelingLabError> {
    if x.nrows() == 0 || x.ncols() == 0 {
        return Err(ModelingLabError::EmptyInput);
    }
    if y.len() != x.nrows() {
        return Err(ModelingLabError::DimensionMismatch("y length must equal x rows".to_string()));
    }
    validate_binary_labels(y)?;
    if samples_info_sets.len() != x.nrows() {
        return Err(ModelingLabError::DimensionMismatch(
            "samples_info_sets length must equal x rows".to_string(),
        ));
    }
    if ind_mat.len() != x.nrows() {
        return Err(ModelingLabError::DimensionMismatch(
            "ind_mat must have one row per sample".to_string(),
        ));
    }
    if ind_mat.iter().any(|row| row.len() != x.nrows()) {
        return Err(ModelingLabError::DimensionMismatch(
            "ind_mat must be square with one label per sample".to_string(),
        ));
    }
    Ok(())
}

fn validate_feature_matrix(x: &[Vec<f64>]) -> Result<(), ModelingLabError> {
    if x.is_empty() {
        return Err(ModelingLabError::EmptyInput);
    }
    let width = x[0].len();
    if width == 0 {
        return Err(ModelingLabError::EmptyInput);
    }
    if x.iter().any(|row| row.len() != width) {
        return Err(ModelingLabError::DimensionMismatch(
            "all x rows must have the same width".to_string(),
        ));
    }
    Ok(())
}

fn validate_binary_labels(y: &[u8]) -> Result<(), ModelingLabError> {
    if y.iter().any(|v| *v > 1) {
        return Err(ModelingLabError::InvalidLabel("labels must be binary in {0,1}".to_string()));
    }
    Ok(())
}

fn sigmoid(z: f64) -> f64 {
    if z >= 0.0 {
        let ez = (-z).exp();
        1.0 / (1.0 + ez)
    } else {
        let ez = z.exp();
        ez / (1.0 + ez)
    }
}

fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f64>() / values.len() as f64
    }
}

fn neg_log_loss_u8(probs: &[f64], y: &[u8]) -> f64 {
    let eps = 1e-15;
    let mut loss = 0.0;
    for (p, t) in probs.iter().zip(y.iter()) {
        let p_clip = p.max(eps).min(1.0 - eps);
        let yv = *t as f64;
        loss += -(yv * p_clip.ln() + (1.0 - yv) * (1.0 - p_clip).ln());
    }
    loss / y.len() as f64
}

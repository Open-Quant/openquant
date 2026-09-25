use nalgebra::DMatrix;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};

use crate::sampling::seq_bootstrap_with_rng;

#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum SbBaggingError {
    #[error("input must not be empty")]
    EmptyInput,
    #[error("inputs disagree on the number of samples")]
    DimensionMismatch,
    #[error("max_samples is out of range")]
    MaxSamplesOutOfRange,
    #[error("max_features is out of range")]
    MaxFeaturesOutOfRange,
    #[error("out-of-bag scoring is not supported with warm_start")]
    WarmStartWithOob,
    #[error("n_estimators must not decrease when warm_start is set")]
    DecreasingEstimators,
    #[error("n_estimators must be positive")]
    InvalidEstimators,
    #[error("the base estimator does not support sample weights")]
    SampleWeightNotSupported,
    #[error("sample weights must be finite, non-negative and not all zero")]
    InvalidSampleWeight,
}

#[derive(Debug, Clone, Copy)]
pub enum MaxSamples {
    Int(usize),
    Float(f64),
}

#[derive(Debug, Clone, Copy)]
pub enum MaxFeatures {
    Int(usize),
    Float(f64),
}

#[derive(Debug, Clone)]
struct ClassifierEstimator {
    feature_idx: usize,
    threshold: f64,
    positive_on_ge: bool,
}

impl ClassifierEstimator {
    fn predicts_one(&self, x: &DMatrix<f64>, row: usize) -> bool {
        let ge = x[(row, self.feature_idx)] >= self.threshold;
        if self.positive_on_ge {
            ge
        } else {
            !ge
        }
    }
}

#[derive(Debug, Clone)]
struct RegressorEstimator {
    feature_idx: usize,
    slope: f64,
    intercept: f64,
}

impl RegressorEstimator {
    fn predict_row(&self, x: &DMatrix<f64>, row: usize) -> f64 {
        self.slope * x[(row, self.feature_idx)] + self.intercept
    }
}

fn validate_and_resolve_max_samples(
    max_samples: MaxSamples,
    n_samples: usize,
) -> Result<usize, SbBaggingError> {
    let resolved = match max_samples {
        MaxSamples::Int(v) => v,
        MaxSamples::Float(v) => {
            if v <= 0.0 {
                return Err(SbBaggingError::MaxSamplesOutOfRange);
            }
            (v * n_samples as f64) as usize
        }
    };
    if resolved == 0 || resolved > n_samples {
        return Err(SbBaggingError::MaxSamplesOutOfRange);
    }
    Ok(resolved)
}

fn validate_and_resolve_max_features(
    max_features: MaxFeatures,
    n_features: usize,
) -> Result<usize, SbBaggingError> {
    let resolved = match max_features {
        MaxFeatures::Int(v) => v,
        MaxFeatures::Float(v) => {
            if v <= 0.0 {
                return Err(SbBaggingError::MaxFeaturesOutOfRange);
            }
            (v * n_features as f64) as usize
        }
    };
    if resolved == 0 || resolved > n_features {
        return Err(SbBaggingError::MaxFeaturesOutOfRange);
    }
    Ok(resolved.max(1))
}

/// Checks shared by both estimators. Label `j` of `ind_mat` (column `j`) is row `j` of `x`,
/// so every row of `ind_mat` must have `x.nrows()` entries.
fn validate_fit_inputs(
    x: &DMatrix<f64>,
    y_len: usize,
    ind_mat: &[Vec<u8>],
    sample_weight: Option<&[f64]>,
    supports_sample_weight: bool,
) -> Result<(), SbBaggingError> {
    if x.nrows() == 0 || x.ncols() == 0 {
        return Err(SbBaggingError::EmptyInput);
    }
    if y_len != x.nrows() {
        return Err(SbBaggingError::DimensionMismatch);
    }
    if ind_mat.is_empty() || ind_mat.iter().any(|row| row.len() != x.nrows()) {
        return Err(SbBaggingError::DimensionMismatch);
    }
    if let Some(w) = sample_weight {
        if !supports_sample_weight {
            return Err(SbBaggingError::SampleWeightNotSupported);
        }
        if w.len() != x.nrows() {
            return Err(SbBaggingError::DimensionMismatch);
        }
        if w.iter().any(|v| !v.is_finite() || *v < 0.0) || w.iter().all(|v| *v == 0.0) {
            return Err(SbBaggingError::InvalidSampleWeight);
        }
    }
    Ok(())
}

fn sampled_features(
    rng: &mut StdRng,
    n_features: usize,
    max_features: usize,
    bootstrap_features: bool,
) -> Vec<usize> {
    if bootstrap_features {
        (0..max_features).map(|_| rng.gen_range(0..n_features)).collect()
    } else {
        let mut all: Vec<usize> = (0..n_features).collect();
        all.shuffle(rng);
        all.into_iter().take(max_features).collect()
    }
}

/// One weight per draw (a row drawn twice counts twice). A bag whose drawn rows all have
/// zero weight falls back to equal weights.
fn bag_weights(samples: &[usize], sample_weight: Option<&[f64]>) -> Vec<f64> {
    match sample_weight {
        Some(w) => {
            let bag: Vec<f64> = samples.iter().map(|&i| w[i]).collect();
            if bag.iter().sum::<f64>() > 0.0 {
                bag
            } else {
                vec![1.0; samples.len()]
            }
        }
        None => vec![1.0; samples.len()],
    }
}

/// `in_bag[e][row]` is true when estimator `e` drew `row`.
fn in_bag_masks(estimators_samples: &[Vec<usize>], n_rows: usize) -> Vec<Vec<bool>> {
    estimators_samples
        .iter()
        .map(|samples| {
            let mut mask = vec![false; n_rows];
            for &i in samples {
                mask[i] = true;
            }
            mask
        })
        .collect()
}

#[derive(Debug, Clone)]
pub struct SequentiallyBootstrappedBaggingClassifier {
    pub n_estimators: usize,
    pub max_samples: MaxSamples,
    pub max_features: MaxFeatures,
    pub bootstrap_features: bool,
    pub oob_score: bool,
    pub warm_start: bool,
    pub verbose: usize,
    pub random_state: u64,
    pub supports_sample_weight: bool,
    pub estimators_samples: Vec<Vec<usize>>,
    pub oob_score_value: Option<f64>,
    estimators: Vec<ClassifierEstimator>,
}

impl SequentiallyBootstrappedBaggingClassifier {
    pub fn new(random_state: u64) -> Self {
        Self {
            n_estimators: 10,
            max_samples: MaxSamples::Float(1.0),
            max_features: MaxFeatures::Float(1.0),
            bootstrap_features: false,
            oob_score: false,
            warm_start: false,
            verbose: 0,
            random_state,
            supports_sample_weight: true,
            estimators_samples: Vec::new(),
            oob_score_value: None,
            estimators: Vec::new(),
        }
    }

    /// Fits `n_estimators` stumps, each on a sequential bootstrap sample of the labels in
    /// `ind_mat` (bars x labels, one label per row of `x`).
    pub fn fit(
        &mut self,
        x: &DMatrix<f64>,
        y: &[u8],
        ind_mat: &[Vec<u8>],
        sample_weight: Option<&[f64]>,
    ) -> Result<(), SbBaggingError> {
        validate_fit_inputs(x, y.len(), ind_mat, sample_weight, self.supports_sample_weight)?;
        if self.n_estimators == 0 {
            return Err(SbBaggingError::InvalidEstimators);
        }
        if self.warm_start && self.oob_score {
            return Err(SbBaggingError::WarmStartWithOob);
        }

        let max_samples = validate_and_resolve_max_samples(self.max_samples, x.nrows())?;
        let max_features = validate_and_resolve_max_features(self.max_features, x.ncols())?;

        if !self.warm_start {
            self.estimators.clear();
            self.estimators_samples.clear();
        }

        let n_more = self.n_estimators as isize - self.estimators.len() as isize;
        if n_more < 0 {
            return Err(SbBaggingError::DecreasingEstimators);
        }
        if n_more == 0 {
            return Ok(());
        }

        let mut rng = StdRng::seed_from_u64(self.random_state + self.estimators.len() as u64);

        for _ in 0..(n_more as usize) {
            let features =
                sampled_features(&mut rng, x.ncols(), max_features, self.bootstrap_features);
            let samples = seq_bootstrap_with_rng(ind_mat, Some(max_samples), None, &mut rng)
                .map_err(|_| SbBaggingError::DimensionMismatch)?;
            let weights = bag_weights(&samples, sample_weight);

            let feature_idx = *features.first().ok_or(SbBaggingError::EmptyInput)?;

            let total: f64 = weights.iter().sum();
            let thr =
                samples.iter().zip(&weights).map(|(&i, w)| w * x[(i, feature_idx)]).sum::<f64>()
                    / total;

            let mut pos_ge = 0.0;
            let mut tot_ge = 0.0;
            let mut pos_lt = 0.0;
            let mut tot_lt = 0.0;
            for (&i, &w) in samples.iter().zip(&weights) {
                let positive = if y[i] == 1 { w } else { 0.0 };
                if x[(i, feature_idx)] >= thr {
                    tot_ge += w;
                    pos_ge += positive;
                } else {
                    tot_lt += w;
                    pos_lt += positive;
                }
            }
            let rate_ge = if tot_ge > 0.0 { pos_ge / tot_ge } else { 0.0 };
            let rate_lt = if tot_lt > 0.0 { pos_lt / tot_lt } else { 0.0 };

            self.estimators.push(ClassifierEstimator {
                feature_idx,
                threshold: thr,
                positive_on_ge: rate_ge >= rate_lt,
            });
            self.estimators_samples.push(samples);
        }

        self.oob_score_value = if self.oob_score { self.out_of_bag_accuracy(x, y) } else { None };

        Ok(())
    }

    /// Accuracy over the rows that at least one estimator did not draw, each row predicted
    /// by majority vote of only those estimators. `None` if every row was drawn by every
    /// estimator.
    fn out_of_bag_accuracy(&self, x: &DMatrix<f64>, y: &[u8]) -> Option<f64> {
        let in_bag = in_bag_masks(&self.estimators_samples, x.nrows());
        let mut scored = 0usize;
        let mut correct = 0usize;
        for (r, &target) in y.iter().enumerate() {
            let mut votes = 0usize;
            let mut voters = 0usize;
            for (est, mask) in self.estimators.iter().zip(&in_bag) {
                if !mask[r] {
                    voters += 1;
                    votes += usize::from(est.predicts_one(x, r));
                }
            }
            if voters > 0 {
                scored += 1;
                let pred = u8::from(votes * 2 >= voters);
                correct += usize::from(pred == target);
            }
        }
        (scored > 0).then(|| correct as f64 / scored as f64)
    }

    pub fn predict(&self, x: &DMatrix<f64>) -> Result<Vec<u8>, SbBaggingError> {
        if self.estimators.is_empty() {
            return Err(SbBaggingError::EmptyInput);
        }
        let mut out = vec![0u8; x.nrows()];
        for (r, pred) in out.iter_mut().enumerate() {
            let votes = self.estimators.iter().filter(|est| est.predicts_one(x, r)).count();
            *pred = u8::from(votes * 2 >= self.estimators.len());
        }
        Ok(out)
    }
}

#[derive(Debug, Clone)]
pub struct SequentiallyBootstrappedBaggingRegressor {
    pub n_estimators: usize,
    pub max_samples: MaxSamples,
    pub max_features: MaxFeatures,
    pub bootstrap_features: bool,
    pub oob_score: bool,
    pub warm_start: bool,
    pub random_state: u64,
    pub supports_sample_weight: bool,
    pub estimators_samples: Vec<Vec<usize>>,
    pub oob_score_value: Option<f64>,
    estimators: Vec<RegressorEstimator>,
}

impl SequentiallyBootstrappedBaggingRegressor {
    pub fn new(random_state: u64) -> Self {
        Self {
            n_estimators: 10,
            max_samples: MaxSamples::Float(1.0),
            max_features: MaxFeatures::Float(1.0),
            bootstrap_features: false,
            oob_score: false,
            warm_start: false,
            random_state,
            supports_sample_weight: true,
            estimators_samples: Vec::new(),
            oob_score_value: None,
            estimators: Vec::new(),
        }
    }

    /// Fits `n_estimators` least-squares lines, each on a sequential bootstrap sample of the
    /// labels in `ind_mat` (bars x labels, one label per row of `x`).
    pub fn fit(
        &mut self,
        x: &DMatrix<f64>,
        y: &[f64],
        ind_mat: &[Vec<u8>],
        sample_weight: Option<&[f64]>,
    ) -> Result<(), SbBaggingError> {
        validate_fit_inputs(x, y.len(), ind_mat, sample_weight, self.supports_sample_weight)?;
        if self.n_estimators == 0 {
            return Err(SbBaggingError::InvalidEstimators);
        }
        if self.warm_start && self.oob_score {
            return Err(SbBaggingError::WarmStartWithOob);
        }

        let max_samples = validate_and_resolve_max_samples(self.max_samples, x.nrows())?;
        let max_features = validate_and_resolve_max_features(self.max_features, x.ncols())?;

        if !self.warm_start {
            self.estimators.clear();
            self.estimators_samples.clear();
        }

        let n_more = self.n_estimators as isize - self.estimators.len() as isize;
        if n_more < 0 {
            return Err(SbBaggingError::DecreasingEstimators);
        }
        if n_more == 0 {
            return Ok(());
        }

        let mut rng = StdRng::seed_from_u64(self.random_state + self.estimators.len() as u64);

        for _ in 0..(n_more as usize) {
            let features =
                sampled_features(&mut rng, x.ncols(), max_features, self.bootstrap_features);
            let samples = seq_bootstrap_with_rng(ind_mat, Some(max_samples), None, &mut rng)
                .map_err(|_| SbBaggingError::DimensionMismatch)?;
            let weights = bag_weights(&samples, sample_weight);

            let feature_idx = *features.first().ok_or(SbBaggingError::EmptyInput)?;
            let total: f64 = weights.iter().sum();
            let mean_x =
                samples.iter().zip(&weights).map(|(&i, w)| w * x[(i, feature_idx)]).sum::<f64>()
                    / total;
            let mean_y = samples.iter().zip(&weights).map(|(&i, w)| w * y[i]).sum::<f64>() / total;
            let mut cov_xy = 0.0;
            let mut var_x = 0.0;
            for (&i, &w) in samples.iter().zip(&weights) {
                let dx = x[(i, feature_idx)] - mean_x;
                cov_xy += w * dx * (y[i] - mean_y);
                var_x += w * dx * dx;
            }
            let slope = if var_x <= 1e-12 { 0.0 } else { cov_xy / var_x };
            let intercept = mean_y - slope * mean_x;

            self.estimators.push(RegressorEstimator { feature_idx, slope, intercept });
            self.estimators_samples.push(samples);
        }

        self.oob_score_value = if self.oob_score { self.out_of_bag_r2(x, y) } else { None };

        Ok(())
    }

    /// R² over the rows that at least one estimator did not draw, each row predicted by the
    /// mean of only those estimators. `None` if every row was drawn by every estimator.
    fn out_of_bag_r2(&self, x: &DMatrix<f64>, y: &[f64]) -> Option<f64> {
        let in_bag = in_bag_masks(&self.estimators_samples, x.nrows());
        let mut scored: Vec<(f64, f64)> = Vec::new();
        for (r, &target) in y.iter().enumerate() {
            let mut sum = 0.0;
            let mut voters = 0usize;
            for (est, mask) in self.estimators.iter().zip(&in_bag) {
                if !mask[r] {
                    voters += 1;
                    sum += est.predict_row(x, r);
                }
            }
            if voters > 0 {
                scored.push((sum / voters as f64, target));
            }
        }
        if scored.is_empty() {
            return None;
        }
        let mean = scored.iter().map(|(_, t)| t).sum::<f64>() / scored.len() as f64;
        let ss_tot = scored.iter().map(|(_, t)| (t - mean) * (t - mean)).sum::<f64>();
        let ss_res = scored.iter().map(|(p, t)| (p - t) * (p - t)).sum::<f64>();
        Some(if ss_tot <= 1e-12 { 0.0 } else { 1.0 - ss_res / ss_tot })
    }

    pub fn predict(&self, x: &DMatrix<f64>) -> Result<Vec<f64>, SbBaggingError> {
        if self.estimators.is_empty() {
            return Err(SbBaggingError::EmptyInput);
        }
        let mut out = vec![0.0; x.nrows()];
        for (r, pred) in out.iter_mut().enumerate() {
            let s: f64 = self.estimators.iter().map(|est| est.predict_row(x, r)).sum();
            *pred = s / self.estimators.len() as f64;
        }
        Ok(out)
    }
}

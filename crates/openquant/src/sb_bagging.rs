use nalgebra::DMatrix;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};

use crate::sampling::seq_bootstrap;

#[derive(Debug, Clone, PartialEq)]
pub enum SbBaggingError {
    EmptyInput,
    DimensionMismatch,
    MaxSamplesOutOfRange,
    MaxFeaturesOutOfRange,
    WarmStartWithOob,
    DecreasingEstimators,
    InvalidEstimators,
    SampleWeightNotSupported,
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

#[derive(Debug, Clone)]
struct RegressorEstimator {
    feature_idx: usize,
    slope: f64,
    intercept: f64,
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

fn warmup_indices(rng: &mut StdRng, n_labels: usize, n: usize) -> Vec<usize> {
    (0..n).map(|_| rng.gen_range(0..n_labels)).collect()
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

    pub fn fit(
        &mut self,
        x: &DMatrix<f64>,
        y: &[u8],
        ind_mat: &[Vec<u8>],
        sample_weight: Option<&[f64]>,
    ) -> Result<(), SbBaggingError> {
        if x.nrows() == 0 || x.ncols() == 0 {
            return Err(SbBaggingError::EmptyInput);
        }
        if y.len() != x.nrows() {
            return Err(SbBaggingError::DimensionMismatch);
        }
        if self.n_estimators == 0 {
            return Err(SbBaggingError::InvalidEstimators);
        }
        if !self.supports_sample_weight && sample_weight.is_some() {
            return Err(SbBaggingError::SampleWeightNotSupported);
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
            let warmup = warmup_indices(
                &mut rng,
                ind_mat.first().map(|r| r.len()).unwrap_or(0).max(1),
                max_samples,
            );
            let samples = seq_bootstrap(ind_mat, Some(max_samples), Some(warmup));

            let feature_idx = *features.first().ok_or(SbBaggingError::EmptyInput)?;

            let mut thr = 0.0;
            for &i in &samples {
                thr += x[(i, feature_idx)];
            }
            thr /= samples.len() as f64;

            let mut pos_ge = 0usize;
            let mut tot_ge = 0usize;
            let mut pos_lt = 0usize;
            let mut tot_lt = 0usize;
            for &i in &samples {
                if x[(i, feature_idx)] >= thr {
                    tot_ge += 1;
                    if y[i] == 1 {
                        pos_ge += 1;
                    }
                } else {
                    tot_lt += 1;
                    if y[i] == 1 {
                        pos_lt += 1;
                    }
                }
            }
            let rate_ge = if tot_ge == 0 { 0.0 } else { pos_ge as f64 / tot_ge as f64 };
            let rate_lt = if tot_lt == 0 { 0.0 } else { pos_lt as f64 / tot_lt as f64 };

            self.estimators.push(ClassifierEstimator {
                feature_idx,
                threshold: thr,
                positive_on_ge: rate_ge >= rate_lt,
            });
            self.estimators_samples.push(samples);
        }

        if self.oob_score {
            let preds = self.predict(x)?;
            let correct = preds.iter().zip(y.iter()).filter(|(p, t)| **p == **t).count();
            self.oob_score_value = Some(correct as f64 / y.len() as f64);
        }

        Ok(())
    }

    pub fn predict(&self, x: &DMatrix<f64>) -> Result<Vec<u8>, SbBaggingError> {
        if self.estimators.is_empty() {
            return Err(SbBaggingError::EmptyInput);
        }
        let mut out = vec![0u8; x.nrows()];
        for r in 0..x.nrows() {
            let mut votes = 0usize;
            for est in &self.estimators {
                let ge = x[(r, est.feature_idx)] >= est.threshold;
                let pred_one = if est.positive_on_ge { ge } else { !ge };
                if pred_one {
                    votes += 1;
                }
            }
            out[r] = if votes * 2 >= self.estimators.len() { 1 } else { 0 };
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

    pub fn fit(
        &mut self,
        x: &DMatrix<f64>,
        y: &[f64],
        ind_mat: &[Vec<u8>],
        sample_weight: Option<&[f64]>,
    ) -> Result<(), SbBaggingError> {
        if x.nrows() == 0 || x.ncols() == 0 {
            return Err(SbBaggingError::EmptyInput);
        }
        if y.len() != x.nrows() {
            return Err(SbBaggingError::DimensionMismatch);
        }
        if self.n_estimators == 0 {
            return Err(SbBaggingError::InvalidEstimators);
        }
        if !self.supports_sample_weight && sample_weight.is_some() {
            return Err(SbBaggingError::SampleWeightNotSupported);
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
            let warmup = warmup_indices(
                &mut rng,
                ind_mat.first().map(|r| r.len()).unwrap_or(0).max(1),
                max_samples,
            );
            let samples = seq_bootstrap(ind_mat, Some(max_samples), Some(warmup));

            let feature_idx = *features.first().ok_or(SbBaggingError::EmptyInput)?;
            let n = samples.len() as f64;
            let mean_x = samples.iter().map(|&i| x[(i, feature_idx)]).sum::<f64>() / n;
            let mean_y = samples.iter().map(|&i| y[i]).sum::<f64>() / n;
            let mut cov_xy = 0.0;
            let mut var_x = 0.0;
            for &i in &samples {
                let dx = x[(i, feature_idx)] - mean_x;
                cov_xy += dx * (y[i] - mean_y);
                var_x += dx * dx;
            }
            let slope = if var_x <= 1e-12 { 0.0 } else { cov_xy / var_x };
            let intercept = mean_y - slope * mean_x;

            self.estimators.push(RegressorEstimator { feature_idx, slope, intercept });
            self.estimators_samples.push(samples);
        }

        if self.oob_score {
            let preds = self.predict(x)?;
            let mean = y.iter().sum::<f64>() / y.len() as f64;
            let ss_tot = y.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>();
            let ss_res = preds
                .iter()
                .zip(y.iter())
                .map(|(p, t)| {
                    let d = p - t;
                    d * d
                })
                .sum::<f64>();
            self.oob_score_value = Some(if ss_tot <= 1e-12 { 0.0 } else { 1.0 - ss_res / ss_tot });
        }

        Ok(())
    }

    pub fn predict(&self, x: &DMatrix<f64>) -> Result<Vec<f64>, SbBaggingError> {
        if self.estimators.is_empty() {
            return Err(SbBaggingError::EmptyInput);
        }
        let mut out = vec![0.0; x.nrows()];
        for r in 0..x.nrows() {
            let mut s = 0.0;
            for est in &self.estimators {
                let xv = x[(r, est.feature_idx)];
                s += est.slope * xv + est.intercept;
            }
            out[r] = s / self.estimators.len() as f64;
        }
        Ok(out)
    }
}

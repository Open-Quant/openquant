//! Bagging ensembles whose bootstrap samples are drawn with the sequential bootstrap (AFML
//! §4.5, §4.5.1; bagging per §6.3 and Breiman, 1996).
//!
//! With overlapping financial labels a uniform bootstrap draws samples full of
//! near-duplicates, so the estimators are too alike for averaging to remove much variance,
//! and out-of-bag rows resemble in-bag rows, inflating out-of-bag scores (AFML §4.5). The
//! types here draw each estimator's sample with [`seq_bootstrap_with_rng`] instead. They port
//! mlfinlab's `SequentiallyBootstrappedBaggingClassifier` and
//! `SequentiallyBootstrappedBaggingRegressor`.
//!
//! **The sampling is real; the base learner is a sketch.** For each estimator, `fit`
//! draws `max_features` column indices and **keeps only the first**, draws `max_samples`
//! label indices with the sequential bootstrap, and fits a fixed one-feature learner on
//! those rows: for [`SequentiallyBootstrappedBaggingClassifier`], a stump whose threshold is
//! the (weighted) *mean* of the feature over the sample, predicting whichever side had the
//! higher rate of `y == 1`; for [`SequentiallyBootstrappedBaggingRegressor`], a weighted
//! least-squares line. A result from this module is a result about sequentially bootstrapped
//! bagging of one-feature stumps or lines, not of a real model; for production, pass
//! sequential-bootstrap indices to your own learners.
//!
//! Conventions:
//!
//! - `x` is observations × features, one row per label.
//! - `ind_mat` is the bars × labels indicator matrix from
//!   [`get_ind_matrix`](crate::sampling::get_ind_matrix): its **columns** correspond
//!   one-to-one with the rows of `x`.
//! - Classifier labels are `u8` with 1 as the positive class; anything else counts as
//!   negative, so a −1/+1 encoding must be mapped to 0/1 first.
//! - Configuration is by public field after `new(random_state)`; settings are validated when
//!   `fit` is called. The same `random_state` and inputs reproduce a fit.
//! - `max_features` above one column changes nothing except the random stream, because only
//!   the first sampled feature is used.
//! - Out-of-bag scores remain optimistic when labels overlap, because a held-out label still
//!   overlaps drawn labels in time (AFML §4.5); treat them as a sanity check and score the
//!   model under purged cross-validation ([`crate::cross_validation`]).
//! - Each sequential draw rescans the whole indicator matrix, so one estimator costs on the
//!   order of bars × labels × `max_samples` operations.
//!
//! ```
//! use nalgebra::DMatrix;
//! use openquant::sampling::get_ind_matrix;
//! use openquant::sb_bagging::{
//!     MaxSamples, SbBaggingError, SequentiallyBootstrappedBaggingClassifier,
//! };
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let n = 40;
//! let spans: Vec<(usize, usize)> = (0..n).map(|i| (2 * i, 2 * i + 5)).collect();
//! let bars: Vec<usize> = (0..2 * n + 6).collect();
//! let ind_mat = get_ind_matrix(&spans, &bars)?;
//!
//! // One feature; the class is its sign.
//! let x = DMatrix::from_fn(n, 1, |r, _| r as f64 - 19.5);
//! let y: Vec<u8> = (0..n).map(|r| u8::from(r >= 20)).collect();
//!
//! let mut model = SequentiallyBootstrappedBaggingClassifier::new(7);
//! model.n_estimators = 25;
//! model.max_samples = MaxSamples::Float(0.5);
//! model.fit(&x, &y, &ind_mat, None)?;
//!
//! assert_eq!(model.estimators_samples.len(), 25);
//! assert_eq!(model.estimators_samples[0].len(), 20);
//! let fresh = DMatrix::from_row_slice(2, 1, &[-15.0, 15.0]);
//! assert_eq!(model.predict(&fresh)?, vec![0, 1]);
//!
//! // Settings are validated at fit time.
//! model.max_samples = MaxSamples::Float(1.5);
//! assert_eq!(model.fit(&x, &y, &ind_mat, None), Err(SbBaggingError::MaxSamplesOutOfRange));
//! # Ok(())
//! # }
//! ```
#![deny(missing_docs)]

use nalgebra::DMatrix;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};

use crate::sampling::seq_bootstrap_with_rng;

/// Errors returned by the sequentially bootstrapped bagging estimators.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum SbBaggingError {
    /// `x` has no rows or no columns at fit time, or `predict` was called before a
    /// successful `fit`.
    #[error("input must not be empty")]
    EmptyInput,
    /// `x` has a different number of columns from the matrix the model was fitted on, at
    /// prediction time or in a `warm_start` fit that adds estimators to a fitted model.
    #[error("x has {found} feature columns but the model was fitted on {expected}")]
    FeatureCountMismatch {
        /// Number of columns of the matrix the fitted estimators were trained on.
        expected: usize,
        /// Number of columns of the `x` that was passed.
        found: usize,
    },
    /// `y`, `sample_weight` or the rows of `ind_mat` do not match `x.nrows()`, or `ind_mat`
    /// is empty.
    #[error("inputs disagree on the number of samples")]
    DimensionMismatch,
    /// `max_samples` resolves to zero or to more than `x.nrows()`, or is a non-positive
    /// fraction.
    #[error("max_samples is out of range")]
    MaxSamplesOutOfRange,
    /// `max_features` resolves to zero or to more than `x.ncols()`, or is a non-positive
    /// fraction.
    #[error("max_features is out of range")]
    MaxFeaturesOutOfRange,
    /// `warm_start` and `oob_score` were both set.
    #[error("out-of-bag scoring is not supported with warm_start")]
    WarmStartWithOob,
    /// With `warm_start`, `n_estimators` is lower than the number already fitted.
    #[error("n_estimators must not decrease when warm_start is set")]
    DecreasingEstimators,
    /// `n_estimators` is zero.
    #[error("n_estimators must be positive")]
    InvalidEstimators,
    /// A `sample_weight` was passed while `supports_sample_weight` is `false`.
    #[error("the base estimator does not support sample weights")]
    SampleWeightNotSupported,
    /// A sample weight is negative or non-finite, or all weights are zero.
    #[error("sample weights must be finite, non-negative and not all zero")]
    InvalidSampleWeight,
}

/// Number of label draws per estimator.
#[derive(Debug, Clone, Copy)]
pub enum MaxSamples {
    /// An absolute count; must be in `1..=x.nrows()`.
    Int(usize),
    /// A fraction of `x.nrows()`, rounded down; must be positive and resolve to a count in
    /// `1..=x.nrows()` (so at most `1.0`).
    Float(f64),
}

/// Number of feature indices drawn per estimator. Only the first drawn index is used by the
/// base learner, so values above one change only the random stream.
#[derive(Debug, Clone, Copy)]
pub enum MaxFeatures {
    /// An absolute count; must be in `1..=x.ncols()`.
    Int(usize),
    /// A fraction of `x.ncols()`, rounded down; must be positive and resolve to a count in
    /// `1..=x.ncols()` (so at most `1.0`).
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

/// Seed of the random stream for a `fit` that starts with `n_fitted` estimators already
/// fitted: `random_state + n_fitted`, wrapping around modulo 2^64. Wrapping only changes
/// seeds that would overflow `u64`, so every seed that fits is unchanged.
fn fit_seed(random_state: u64, n_fitted: usize) -> u64 {
    random_state.wrapping_add(n_fitted as u64)
}

/// Number of estimators a `fit` must add to reach `n_estimators`, after the column count of
/// `x` has been checked against the estimators kept by `warm_start`. Records the column
/// count the model is now fitted on in `n_features_in`.
fn estimators_to_add(
    n_estimators: usize,
    n_fitted: usize,
    n_features_in: &mut usize,
    x: &DMatrix<f64>,
) -> Result<usize, SbBaggingError> {
    if n_fitted > 0 && x.ncols() != *n_features_in {
        return Err(SbBaggingError::FeatureCountMismatch {
            expected: *n_features_in,
            found: x.ncols(),
        });
    }
    let n_more = n_estimators.checked_sub(n_fitted).ok_or(SbBaggingError::DecreasingEstimators)?;
    *n_features_in = x.ncols();
    Ok(n_more)
}

/// Checks that the model is fitted and that `x` has the column count it was fitted on, so
/// every estimator's feature index is a valid column of `x`.
fn check_predict_input(
    n_fitted: usize,
    n_features_in: usize,
    x: &DMatrix<f64>,
) -> Result<(), SbBaggingError> {
    if n_fitted == 0 {
        return Err(SbBaggingError::EmptyInput);
    }
    if x.ncols() != n_features_in {
        return Err(SbBaggingError::FeatureCountMismatch {
            expected: n_features_in,
            found: x.ncols(),
        });
    }
    Ok(())
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

/// Bagging classifier over one-feature decision stumps, each fitted on a sequential bootstrap
/// sample of the labels (AFML §4.5).
///
/// Each stump's threshold is the weighted *mean* of its feature over the sample (not a
/// fitted split), and it predicts 1 on whichever side had the higher weighted rate of
/// `y == 1` (ties favour the `>=` side). [`predict`](Self::predict) takes a majority vote,
/// ties going to 1. See the module example.
#[derive(Debug, Clone)]
pub struct SequentiallyBootstrappedBaggingClassifier {
    /// Number of estimators to fit (default 10). With `warm_start`, the total after `fit`.
    pub n_estimators: usize,
    /// Label draws per estimator (default `Float(1.0)`, i.e. as many as rows of `x`).
    pub max_samples: MaxSamples,
    /// Feature indices drawn per estimator (default `Float(1.0)`); only the first is used.
    pub max_features: MaxFeatures,
    /// Draw feature indices with replacement instead of as a random permutation (default
    /// `false`).
    pub bootstrap_features: bool,
    /// Compute [`oob_score_value`](Self::oob_score_value) during `fit` (default `false`).
    pub oob_score: bool,
    /// Keep already fitted estimators and add up to `n_estimators` on the next `fit`
    /// (default `false`). Cannot be combined with `oob_score`.
    pub warm_start: bool,
    /// Ignored: nothing is logged. Kept only so existing code still compiles.
    #[deprecated(note = "ignored: the classifier logs nothing")]
    pub verbose: usize,
    /// Seed of the random stream; a `fit` seeds its stream with `random_state` plus the
    /// number of estimators already fitted, wrapping around at `u64::MAX`.
    pub random_state: u64,
    /// Whether `fit` accepts a `sample_weight` (default `true`).
    pub supports_sample_weight: bool,
    /// Row indices of `x` (label indices) drawn for each fitted estimator, in fit order; a
    /// row drawn twice appears twice.
    pub estimators_samples: Vec<Vec<usize>>,
    /// Out-of-bag accuracy set by `fit` when `oob_score` is on: over the rows that at least
    /// one estimator did not draw, each predicted by majority vote of only those estimators.
    /// `None` if `oob_score` is off or every row was drawn by every estimator.
    pub oob_score_value: Option<f64>,
    estimators: Vec<ClassifierEstimator>,
    /// Column count of the `x` the estimators were fitted on (meaningful once fitted).
    n_features_in: usize,
}

impl SequentiallyBootstrappedBaggingClassifier {
    /// Creates an unfitted classifier with default settings and the given seed.
    pub fn new(random_state: u64) -> Self {
        #[allow(deprecated)]
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
            n_features_in: 0,
        }
    }

    /// Fits `n_estimators` stumps, each on a sequential bootstrap sample of the labels in
    /// `ind_mat` (bars x labels, one label per row of `x`).
    ///
    /// `y` holds one `u8` label per row of `x` (1 is positive, anything else negative).
    /// `sample_weight`, if given, holds one weight per row; each draw is weighted by its
    /// row's weight (a row drawn twice counts twice), and a bag whose drawn rows all have
    /// zero weight falls back to equal weights. Without `warm_start`, previously fitted
    /// estimators are discarded.
    ///
    /// # Errors
    ///
    /// - [`SbBaggingError::EmptyInput`] if `x` has no rows or no columns.
    /// - [`SbBaggingError::DimensionMismatch`] if `y.len()` or `sample_weight.len()` differs
    ///   from `x.nrows()`, if `ind_mat` is empty, or if any row of `ind_mat` does not have
    ///   `x.nrows()` entries.
    /// - [`SbBaggingError::SampleWeightNotSupported`] if a weight vector is passed while
    ///   `supports_sample_weight` is `false`.
    /// - [`SbBaggingError::InvalidSampleWeight`] for a negative or non-finite weight, or all
    ///   zero weights.
    /// - [`SbBaggingError::InvalidEstimators`] if `n_estimators == 0`.
    /// - [`SbBaggingError::WarmStartWithOob`] if both `warm_start` and `oob_score` are set.
    /// - [`SbBaggingError::MaxSamplesOutOfRange`] or [`SbBaggingError::MaxFeaturesOutOfRange`]
    ///   if `max_samples` or `max_features` does not resolve to a valid count.
    /// - [`SbBaggingError::DecreasingEstimators`] if, with `warm_start`, `n_estimators` is
    ///   below the number already fitted.
    /// - [`SbBaggingError::FeatureCountMismatch`] if, with `warm_start`, the model is already
    ///   fitted and `x` has a different number of columns from the matrix it was fitted on.
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

        let n_more = estimators_to_add(
            self.n_estimators,
            self.estimators.len(),
            &mut self.n_features_in,
            x,
        )?;
        if n_more == 0 {
            return Ok(());
        }

        let mut rng = StdRng::seed_from_u64(fit_seed(self.random_state, self.estimators.len()));

        for _ in 0..n_more {
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

    /// Predicts a 0/1 label per row of `x` by majority vote of all estimators, ties going
    /// to 1.
    ///
    /// # Errors
    ///
    /// - [`SbBaggingError::EmptyInput`] if the model has not been fitted.
    /// - [`SbBaggingError::FeatureCountMismatch`] if `x` does not have the same number of
    ///   columns as the matrix the model was fitted on.
    pub fn predict(&self, x: &DMatrix<f64>) -> Result<Vec<u8>, SbBaggingError> {
        check_predict_input(self.estimators.len(), self.n_features_in, x)?;
        let mut out = vec![0u8; x.nrows()];
        for (r, pred) in out.iter_mut().enumerate() {
            let votes = self.estimators.iter().filter(|est| est.predicts_one(x, r)).count();
            *pred = u8::from(votes * 2 >= self.estimators.len());
        }
        Ok(out)
    }

    /// Probability of class 1 for each row of `x`: the fraction of estimators that vote 1.
    ///
    /// The stumps have no probability of their own, so this is the vote share, as in
    /// scikit-learn's `BaggingClassifier` over estimators without `predict_proba`. It is
    /// consistent with [`predict`](Self::predict): a row is predicted 1 exactly when its
    /// probability is at least 0.5. The probability of class 0 is one minus this value.
    ///
    /// # Errors
    ///
    /// - [`SbBaggingError::EmptyInput`] if the model has not been fitted.
    /// - [`SbBaggingError::FeatureCountMismatch`] if `x` does not have the same number of
    ///   columns as the matrix the model was fitted on.
    pub fn predict_proba(&self, x: &DMatrix<f64>) -> Result<Vec<f64>, SbBaggingError> {
        check_predict_input(self.estimators.len(), self.n_features_in, x)?;
        let n = self.estimators.len() as f64;
        Ok((0..x.nrows())
            .map(|r| self.estimators.iter().filter(|est| est.predicts_one(x, r)).count() as f64 / n)
            .collect())
    }
}

/// Bagging regressor over one-feature weighted least-squares lines, each fitted on a
/// sequential bootstrap sample of the labels (AFML §4.5).
///
/// A line whose sample has (near-)zero feature variance is flat at the weighted mean of
/// `y`. [`predict`](Self::predict) returns the mean of the lines.
///
/// ```
/// use nalgebra::DMatrix;
/// use openquant::sampling::get_ind_matrix;
/// use openquant::sb_bagging::{MaxSamples, SequentiallyBootstrappedBaggingRegressor};
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let n = 30;
/// let spans: Vec<(usize, usize)> = (0..n).map(|i| (i, i + 3)).collect();
/// let bars: Vec<usize> = (0..n + 4).collect();
/// let ind_mat = get_ind_matrix(&spans, &bars)?;
///
/// // y = 2x + 1 exactly, so every line recovers it.
/// let x = DMatrix::from_fn(n, 1, |r, _| r as f64);
/// let y: Vec<f64> = (0..n).map(|r| 2.0 * r as f64 + 1.0).collect();
///
/// let mut model = SequentiallyBootstrappedBaggingRegressor::new(11);
/// model.max_samples = MaxSamples::Float(0.5);
/// model.oob_score = true;
/// model.fit(&x, &y, &ind_mat, None)?;
///
/// let pred = model.predict(&DMatrix::from_row_slice(1, 1, &[100.0]))?;
/// assert!((pred[0] - 201.0).abs() < 1e-9);
/// assert!((model.oob_score_value.unwrap() - 1.0).abs() < 1e-12);
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct SequentiallyBootstrappedBaggingRegressor {
    /// Number of estimators to fit (default 10). With `warm_start`, the total after `fit`.
    pub n_estimators: usize,
    /// Label draws per estimator (default `Float(1.0)`, i.e. as many as rows of `x`).
    pub max_samples: MaxSamples,
    /// Feature indices drawn per estimator (default `Float(1.0)`); only the first is used.
    pub max_features: MaxFeatures,
    /// Draw feature indices with replacement instead of as a random permutation (default
    /// `false`).
    pub bootstrap_features: bool,
    /// Compute [`oob_score_value`](Self::oob_score_value) during `fit` (default `false`).
    pub oob_score: bool,
    /// Keep already fitted estimators and add up to `n_estimators` on the next `fit`
    /// (default `false`). Cannot be combined with `oob_score`.
    pub warm_start: bool,
    /// Seed of the random stream; a `fit` seeds its stream with `random_state` plus the
    /// number of estimators already fitted, wrapping around at `u64::MAX`.
    pub random_state: u64,
    /// Whether `fit` accepts a `sample_weight` (default `true`).
    pub supports_sample_weight: bool,
    /// Row indices of `x` (label indices) drawn for each fitted estimator, in fit order; a
    /// row drawn twice appears twice.
    pub estimators_samples: Vec<Vec<usize>>,
    /// Out-of-bag R² set by `fit` when `oob_score` is on: over the rows that at least one
    /// estimator did not draw, each predicted by the mean of only those estimators (0 when
    /// the scored targets are constant). `None` if `oob_score` is off or every row was drawn
    /// by every estimator.
    pub oob_score_value: Option<f64>,
    estimators: Vec<RegressorEstimator>,
    /// Column count of the `x` the estimators were fitted on (meaningful once fitted).
    n_features_in: usize,
}

impl SequentiallyBootstrappedBaggingRegressor {
    /// Creates an unfitted regressor with default settings and the given seed.
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
            n_features_in: 0,
        }
    }

    /// Fits `n_estimators` least-squares lines, each on a sequential bootstrap sample of the
    /// labels in `ind_mat` (bars x labels, one label per row of `x`).
    ///
    /// `y` holds one target per row of `x` and is not checked for `NaN`. `sample_weight`
    /// behaves as in [`SequentiallyBootstrappedBaggingClassifier::fit`].
    ///
    /// # Errors
    ///
    /// The same as [`SequentiallyBootstrappedBaggingClassifier::fit`].
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

        let n_more = estimators_to_add(
            self.n_estimators,
            self.estimators.len(),
            &mut self.n_features_in,
            x,
        )?;
        if n_more == 0 {
            return Ok(());
        }

        let mut rng = StdRng::seed_from_u64(fit_seed(self.random_state, self.estimators.len()));

        for _ in 0..n_more {
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

    /// Predicts one value per row of `x` as the mean of all estimators' lines.
    ///
    /// # Errors
    ///
    /// - [`SbBaggingError::EmptyInput`] if the model has not been fitted.
    /// - [`SbBaggingError::FeatureCountMismatch`] if `x` does not have the same number of
    ///   columns as the matrix the model was fitted on.
    pub fn predict(&self, x: &DMatrix<f64>) -> Result<Vec<f64>, SbBaggingError> {
        check_predict_input(self.estimators.len(), self.n_features_in, x)?;
        let mut out = vec![0.0; x.nrows()];
        for (r, pred) in out.iter_mut().enumerate() {
            let s: f64 = self.estimators.iter().map(|est| est.predict_row(x, r)).sum();
            *pred = s / self.estimators.len() as f64;
        }
        Ok(out)
    }
}

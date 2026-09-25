//! Leakage-aware hyperparameter search (AFML chapter 9).
//!
//! [`grid_search`] (§9.2) and [`randomized_search`] (§9.3) evaluate parameter sets with
//! purged, embargoed k-fold cross-validation ([`PurgedKFold`], chapter 7). For every trial
//! and fold a fresh model is built from the [`ParamSet`], fitted with the training sample
//! weights, and scored on the test fold **with the test sample weights**, the correction
//! Snippet 9.1 makes to scikit-learn. [`SearchScoring::NegLogLoss`] is AFML's recommended
//! score (§9.4) because it penalises confident mistakes, which matter to a strategy that
//! sizes bets by probability.
//!
//! Conventions: labels are binary `0.0`/`1.0`; the classifier returns the probability of
//! class 1; `samples_info_sets` holds each label's `(start, end)` span, in the same order as
//! the rows of `x`. Searches are deterministic for a given classifier and (for
//! [`randomized_search`]) seed. The search does not refit the winner.
//!
//! ```
//! use std::collections::BTreeMap;
//!
//! use chrono::{Duration, NaiveDate};
//! use openquant::cross_validation::SimpleClassifier;
//! use openquant::hyperparameter_tuning::{
//!     grid_search, HyperParamValue, ParamSet, SearchData, SearchScoring,
//! };
//!
//! /// Ignores the features and predicts a fixed probability `p` of class 1.
//! struct Constant(f64);
//! impl SimpleClassifier for Constant {
//!     fn fit(&mut self, _x: &[Vec<f64>], _y: &[f64], _w: Option<&[f64]>) {}
//!     fn predict_proba(&self, x: &[Vec<f64>]) -> Vec<f64> {
//!         vec![self.0; x.len()]
//!     }
//! }
//!
//! # fn main() -> Result<(), openquant::hyperparameter_tuning::TuningError> {
//! // 50 samples, 7 in every 10 labelled 1, each label spanning one hour.
//! let x = vec![vec![0.0]; 50];
//! let y: Vec<f64> = (0..50).map(|i| if i % 10 < 7 { 1.0 } else { 0.0 }).collect();
//! let t0 = NaiveDate::from_ymd_opt(2024, 1, 2).unwrap().and_hms_opt(0, 0, 0).unwrap();
//! let info: Vec<_> =
//!     (0..50).map(|i| (t0 + Duration::hours(i), t0 + Duration::hours(i + 1))).collect();
//! let data = SearchData { x: &x, y: &y, sample_weight: None, samples_info_sets: &info };
//!
//! let grid = BTreeMap::from([(
//!     "p".to_string(),
//!     [0.5, 0.7, 0.9].map(HyperParamValue::Float).to_vec(),
//! )]);
//! let build = |params: &ParamSet| Constant(params["p"].as_f64().unwrap());
//! let result = grid_search(build, &grid, data, 5, 0.0, SearchScoring::NegLogLoss)?;
//!
//! // Log loss is minimised by the calibrated probability, 0.7.
//! assert_eq!(result.best_params["p"], HyperParamValue::Float(0.7));
//! let expected = 0.7 * 0.7f64.ln() + 0.3 * 0.3f64.ln();
//! assert!((result.best_score - expected).abs() < 1e-12);
//! assert_eq!(result.trials.len(), 3);
//! # Ok(())
//! # }
//! ```
#![deny(missing_docs)]

use std::collections::BTreeMap;

use chrono::NaiveDateTime;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use crate::cross_validation::{PurgedKFold, SimpleClassifier};

/// Errors returned by the hyperparameter search functions.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum TuningError {
    /// A grid entry has no values.
    #[error("param_grid entry '{0}' cannot be empty")]
    EmptyGridEntry(String),
    /// A parameter has no distribution (internal consistency check).
    #[error("missing distribution for key '{0}'")]
    MissingDistribution(String),
    /// [`PurgedKFold`] rejected the configuration (e.g. more splits than samples).
    #[error(transparent)]
    CrossValidation(#[from] crate::cross_validation::CrossValidationError),
    /// A log-uniform bound is not strictly positive.
    #[error("log-uniform bounds must be strictly positive")]
    NonPositiveLogUniformBounds,
    /// A log-uniform `low` is not below `high`.
    #[error("log-uniform low must be < high")]
    InvalidLogUniformOrder,
    /// The named input is empty.
    #[error("{0} cannot be empty")]
    Empty(&'static str),
    /// `probabilities` and `y_true` differ in length.
    #[error("probabilities/y_true length mismatch")]
    ProbabilitiesLabelsLengthMismatch,
    /// The named input's length does not match the labels.
    #[error("{0} length mismatch")]
    LengthMismatch(&'static str),
    /// A sample weight is negative.
    #[error("sample_weight cannot contain negative values")]
    NegativeSampleWeight,
    /// A probability is not finite or outside `[0, 1]`.
    #[error("probabilities must be finite and in [0,1]")]
    InvalidProbabilities,
    /// A label is not 0 or 1.
    #[error("y_true must contain only binary labels in {{0,1}}")]
    NonBinaryLabels,
    /// The sample weights sum to zero.
    #[error("sum of sample_weight must be > 0")]
    ZeroSampleWeightSum,
    /// Balanced accuracy found no weighted sample of either class.
    #[error("balanced accuracy requires at least one labeled sample")]
    NoLabeledSamples,
    /// The named parameter violates its requirement.
    #[error("{name} must be {requirement}")]
    Invalid {
        /// Parameter name.
        name: &'static str,
        /// What it must satisfy.
        requirement: &'static str,
    },
    /// A [`RandomParamDistribution::Choice`] has no values.
    #[error("choice distribution cannot be empty")]
    EmptyChoice,
    /// A [`RandomParamDistribution::Uniform`] has non-finite bounds or `low >= high`.
    #[error("uniform bounds must be finite and satisfy low < high")]
    InvalidUniformBounds,
    /// A [`RandomParamDistribution::IntRangeInclusive`] has `low > high`.
    #[error("IntRangeInclusive requires low <= high")]
    InvalidIntRange,
    /// No parameter set was evaluated.
    #[error("no trials produced")]
    NoTrials,
    /// `x` and `y` differ in length.
    #[error("x/y length mismatch")]
    XyLengthMismatch,
    /// `samples_info_sets` and `x` differ in length.
    #[error("samples_info_sets length must match x length")]
    SamplesInfoSetsLengthMismatch,
    /// Purging left a train or test fold empty.
    #[error("PurgedKFold generated an empty train/test fold")]
    EmptyFold,
}

#[derive(Debug, Clone, PartialEq)]
/// A hyperparameter value.
pub enum HyperParamValue {
    /// An integer value.
    Int(i64),
    /// A floating-point value.
    Float(f64),
    /// A boolean flag.
    Bool(bool),
}

impl HyperParamValue {
    /// The value if it is an [`Int`](Self::Int), else `None`.
    pub fn as_i64(&self) -> Option<i64> {
        match self {
            Self::Int(v) => Some(*v),
            _ => None,
        }
    }

    /// The value as `f64` if it is a [`Float`](Self::Float) or an [`Int`](Self::Int), else
    /// `None`.
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Self::Int(v) => Some(*v as f64),
            Self::Float(v) => Some(*v),
            _ => None,
        }
    }

    /// The value if it is a [`Bool`](Self::Bool), else `None`.
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Self::Bool(v) => Some(*v),
            _ => None,
        }
    }
}

/// One parameter set: parameter name to value, ordered by name.
pub type ParamSet = BTreeMap<String, HyperParamValue>;

#[derive(Debug, Clone)]
/// A distribution to draw one parameter from in [`randomized_search`].
pub enum RandomParamDistribution {
    /// Uniform choice among the listed values.
    Choice(Vec<HyperParamValue>),
    /// Uniform float in `[low, high)`; bounds finite with `low < high`.
    Uniform {
        /// Lower bound (inclusive).
        low: f64,
        /// Upper bound (exclusive).
        high: f64,
    },
    /// Float whose logarithm is uniform in `[ln low, ln high)` (§9.3.1); `0 < low < high`.
    LogUniform {
        /// Lower bound (inclusive), positive.
        low: f64,
        /// Upper bound (exclusive), positive.
        high: f64,
    },
    /// Uniform integer in `[low, high]`.
    IntRangeInclusive {
        /// Lower bound (inclusive).
        low: i64,
        /// Upper bound (inclusive).
        high: i64,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Cross-validation score to maximise; see [`classification_score`].
pub enum SearchScoring {
    /// Weighted share of labels matched at a 0.5 threshold.
    Accuracy,
    /// Mean of the weighted per-class recalls, over the classes present in the fold.
    BalancedAccuracy,
    /// Weighted mean log-likelihood of the true label (higher is better; at most 0).
    NegLogLoss,
}

#[derive(Debug, Clone, PartialEq)]
/// The evaluation of one parameter set.
pub struct SearchTrial {
    /// The parameters evaluated.
    pub params: ParamSet,
    /// Score on each test fold, in fold order.
    pub fold_scores: Vec<f64>,
    /// Unweighted mean of `fold_scores`.
    pub mean_score: f64,
}

#[derive(Debug, Clone, PartialEq)]
/// The outcome of a search.
pub struct SearchResult {
    /// Parameters of the trial with the highest mean score (ties go to the last trial).
    pub best_params: ParamSet,
    /// That trial's mean score. It was selected for being high, so it is not an
    /// out-of-sample estimate.
    pub best_score: f64,
    /// Every trial in evaluation order; `trials.len()` is the number of trials to carry into
    /// a multiple-testing correction such as the deflated Sharpe ratio.
    pub trials: Vec<SearchTrial>,
}

/// Draws `x` with `ln x` uniform on `[ln low, ln high)` (AFML §9.3.1), so every order of
/// magnitude between the bounds is equally likely.
///
/// # Errors
///
/// - [`TuningError::NonPositiveLogUniformBounds`] if either bound is not positive.
/// - [`TuningError::InvalidLogUniformOrder`] if `low >= high`.
pub fn sample_log_uniform<R: Rng + ?Sized>(
    low: f64,
    high: f64,
    rng: &mut R,
) -> Result<f64, TuningError> {
    if low <= 0.0 || high <= 0.0 {
        return Err(TuningError::NonPositiveLogUniformBounds);
    }
    if low >= high {
        return Err(TuningError::InvalidLogUniformOrder);
    }
    let log_low = low.ln();
    let log_high = high.ln();
    let draw = rng.gen_range(log_low..log_high);
    Ok(draw.exp())
}

/// Scores binary predictions, optionally sample-weighted (AFML Snippet 9.1's weighted
/// scoring).
///
/// `y_true` holds 0/1 labels, `probabilities` the predicted probability of class 1, and
/// `sample_weight` a non-negative weight per sample (samples with weight zero are skipped).
/// Predictions for accuracy use a 0.5 threshold (`p >= 0.5` is class 1); log loss clips
/// probabilities to `[1e-15, 1 - 1e-15]`.
///
/// # Errors
///
/// - [`TuningError::Empty`] if `y_true` is empty.
/// - [`TuningError::ProbabilitiesLabelsLengthMismatch`] if `probabilities` differs in length.
/// - [`TuningError::LengthMismatch`] if `sample_weight` differs in length.
/// - [`TuningError::NegativeSampleWeight`] if a weight is negative.
/// - [`TuningError::InvalidProbabilities`] if a probability is not finite or outside `[0, 1]`.
/// - [`TuningError::NonBinaryLabels`] if a label is not 0 or 1.
/// - [`TuningError::ZeroSampleWeightSum`] if the weights sum to zero.
/// - [`TuningError::NoLabeledSamples`] for balanced accuracy with no weighted sample.
///
/// ```
/// use openquant::hyperparameter_tuning::{classification_score, SearchScoring};
///
/// # fn main() -> Result<(), openquant::hyperparameter_tuning::TuningError> {
/// let y = [1.0, 0.0, 1.0, 1.0];
/// let p = [0.9, 0.2, 0.4, 0.8];
/// assert_eq!(classification_score(&y, &p, None, SearchScoring::Accuracy)?, 0.75);
/// // Recall is 2/3 on class 1 and 1 on class 0.
/// let balanced = classification_score(&y, &p, None, SearchScoring::BalancedAccuracy)?;
/// assert!((balanced - 5.0 / 6.0).abs() < 1e-12);
/// # Ok(())
/// # }
/// ```
pub fn classification_score(
    y_true: &[f64],
    probabilities: &[f64],
    sample_weight: Option<&[f64]>,
    scoring: SearchScoring,
) -> Result<f64, TuningError> {
    if y_true.is_empty() {
        return Err(TuningError::Empty("y_true"));
    }
    if probabilities.len() != y_true.len() {
        return Err(TuningError::ProbabilitiesLabelsLengthMismatch);
    }
    if let Some(sw) = sample_weight {
        if sw.len() != y_true.len() {
            return Err(TuningError::LengthMismatch("sample_weight"));
        }
        if sw.iter().any(|w| *w < 0.0) {
            return Err(TuningError::NegativeSampleWeight);
        }
    }
    if probabilities.iter().any(|p| !p.is_finite() || *p < 0.0 || *p > 1.0) {
        return Err(TuningError::InvalidProbabilities);
    }
    if y_true.iter().any(|y| (*y - 0.0).abs() > 1e-12 && (*y - 1.0).abs() > 1e-12) {
        return Err(TuningError::NonBinaryLabels);
    }

    let mut sum_w = 0.0;
    let mut weighted_correct = 0.0;
    let mut weighted_loss = 0.0;

    let mut pos_total = 0.0;
    let mut neg_total = 0.0;
    let mut pos_correct = 0.0;
    let mut neg_correct = 0.0;

    let eps = 1e-15;
    for i in 0..y_true.len() {
        let w = sample_weight.map(|sw| sw[i]).unwrap_or(1.0);
        if w == 0.0 {
            continue;
        }
        let y = y_true[i];
        let p = probabilities[i].max(eps).min(1.0 - eps);
        let pred = if probabilities[i] >= 0.5 { 1.0 } else { 0.0 };

        sum_w += w;
        if (pred - y).abs() < 1e-12 {
            weighted_correct += w;
        }

        weighted_loss += -w * (y * p.ln() + (1.0 - y) * (1.0 - p).ln());

        if y == 1.0 {
            pos_total += w;
            if pred == 1.0 {
                pos_correct += w;
            }
        } else {
            neg_total += w;
            if pred == 0.0 {
                neg_correct += w;
            }
        }
    }

    if sum_w <= 0.0 {
        return Err(TuningError::ZeroSampleWeightSum);
    }

    let accuracy = weighted_correct / sum_w;
    let neg_log_loss = -(weighted_loss / sum_w);

    match scoring {
        SearchScoring::Accuracy => Ok(accuracy),
        SearchScoring::NegLogLoss => Ok(neg_log_loss),
        SearchScoring::BalancedAccuracy => {
            // Handle single-class folds by averaging recall over classes present in the fold.
            let mut recalls = Vec::new();
            if pos_total > 0.0 {
                recalls.push(pos_correct / pos_total);
            }
            if neg_total > 0.0 {
                recalls.push(neg_correct / neg_total);
            }
            if recalls.is_empty() {
                return Err(TuningError::NoLabeledSamples);
            }
            Ok(recalls.iter().sum::<f64>() / recalls.len() as f64)
        }
    }
}

/// Expands a grid into every combination of its values (the Cartesian product), iterating
/// keys in name order with the last key varying fastest.
///
/// # Errors
///
/// - [`TuningError::Empty`] if the grid has no keys.
/// - [`TuningError::EmptyGridEntry`] if a key has no values.
pub fn expand_param_grid(
    param_grid: &BTreeMap<String, Vec<HyperParamValue>>,
) -> Result<Vec<ParamSet>, TuningError> {
    if param_grid.is_empty() {
        return Err(TuningError::Empty("param_grid"));
    }
    for (name, values) in param_grid {
        if values.is_empty() {
            return Err(TuningError::EmptyGridEntry(name.clone()));
        }
    }

    let keys: Vec<String> = param_grid.keys().cloned().collect();
    let mut out = Vec::new();
    let mut current = ParamSet::new();
    expand_grid_recursive(&keys, 0, param_grid, &mut current, &mut out);
    Ok(out)
}

fn expand_grid_recursive(
    keys: &[String],
    idx: usize,
    grid: &BTreeMap<String, Vec<HyperParamValue>>,
    current: &mut ParamSet,
    out: &mut Vec<ParamSet>,
) {
    if idx == keys.len() {
        out.push(current.clone());
        return;
    }

    let key = &keys[idx];
    if let Some(values) = grid.get(key) {
        for value in values {
            current.insert(key.clone(), value.clone());
            expand_grid_recursive(keys, idx + 1, grid, current, out);
        }
    }
}

/// The data a search cross-validates on; all slices are aligned by sample.
pub struct SearchData<'a> {
    /// Features, one row per sample.
    pub x: &'a [Vec<f64>],
    /// Binary labels (`0.0` or `1.0`).
    pub y: &'a [f64],
    /// Optional non-negative sample weights, used both to fit and to score.
    pub sample_weight: Option<&'a [f64]>,
    /// Each label's `(start, end)` span, used to purge and embargo the folds.
    pub samples_info_sets: &'a [(NaiveDateTime, NaiveDateTime)],
}

/// Exhaustive search over every combination in `param_grid` (AFML §9.2), scored with purged
/// k-fold cross-validation.
///
/// `build_classifier` creates a fresh model from a [`ParamSet`] for every trial and fold.
/// `n_splits` is the number of folds and `pct_embargo` the embargo as a fraction of the
/// sample count. Returns every trial and the best by mean fold score.
///
/// # Errors
///
/// - [`TuningError::Empty`] or [`TuningError::EmptyGridEntry`] from [`expand_param_grid`].
/// - [`TuningError::Empty`] if `x` or `y` is empty.
/// - [`TuningError::XyLengthMismatch`] or [`TuningError::SamplesInfoSetsLengthMismatch`] if
///   the data slices disagree in length.
/// - [`TuningError::Invalid`] if `n_splits < 2`.
/// - [`TuningError::LengthMismatch`] or [`TuningError::NegativeSampleWeight`] for bad sample
///   weights.
/// - [`TuningError::CrossValidation`] if [`PurgedKFold`] rejects the setup (e.g. more splits
///   than samples).
/// - [`TuningError::EmptyFold`] if purging empties a train or test fold.
/// - Any [`classification_score`] error on a fold (non-binary labels, invalid
///   probabilities, a fold whose weights sum to zero).
pub fn grid_search<C, F>(
    build_classifier: F,
    param_grid: &BTreeMap<String, Vec<HyperParamValue>>,
    data: SearchData<'_>,
    n_splits: usize,
    pct_embargo: f64,
    scoring: SearchScoring,
) -> Result<SearchResult, TuningError>
where
    C: SimpleClassifier,
    F: Fn(&ParamSet) -> C,
{
    let params = expand_param_grid(param_grid)?;
    search_over_params(build_classifier, params, data, n_splits, pct_embargo, scoring)
}

/// Randomised search (AFML §9.3): draws `n_iter` parameter sets from `param_space` and
/// scores each with purged k-fold cross-validation, as [`grid_search`] does.
///
/// Draws come from a [`StdRng`] seeded with `seed`, so results are reproducible for a given
/// seed and classifier. Random search usually spends a fixed budget better than a grid when
/// only some parameters matter.
///
/// # Errors
///
/// - [`TuningError::Empty`] if `param_space` is empty.
/// - [`TuningError::Invalid`] if `n_iter` is zero.
/// - [`TuningError::EmptyChoice`], [`TuningError::InvalidUniformBounds`],
///   [`TuningError::NonPositiveLogUniformBounds`], [`TuningError::InvalidLogUniformOrder`] or
///   [`TuningError::InvalidIntRange`] for an invalid distribution.
/// - [`TuningError::Empty`] if `x` or `y` is empty.
/// - [`TuningError::XyLengthMismatch`] or [`TuningError::SamplesInfoSetsLengthMismatch`] if
///   the data slices disagree in length.
/// - [`TuningError::Invalid`] if `n_splits < 2`.
/// - [`TuningError::LengthMismatch`] or [`TuningError::NegativeSampleWeight`] for bad sample
///   weights.
/// - [`TuningError::CrossValidation`] if [`PurgedKFold`] rejects the setup (e.g. more splits
///   than samples).
/// - [`TuningError::EmptyFold`] if purging empties a train or test fold.
/// - Any [`classification_score`] error on a fold (non-binary labels, invalid
///   probabilities, a fold whose weights sum to zero).
// Public search API: `grid_search`'s arguments plus `n_iter` and `seed`; signature kept stable.
#[allow(clippy::too_many_arguments)]
pub fn randomized_search<C, F>(
    build_classifier: F,
    param_space: &BTreeMap<String, RandomParamDistribution>,
    n_iter: usize,
    seed: u64,
    data: SearchData<'_>,
    n_splits: usize,
    pct_embargo: f64,
    scoring: SearchScoring,
) -> Result<SearchResult, TuningError>
where
    C: SimpleClassifier,
    F: Fn(&ParamSet) -> C,
{
    let params = sample_param_sets(param_space, n_iter, seed)?;
    search_over_params(build_classifier, params, data, n_splits, pct_embargo, scoring)
}

/// The `n_iter` parameter sets [`randomized_search`] evaluates for `param_space` and `seed`,
/// in the same order.
///
/// Each draw samples every key of `param_space` in key order from one `StdRng` seeded with
/// `seed`, so the same inputs always give the same sets. Useful to run the search's
/// candidates through a model this crate cannot call, for example from Python.
///
/// # Errors
/// [`TuningError::Empty`] when `param_space` is empty, [`TuningError::Invalid`] when `n_iter`
/// is 0, and the distribution errors of [`RandomParamDistribution`] (`EmptyChoice`,
/// `InvalidUniformBounds`, log-uniform bound errors, `InvalidIntRange`).
pub fn sample_param_sets(
    param_space: &BTreeMap<String, RandomParamDistribution>,
    n_iter: usize,
    seed: u64,
) -> Result<Vec<ParamSet>, TuningError> {
    if param_space.is_empty() {
        return Err(TuningError::Empty("param_space"));
    }
    if n_iter == 0 {
        return Err(TuningError::Invalid { name: "n_iter", requirement: "> 0" });
    }

    let mut rng = StdRng::seed_from_u64(seed);
    let mut params = Vec::with_capacity(n_iter);
    for _ in 0..n_iter {
        let mut draw = ParamSet::new();
        for (key, dist) in param_space {
            draw.insert(key.clone(), sample_distribution(dist, &mut rng)?);
        }
        params.push(draw);
    }
    Ok(params)
}

fn sample_distribution<R: Rng + ?Sized>(
    dist: &RandomParamDistribution,
    rng: &mut R,
) -> Result<HyperParamValue, TuningError> {
    match dist {
        RandomParamDistribution::Choice(values) => {
            if values.is_empty() {
                return Err(TuningError::EmptyChoice);
            }
            let idx = rng.gen_range(0..values.len());
            Ok(values[idx].clone())
        }
        RandomParamDistribution::Uniform { low, high } => {
            if !low.is_finite() || !high.is_finite() || low >= high {
                return Err(TuningError::InvalidUniformBounds);
            }
            Ok(HyperParamValue::Float(rng.gen_range(*low..*high)))
        }
        RandomParamDistribution::LogUniform { low, high } => {
            let v = sample_log_uniform(*low, *high, rng)?;
            Ok(HyperParamValue::Float(v))
        }
        RandomParamDistribution::IntRangeInclusive { low, high } => {
            if low > high {
                return Err(TuningError::InvalidIntRange);
            }
            Ok(HyperParamValue::Int(rng.gen_range(*low..=*high)))
        }
    }
}

fn search_over_params<C, F>(
    build_classifier: F,
    param_sets: Vec<ParamSet>,
    data: SearchData<'_>,
    n_splits: usize,
    pct_embargo: f64,
    scoring: SearchScoring,
) -> Result<SearchResult, TuningError>
where
    C: SimpleClassifier,
    F: Fn(&ParamSet) -> C,
{
    validate_search_data(&data, n_splits)?;
    let cv = PurgedKFold::new(n_splits, data.samples_info_sets.to_vec(), pct_embargo)?;
    let splits = cv.split(data.x.len())?;

    let mut trials = Vec::with_capacity(param_sets.len());
    for params in param_sets {
        let fold_scores = evaluate_params(
            &build_classifier,
            &params,
            &splits,
            data.x,
            data.y,
            data.sample_weight,
            scoring,
        )?;
        let mean_score = fold_scores.iter().sum::<f64>() / fold_scores.len() as f64;
        trials.push(SearchTrial { params, fold_scores, mean_score });
    }

    let best = trials
        .iter()
        .max_by(|a, b| a.mean_score.partial_cmp(&b.mean_score).unwrap_or(std::cmp::Ordering::Equal))
        .cloned()
        .ok_or(TuningError::NoTrials)?;

    Ok(SearchResult { best_params: best.params, best_score: best.mean_score, trials })
}

fn validate_search_data(data: &SearchData<'_>, n_splits: usize) -> Result<(), TuningError> {
    if data.x.is_empty() {
        return Err(TuningError::Empty("x"));
    }
    if data.y.is_empty() {
        return Err(TuningError::Empty("y"));
    }
    if data.x.len() != data.y.len() {
        return Err(TuningError::XyLengthMismatch);
    }
    if data.samples_info_sets.len() != data.x.len() {
        return Err(TuningError::SamplesInfoSetsLengthMismatch);
    }
    if n_splits < 2 {
        return Err(TuningError::Invalid { name: "n_splits", requirement: ">= 2" });
    }
    if let Some(sw) = data.sample_weight {
        if sw.len() != data.y.len() {
            return Err(TuningError::LengthMismatch("sample_weight"));
        }
        if sw.iter().any(|w| *w < 0.0) {
            return Err(TuningError::NegativeSampleWeight);
        }
    }
    Ok(())
}

fn evaluate_params<C, F>(
    build_classifier: &F,
    params: &ParamSet,
    splits: &[(Vec<usize>, Vec<usize>)],
    x: &[Vec<f64>],
    y: &[f64],
    sample_weight: Option<&[f64]>,
    scoring: SearchScoring,
) -> Result<Vec<f64>, TuningError>
where
    C: SimpleClassifier,
    F: Fn(&ParamSet) -> C,
{
    let mut fold_scores = Vec::with_capacity(splits.len());

    for (train_idx, test_idx) in splits {
        if train_idx.is_empty() || test_idx.is_empty() {
            return Err(TuningError::EmptyFold);
        }

        let x_train: Vec<Vec<f64>> = train_idx.iter().map(|i| x[*i].clone()).collect();
        let y_train: Vec<f64> = train_idx.iter().map(|i| y[*i]).collect();
        let x_test: Vec<Vec<f64>> = test_idx.iter().map(|i| x[*i].clone()).collect();
        let y_test: Vec<f64> = test_idx.iter().map(|i| y[*i]).collect();

        let sw_train: Option<Vec<f64>> =
            sample_weight.map(|sw| train_idx.iter().map(|i| sw[*i]).collect());
        let sw_test: Option<Vec<f64>> =
            sample_weight.map(|sw| test_idx.iter().map(|i| sw[*i]).collect());

        let mut clf = build_classifier(params);
        clf.fit(&x_train, &y_train, sw_train.as_deref());
        let probs = clf.predict_proba(&x_test);
        let fold_score = classification_score(&y_test, &probs, sw_test.as_deref(), scoring)?;
        fold_scores.push(fold_score);
    }

    Ok(fold_scores)
}

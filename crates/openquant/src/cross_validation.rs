//! Purged k-fold cross-validation with an embargo, and combinatorial purged cross-validation
//! (AFML Chapter 7 and §12.4).
//!
//! Financial labels overlap in time: a label formed at bar `t` and resolved at `t + h` depends
//! on every price in between, so ordinary k-fold leaks test information into training (AFML
//! §7.3). [`PurgedKFold`] removes that leak in two steps:
//!
//! - **Purging** (§7.4.1, Snippet 7.1): every non-test sample whose information set
//!   `(start, end)` intersects a test block's window is dropped from training. The window runs
//!   from the block's first start to the *latest* end among its samples.
//! - **Embargo** (§7.4.2, Snippet 7.3): `h = ceil(pct_embargo * n_samples)` further samples
//!   are dropped *after* each test block, counting from where the purge ends (the book's
//!   `maxT1Idx`). Nothing before a block is embargoed.
//!
//! [`PurgedKFold::cpcv_splits`] and [`PurgedKFold::cpcv_paths`] give the combinatorial
//! version (§12.4); [`crate::backtesting_engine`] shares the embargo code and uses the same
//! split and path numbering. [`naive_kfold_splits`] and [`count_train_test_overlaps`] measure
//! the leak that purging removes, and [`ml_cross_val_score`] scores a [`SimpleClassifier`] on
//! any list of splits.
//!
//! Conventions:
//!
//! - One information set `(start, end)` per sample, **in time order** (sorted by start), one
//!   per row of the feature matrix. Folds are blocks of consecutive indices; the purge
//!   compares timestamps, but the embargo counts positions, so unsorted input purges correctly
//!   and embargoes nonsense. The order is not checked.
//! - Overlap is tested on **closed** intervals: a label that ends exactly when a test label
//!   starts is purged (Snippet 7.3 keeps it).
//! - `pct_embargo` is a fraction of the *whole* sample count, in `[0, 1)`, rounded up (Snippet
//!   7.3 truncates), so any positive value embargoes at least one sample. AFML suggests about
//!   0.01.
//! - Folds are contiguous and never shuffled; the first `n_samples % n_splits` folds hold one
//!   extra sample. All returned index lists are sorted.
//! - Purging can empty a training set when labels are long relative to the folds; check
//!   `train.len()` per fold.
//!
//! ```
//! use chrono::{Duration, NaiveDate};
//! use openquant::cross_validation::PurgedKFold;
//!
//! # fn main() -> Result<(), openquant::cross_validation::CrossValidationError> {
//! // Forty hourly labels, each resolved three hours after it starts.
//! let open = NaiveDate::from_ymd_opt(2024, 1, 2).unwrap().and_hms_opt(9, 0, 0).unwrap();
//! let info_sets: Vec<_> =
//!     (0..40).map(|i| (open + Duration::hours(i), open + Duration::hours(i + 3))).collect();
//!
//! // Third of five folds: test 16-23. Purging removes 13-15 and 24-26, and an embargo of
//! // ceil(0.15 * 40) = 6 samples starts where the purge ends: 27-32.
//! let (train, test) = &PurgedKFold::new(5, info_sets, 0.15)?.split(40)?[2];
//! assert_eq!(*test, (16..=23).collect::<Vec<usize>>());
//! assert_eq!(*train, (0..=12).chain(33..=39).collect::<Vec<usize>>());
//! # Ok(())
//! # }
//! ```
#![deny(missing_docs)]

use chrono::NaiveDateTime;
use itertools::Itertools;

/// Errors returned by the cross-validation splitters and diagnostics.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum CrossValidationError {
    /// `n_splits` is below 2 or above the number of samples.
    #[error("n_splits must be between 2 and the number of samples ({n_samples}), got {n_splits}")]
    InvalidSplits {
        /// The requested number of folds.
        n_splits: usize,
        /// The number of samples.
        n_samples: usize,
    },
    /// A required input is empty.
    #[error("{0} cannot be empty")]
    Empty(&'static str),
    /// `n_samples` passed to a split method differs from the number of information sets.
    #[error("Dataset length must match samples_info_sets")]
    DatasetLengthMismatch,
    /// `pct_embargo` is not a finite number in `[0, 1)`.
    #[error("pct_embargo must be finite and in [0, 1), got {pct_embargo}")]
    InvalidEmbargo {
        /// The rejected value.
        pct_embargo: f64,
    },
    /// An information set ends before it starts.
    #[error("samples_info_sets[{index}] ends before it starts")]
    InvalidInfoSet {
        /// Position of the first offending information set.
        index: usize,
    },
    /// `n_test_splits` for CPCV is not in `[1, n_splits)`.
    #[error("n_test_splits must be between 1 and {n_splits} - 1, got {n_test_splits}")]
    InvalidTestSplits {
        /// The requested number of test folds per split.
        n_test_splits: usize,
        /// The splitter's number of folds.
        n_splits: usize,
    },
    /// The number of CPCV splits, C(n_splits, n_test_splits), does not fit in `usize`.
    #[error("C({n_splits}, {n_test_splits}) CPCV splits overflow usize")]
    TooManySplits {
        /// The splitter's number of folds.
        n_splits: usize,
        /// The requested number of test folds per split.
        n_test_splits: usize,
    },
    /// A sample index is outside the information sets.
    #[error("sample index {index} is out of range for {len} information sets")]
    IndexOutOfRange {
        /// The first out-of-range index found (training indices are checked before test ones).
        index: usize,
        /// The number of information sets.
        len: usize,
    },
    /// An index in `splits` is not a row of `x` ([`ml_cross_val_score`]).
    #[error("split index {index} is out of range for {n_rows} rows")]
    SplitIndexOutOfRange {
        /// The first out-of-range index found (each split's training indices are checked
        /// before its test indices).
        index: usize,
        /// The number of rows of `x`.
        n_rows: usize,
    },
    /// `y` or `sample_weight` does not have one entry per row of `x` ([`ml_cross_val_score`]).
    #[error("{name} has {len} values, expected one per row of x ({expected})")]
    LengthMismatch {
        /// The argument whose length is wrong.
        name: &'static str,
        /// Its length.
        len: usize,
        /// The number of rows of `x`.
        expected: usize,
    },
    /// The classifier returned the wrong number of predictions for a test fold
    /// ([`ml_cross_val_score`]).
    #[error("classifier returned {got} predictions for {expected} test rows")]
    PredictionCountMismatch {
        /// The number of test rows.
        expected: usize,
        /// The number of predictions returned.
        got: usize,
    },
}

/// Checks the inputs of a scoring loop over `splits`: `y` and `sample_weight` have one entry
/// per row, and every split index is a row.
pub(crate) fn check_score_inputs(
    n_rows: usize,
    y_len: usize,
    sample_weight: Option<&[f64]>,
    splits: &[(Vec<usize>, Vec<usize>)],
) -> Result<(), CrossValidationError> {
    if y_len != n_rows {
        return Err(CrossValidationError::LengthMismatch {
            name: "y",
            len: y_len,
            expected: n_rows,
        });
    }
    if let Some(sw) = sample_weight {
        if sw.len() != n_rows {
            return Err(CrossValidationError::LengthMismatch {
                name: "sample_weight",
                len: sw.len(),
                expected: n_rows,
            });
        }
    }
    for (train, test) in splits {
        if let Some(&index) = train.iter().chain(test).find(|&&i| i >= n_rows) {
            return Err(CrossValidationError::SplitIndexOutOfRange { index, n_rows });
        }
    }
    Ok(())
}

/// Checks that a classifier returned one prediction per test row.
pub(crate) fn check_prediction_count(
    expected: usize,
    got: usize,
) -> Result<(), CrossValidationError> {
    if expected == got {
        Ok(())
    } else {
        Err(CrossValidationError::PredictionCountMismatch { expected, got })
    }
}

/// Minimal binary classifier interface used by [`ml_cross_val_score`].
///
/// Features are row-major: `x[i]` is the feature vector of sample `i`. Labels are `0.0` or
/// `1.0`, and probabilities are `P(y = 1)`.
///
/// ```
/// use openquant::cross_validation::SimpleClassifier;
///
/// /// Predicts the training base rate, whatever the features.
/// struct BaseRate(f64);
///
/// impl SimpleClassifier for BaseRate {
///     fn fit(&mut self, _x: &[Vec<f64>], y: &[f64], _sample_weight: Option<&[f64]>) {
///         self.0 = y.iter().sum::<f64>() / y.len() as f64;
///     }
///     fn predict_proba(&self, x: &[Vec<f64>]) -> Vec<f64> {
///         vec![self.0; x.len()]
///     }
/// }
///
/// let mut model = BaseRate(0.0);
/// model.fit(&vec![vec![0.0]; 4], &[1.0, 1.0, 1.0, 0.0], None);
/// assert_eq!(model.predict_proba(&[vec![9.0]]), vec![0.75]);
/// // The default `predict` thresholds the probability at 0.5.
/// assert_eq!(model.predict(&[vec![9.0]]), vec![1.0]);
/// ```
pub trait SimpleClassifier {
    /// Fits the model on the rows `x` with labels `y` (`0.0` or `1.0`) and optional per-row
    /// sample weights, all of the same length.
    fn fit(&mut self, x: &[Vec<f64>], y: &[f64], sample_weight: Option<&[f64]>);
    /// Returns `P(y = 1)` for each row of `x`, one value per row, in order.
    fn predict_proba(&self, x: &[Vec<f64>]) -> Vec<f64>;
    /// Returns hard `0.0`/`1.0` predictions: by default `1.0` where
    /// [`SimpleClassifier::predict_proba`] is at least 0.5.
    ///
    /// Every scorer uses it for the hard-label scores (accuracy and F1): [`ml_cross_val_score`],
    /// and through it single feature importance, and mean decrease accuracy. Override it to
    /// change the decision rule; log loss always uses `predict_proba`.
    fn predict(&self, x: &[Vec<f64>]) -> Vec<f64> {
        self.predict_proba(x).into_iter().map(|p| if p >= 0.5 { 1.0 } else { 0.0 }).collect()
    }
}

/// Scoring rule for [`ml_cross_val_score`]; every rule is higher-is-better.
///
/// Hard predictions come from [`SimpleClassifier::predict`] (by default `predict_proba >=
/// 0.5`). Labels are expected to be `0.0` or `1.0`.
/// AFML §7.5 and Chapter 9 recommend log loss for anything sized by probability.
#[derive(Clone, Copy)]
pub enum Scoring {
    /// Share of test samples whose hard prediction equals the label (compared within 1e-12,
    /// so a label other than `0.0` or `1.0` never matches).
    Accuracy,
    /// Negative mean log loss, `mean(y ln p + (1 - y) ln(1 - p))`, with `p` clipped to
    /// `[1e-15, 1 - 1e-15]`. Always `<= 0`.
    NegLogLoss,
    /// F1 score of the positive class (label `> 0.5`); `0.0` when precision and recall are
    /// both 0, including when there are no positive predictions. `NaN` on an empty test set,
    /// like the other rules.
    F1,
}

/// Fits `classifier` on each split's training rows and scores it on the test rows (AFML §7.5,
/// Snippet 7.4), returning one score per split, in order.
///
/// `x` holds one feature row per sample and `y` one `0.0`/`1.0` label per sample; `splits`
/// holds `(train_indices, test_indices)` into both, typically from [`PurgedKFold::split`].
/// The training slice of `sample_weight` is passed to [`SimpleClassifier::fit`].
///
/// Unlike Snippet 7.4, the weights are **not** passed to the metric: every test sample counts
/// equally. If the weights matter, compute the weighted score from `splits` yourself.
///
/// A test set of length 0 scores `NaN` under every rule, so an empty fold can't pass for a
/// real score of 0 in an average; `f64::is_nan` finds it.
///
/// # Errors
///
/// - [`CrossValidationError::LengthMismatch`] if `y` or `sample_weight` does not have one
///   entry per row of `x`.
/// - [`CrossValidationError::SplitIndexOutOfRange`] if an index in `splits` is not a row of
///   `x` (checked for every split before any fitting).
/// - [`CrossValidationError::PredictionCountMismatch`] if `predict` (accuracy, F1) or
///   `predict_proba` (log loss) does not return one value per test row.
///
/// ```
/// use chrono::{Duration, NaiveDate};
/// use openquant::cross_validation::{ml_cross_val_score, PurgedKFold, Scoring, SimpleClassifier};
///
/// struct BaseRate(f64);
///
/// impl SimpleClassifier for BaseRate {
///     fn fit(&mut self, _x: &[Vec<f64>], y: &[f64], _sample_weight: Option<&[f64]>) {
///         self.0 = y.iter().sum::<f64>() / y.len() as f64;
///     }
///     fn predict_proba(&self, x: &[Vec<f64>]) -> Vec<f64> {
///         vec![self.0; x.len()]
///     }
/// }
///
/// # fn main() -> Result<(), openquant::cross_validation::CrossValidationError> {
/// let open = NaiveDate::from_ymd_opt(2024, 1, 2).unwrap().and_hms_opt(9, 0, 0).unwrap();
/// let info_sets: Vec<_> =
///     (0..40).map(|i| (open + Duration::hours(i), open + Duration::hours(i + 3))).collect();
/// let x: Vec<Vec<f64>> = (0..40).map(|i| vec![f64::from(i)]).collect();
/// let y: Vec<f64> = (0..40).map(|i| f64::from(u8::from(i % 4 == 0))).collect();
/// let splits = PurgedKFold::new(5, info_sets, 0.0)?.split(40)?;
///
/// // The base rate is always below 0.5, so every hard prediction is 0: 3/4 are right.
/// let acc = ml_cross_val_score(&mut BaseRate(0.0), &x, &y, None, &splits, Scoring::Accuracy)?;
/// assert_eq!(acc, vec![0.75; 5]);
/// // With no positive predictions, F1 is 0.
/// let f1 = ml_cross_val_score(&mut BaseRate(0.0), &x, &y, None, &splits, Scoring::F1)?;
/// assert_eq!(f1, vec![0.0; 5]);
///
/// // Log loss is close to the entropy of a p = 0.25 coin.
/// let nll = ml_cross_val_score(&mut BaseRate(0.0), &x, &y, None, &splits, Scoring::NegLogLoss)?;
/// let entropy = -(0.25 * 0.25f64.ln() + 0.75 * 0.75f64.ln());
/// assert!(nll.iter().all(|s| (s + entropy).abs() < 0.002));
/// # Ok(())
/// # }
/// ```
pub fn ml_cross_val_score<C: SimpleClassifier>(
    classifier: &mut C,
    x: &[Vec<f64>],
    y: &[f64],
    sample_weight: Option<&[f64]>,
    splits: &[(Vec<usize>, Vec<usize>)],
    scoring: Scoring,
) -> Result<Vec<f64>, CrossValidationError> {
    check_score_inputs(x.len(), y.len(), sample_weight, splits)?;
    let mut scores = Vec::with_capacity(splits.len());
    for (train_idx, test_idx) in splits {
        let x_train: Vec<Vec<f64>> = train_idx.iter().map(|i| x[*i].clone()).collect();
        let y_train: Vec<f64> = train_idx.iter().map(|i| y[*i]).collect();
        let sw_train: Option<Vec<f64>> =
            sample_weight.map(|sw| train_idx.iter().map(|i| sw[*i]).collect());

        classifier.fit(&x_train, &y_train, sw_train.as_deref());
        let x_test: Vec<Vec<f64>> = test_idx.iter().map(|i| x[*i].clone()).collect();
        let y_test: Vec<f64> = test_idx.iter().map(|i| y[*i]).collect();
        // Hard-label scores use `predict`, as mean decrease accuracy does, so an overridden
        // decision rule is honoured; log loss needs the probabilities.
        let (preds, probs) = match scoring {
            Scoring::NegLogLoss => (Vec::new(), classifier.predict_proba(&x_test)),
            Scoring::Accuracy | Scoring::F1 => (classifier.predict(&x_test), Vec::new()),
        };
        check_prediction_count(y_test.len(), preds.len().max(probs.len()))?;

        let score = match scoring {
            Scoring::Accuracy => {
                let correct = preds
                    .iter()
                    .zip(y_test.iter())
                    .filter(|(p, y_true)| (**p - *y_true).abs() < 1e-12)
                    .count();
                correct as f64 / y_test.len() as f64
            }
            Scoring::NegLogLoss => {
                let eps = 1e-15;
                let mut loss = 0.0;
                for (p, y_true) in probs.iter().zip(y_test.iter()) {
                    let p_clip = p.max(eps).min(1.0 - eps);
                    loss += -(*y_true * p_clip.ln() + (1.0 - *y_true) * (1.0 - p_clip).ln());
                }
                -(loss / y_test.len() as f64)
            }
            Scoring::F1 if y_test.is_empty() => f64::NAN,
            Scoring::F1 => {
                let mut tp = 0.0;
                let mut fp = 0.0;
                let mut fnn = 0.0;
                for (p, y_true) in preds.iter().zip(y_test.iter()) {
                    let p_pos = *p > 0.5;
                    let y_pos = *y_true > 0.5;
                    if p_pos && y_pos {
                        tp += 1.0;
                    } else if p_pos && !y_pos {
                        fp += 1.0;
                    } else if !p_pos && y_pos {
                        fnn += 1.0;
                    }
                }
                let precision = if tp + fp > 0.0 { tp / (tp + fp) } else { 0.0 };
                let recall = if tp + fnn > 0.0 { tp / (tp + fnn) } else { 0.0 };
                if precision + recall > 0.0 {
                    2.0 * precision * recall / (precision + recall)
                } else {
                    0.0
                }
            }
        };
        scores.push(score);
    }
    Ok(scores)
}

/// Purges training information sets that overlap any test window (AFML §7.4.1, Snippet 7.1).
///
/// Returns the `(start, end)` pairs of `info_sets`, in their original order, that neither
/// start inside, end inside, nor envelop any `(test_start, test_end)` window of `test_times`
/// (all bounds inclusive). This is the purge alone, on timestamps: it applies no embargo and
/// nothing else in the crate calls it. Use it when building your own splits, for instance
/// several disjoint test blocks at once.
///
/// ```
/// use chrono::{Duration, NaiveDate};
/// use openquant::cross_validation::ml_get_train_times;
///
/// let day = |d: i64| NaiveDate::from_ymd_opt(2024, 1, 1).unwrap().and_hms_opt(0, 0, 0).unwrap()
///     + Duration::days(d);
/// let info_sets: Vec<_> = (0..10).map(|d| (day(d), day(d + 2))).collect();
///
/// // Testing days 4-5 purges every label that touches them: those starting on days 2-5.
/// let train = ml_get_train_times(&info_sets, &[(day(4), day(5))]);
/// let starts: Vec<_> = train.iter().map(|(s, _)| (*s - day(0)).num_days()).collect();
/// assert_eq!(starts, vec![0, 1, 6, 7, 8, 9]);
/// ```
pub fn ml_get_train_times(
    info_sets: &[(NaiveDateTime, NaiveDateTime)],
    test_times: &[(NaiveDateTime, NaiveDateTime)],
) -> Vec<(NaiveDateTime, NaiveDateTime)> {
    let mut out = Vec::new();
    for (start, end) in info_sets {
        let mut keep = true;
        for (test_start, test_end) in test_times {
            let start_in = *start >= *test_start && *start <= *test_end;
            let end_in = *end >= *test_start && *end <= *test_end;
            let envelop = *start <= *test_start && *end >= *test_end;
            if start_in || end_in || envelop {
                keep = false;
                break;
            }
        }
        if keep {
            out.push((*start, *end));
        }
    }
    out
}

/// One cross-validation split: `(train_indices, test_indices)`.
pub type TrainTestSplit = (Vec<usize>, Vec<usize>);

/// Why each non-training sample of a [`PurgedSplit`] was left out of training.
///
/// The training samples are exactly those that are neither test, purged nor embargoed. A
/// sample can be both purged and embargoed. All index lists are sorted.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PurgedSplitDiagnostics {
    /// Position of the split in the returned list.
    pub split_id: usize,
    /// The test set as sorted, disjoint, half-open index ranges `[start, stop)`.
    pub test_ranges: Vec<(usize, usize)>,
    /// Non-test samples whose information set overlaps a test block's window (AFML §7.4.1).
    pub purged_indices: Vec<usize>,
    /// Non-test samples inside an embargo window (§7.4.2), whether or not also purged. Each
    /// window follows a test block and starts where that block's purge ends (Snippet 7.3).
    pub embargo_indices: Vec<usize>,
    /// Training samples whose information set still overlaps some test sample's. Purging
    /// guarantees 0; it is reported so callers can assert it.
    pub overlap_count_after_purge: usize,
}

/// A purged split with the diagnostics that explain it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PurgedSplit {
    /// Sorted training indices.
    pub train_indices: Vec<usize>,
    /// Sorted test indices.
    pub test_indices: Vec<usize>,
    /// Which samples were removed, and why.
    pub diagnostics: PurgedSplitDiagnostics,
}

/// One combinatorial purged cross-validation split (AFML §12.4): `n_test_splits` of the
/// `n_splits` folds are tested together, and the rest are purged, embargoed and trained on.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CpcvSplit {
    /// Position of the split in lexicographic order of `test_fold_ids`.
    pub split_id: usize,
    /// The folds tested in this split, ascending.
    pub test_fold_ids: Vec<usize>,
    /// The purged split; `split.diagnostics.split_id == split_id`.
    pub split: PurgedSplit,
}

/// One CPCV backtest path (AFML §12.4): for every fold, the split whose predictions for that
/// fold the path uses.
///
/// Taking, fold by fold, the predictions that split `split_for_fold[g]` made for fold `g`
/// gives one out-of-sample prediction for every sample.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CpcvPath {
    /// Position of the path, from 0 to φ − 1.
    pub path_id: usize,
    /// `split_for_fold[g]` is the `split_id` of a [`CpcvSplit`] that tests fold `g`.
    pub split_for_fold: Vec<usize>,
}

/// Purged k-fold cross-validation with an embargo (AFML §7.4.3, Snippet 7.3), and its
/// combinatorial extension (AFML §12.4).
///
/// Holds one `(start, end)` information set per sample, in time order, the number of
/// contiguous folds, and the embargo fraction. See the [module docs](crate::cross_validation) for the purge and
/// embargo rules and an example.
#[derive(Debug, Clone)]
pub struct PurgedKFold {
    n_splits: usize,
    samples_info_sets: Vec<(NaiveDateTime, NaiveDateTime)>,
    pct_embargo: f64,
}

impl PurgedKFold {
    /// Builds a splitter over one `(start, end)` information set per sample, in time order.
    ///
    /// # Errors
    /// [`CrossValidationError::Empty`] when there are no samples,
    /// [`CrossValidationError::InvalidSplits`] unless `2 <= n_splits <= n_samples`,
    /// [`CrossValidationError::InvalidEmbargo`] unless `pct_embargo` is finite and in `[0, 1)`,
    /// and [`CrossValidationError::InvalidInfoSet`] when an information set ends before it
    /// starts. The time order of the information sets is not checked.
    pub fn new(
        n_splits: usize,
        samples_info_sets: Vec<(NaiveDateTime, NaiveDateTime)>,
        pct_embargo: f64,
    ) -> Result<Self, CrossValidationError> {
        if samples_info_sets.is_empty() {
            return Err(CrossValidationError::Empty("samples_info_sets"));
        }
        if n_splits < 2 || n_splits > samples_info_sets.len() {
            return Err(CrossValidationError::InvalidSplits {
                n_splits,
                n_samples: samples_info_sets.len(),
            });
        }
        if !pct_embargo.is_finite() || !(0.0..1.0).contains(&pct_embargo) {
            return Err(CrossValidationError::InvalidEmbargo { pct_embargo });
        }
        if let Some(index) = samples_info_sets.iter().position(|(start, end)| start > end) {
            return Err(CrossValidationError::InvalidInfoSet { index });
        }
        Ok(Self { n_splits, samples_info_sets, pct_embargo })
    }

    /// Splits the samples into `n_splits` contiguous test folds and returns
    /// `(train_indices, test_indices)` for each.
    ///
    /// # Errors
    /// [`CrossValidationError::DatasetLengthMismatch`] when `n_samples` differs from the
    /// number of information sets.
    pub fn split(&self, n_samples: usize) -> Result<Vec<TrainTestSplit>, CrossValidationError> {
        Ok(self
            .split_with_diagnostics(n_samples)?
            .into_iter()
            .map(|s| (s.train_indices, s.test_indices))
            .collect())
    }

    /// The folds of [`PurgedKFold::split`], with the purged and embargoed indices of each.
    ///
    /// # Errors
    /// [`CrossValidationError::DatasetLengthMismatch`] when `n_samples` differs from the
    /// number of information sets.
    ///
    /// ```
    /// use chrono::{Duration, NaiveDate};
    /// use openquant::cross_validation::PurgedKFold;
    ///
    /// # fn main() -> Result<(), openquant::cross_validation::CrossValidationError> {
    /// let open = NaiveDate::from_ymd_opt(2024, 1, 2).unwrap().and_hms_opt(9, 0, 0).unwrap();
    /// let info_sets: Vec<_> =
    ///     (0..40).map(|i| (open + Duration::hours(i), open + Duration::hours(i + 3))).collect();
    ///
    /// let fold = &PurgedKFold::new(5, info_sets, 0.15)?.split_with_diagnostics(40)?[2];
    /// assert_eq!(fold.diagnostics.test_ranges, vec![(16, 24)]);
    /// assert_eq!(fold.diagnostics.purged_indices, vec![13, 14, 15, 24, 25, 26]);
    /// // ceil(0.15 * 40) = 6 samples after the purged zone, none before the fold.
    /// assert_eq!(fold.diagnostics.embargo_indices, (27..=32).collect::<Vec<usize>>());
    /// assert_eq!(fold.diagnostics.overlap_count_after_purge, 0);
    /// # Ok(())
    /// # }
    /// ```
    pub fn split_with_diagnostics(
        &self,
        n_samples: usize,
    ) -> Result<Vec<PurgedSplit>, CrossValidationError> {
        self.check_n_samples(n_samples)?;
        Ok(contiguous_fold_bounds(n_samples, self.n_splits)
            .into_iter()
            .enumerate()
            .map(|(split_id, fold)| self.build_split(split_id, &[fold]))
            .collect())
    }

    /// Combinatorial purged cross-validation splits (AFML §12.4): one split for each of the
    /// C(`n_splits`, `n_test_splits`) ways to choose the test folds, in lexicographic order.
    ///
    /// Each contiguous run of test folds is purged and embargoed exactly as a
    /// [`PurgedKFold::split`] fold is.
    ///
    /// # Errors
    /// [`CrossValidationError::DatasetLengthMismatch`] when `n_samples` differs from the
    /// number of information sets, [`CrossValidationError::InvalidTestSplits`] unless
    /// `1 <= n_test_splits < n_splits`, and [`CrossValidationError::TooManySplits`] when the
    /// split count overflows `usize`.
    ///
    /// The splits are materialised eagerly, each with its own index vectors: `n_splits = 10`,
    /// `n_test_splits = 5` is already 252 splits.
    ///
    /// ```
    /// use chrono::{Duration, NaiveDate};
    /// use openquant::cross_validation::PurgedKFold;
    ///
    /// # fn main() -> Result<(), openquant::cross_validation::CrossValidationError> {
    /// let open = NaiveDate::from_ymd_opt(2024, 1, 2).unwrap().and_hms_opt(9, 0, 0).unwrap();
    /// let info_sets: Vec<_> =
    ///     (0..40).map(|i| (open + Duration::hours(i), open + Duration::hours(i + 3))).collect();
    /// let cv = PurgedKFold::new(5, info_sets, 0.0)?;
    ///
    /// // C(5, 2) = 10 splits in lexicographic order of the tested folds.
    /// let splits = cv.cpcv_splits(40, 2)?;
    /// assert_eq!(splits.len(), 10);
    /// assert_eq!(splits[1].test_fold_ids, vec![0, 2]);
    /// // Folds 0 and 2 are separate blocks: samples 0-7 and 16-23.
    /// assert_eq!(splits[1].split.diagnostics.test_ranges, vec![(0, 8), (16, 24)]);
    /// // With k = 1 the splits are the k-fold ones.
    /// assert_eq!(cv.cpcv_splits(40, 1)?[2].split, cv.split_with_diagnostics(40)?[2]);
    /// # Ok(())
    /// # }
    /// ```
    pub fn cpcv_splits(
        &self,
        n_samples: usize,
        n_test_splits: usize,
    ) -> Result<Vec<CpcvSplit>, CrossValidationError> {
        self.check_n_samples(n_samples)?;
        let n_combinations = self.check_test_splits(n_test_splits)?;
        let folds = contiguous_fold_bounds(n_samples, self.n_splits);

        let mut out = Vec::with_capacity(n_combinations);
        for (split_id, test_fold_ids) in (0..self.n_splits).combinations(n_test_splits).enumerate()
        {
            let blocks = merge_adjacent(test_fold_ids.iter().map(|&g| folds[g]));
            let split = self.build_split(split_id, &blocks);
            out.push(CpcvSplit { split_id, test_fold_ids, split });
        }
        Ok(out)
    }

    /// The φ = `n_test_splits` / `n_splits` · C(`n_splits`, `n_test_splits`) backtest paths
    /// (AFML §12.4) that [`PurgedKFold::cpcv_splits`] produces.
    ///
    /// Each fold is tested in exactly φ splits. Path `j` takes, for every fold, the `j`-th
    /// split (in `split_id` order) that tests it, as in AFML §12.4.
    ///
    /// # Errors
    /// [`CrossValidationError::InvalidTestSplits`] unless `1 <= n_test_splits < n_splits`, and
    /// [`CrossValidationError::TooManySplits`] when the split count overflows `usize`.
    ///
    /// ```
    /// use chrono::{Duration, NaiveDate};
    /// use openquant::cross_validation::PurgedKFold;
    ///
    /// # fn main() -> Result<(), openquant::cross_validation::CrossValidationError> {
    /// let open = NaiveDate::from_ymd_opt(2024, 1, 2).unwrap().and_hms_opt(9, 0, 0).unwrap();
    /// let info_sets: Vec<_> =
    ///     (0..40).map(|i| (open + Duration::hours(i), open + Duration::hours(i + 3))).collect();
    ///
    /// // φ[5, 2] = 2/5 · C(5, 2) = 4 paths.
    /// let paths = PurgedKFold::new(5, info_sets, 0.0)?.cpcv_paths(2)?;
    /// assert_eq!(paths.len(), 4);
    /// // Path 0 takes folds 0 and 1 from split 0 ({0, 1}), fold 2 from split 1 ({0, 2}), ...
    /// assert_eq!(paths[0].split_for_fold, vec![0, 0, 1, 2, 3]);
    /// assert_eq!(paths[3].split_for_fold, vec![3, 6, 8, 9, 9]);
    /// # Ok(())
    /// # }
    /// ```
    pub fn cpcv_paths(&self, n_test_splits: usize) -> Result<Vec<CpcvPath>, CrossValidationError> {
        self.check_test_splits(n_test_splits)?;
        let mut splits_testing: Vec<Vec<usize>> = vec![Vec::new(); self.n_splits];
        for (split_id, test_fold_ids) in (0..self.n_splits).combinations(n_test_splits).enumerate()
        {
            for g in test_fold_ids {
                splits_testing[g].push(split_id);
            }
        }
        // Every fold is in C(n_splits - 1, n_test_splits - 1) = φ of the combinations.
        let n_paths = splits_testing[0].len();
        Ok((0..n_paths)
            .map(|path_id| CpcvPath {
                path_id,
                split_for_fold: splits_testing.iter().map(|ids| ids[path_id]).collect(),
            })
            .collect())
    }

    fn check_n_samples(&self, n_samples: usize) -> Result<(), CrossValidationError> {
        if n_samples != self.samples_info_sets.len() {
            return Err(CrossValidationError::DatasetLengthMismatch);
        }
        Ok(())
    }

    /// Validates `n_test_splits` and returns C(n_splits, n_test_splits).
    fn check_test_splits(&self, n_test_splits: usize) -> Result<usize, CrossValidationError> {
        if n_test_splits == 0 || n_test_splits >= self.n_splits {
            return Err(CrossValidationError::InvalidTestSplits {
                n_test_splits,
                n_splits: self.n_splits,
            });
        }
        n_choose_k(self.n_splits, n_test_splits)
            .ok_or(CrossValidationError::TooManySplits { n_splits: self.n_splits, n_test_splits })
    }

    /// Purges and embargoes around each test block. Blocks are non-empty, disjoint,
    /// ascending and non-adjacent half-open ranges.
    fn build_split(&self, split_id: usize, test_blocks: &[(usize, usize)]) -> PurgedSplit {
        let info = &self.samples_info_sets;
        let n = info.len();
        let mut test = vec![false; n];
        for &(start, stop) in test_blocks {
            test[start..stop].fill(true);
        }

        // Purge against the window each test block covers: from its first label's start to
        // the latest end among its labels (AFML snippet 7.3). With variable-length labels
        // that is not necessarily the last label's end.
        let mut purged = vec![false; n];
        for &(start, stop) in test_blocks {
            let window_start = info[start].0;
            let window_end = info[start..stop].iter().fold(info[start].1, |m, (_, e)| m.max(*e));
            for (i, span) in info.iter().enumerate() {
                if !test[i] && intervals_overlap(*span, (window_start, window_end)) {
                    purged[i] = true;
                }
            }
        }

        // Embargo after each block only, starting where its purge ends (AFML Snippet 7.3).
        let mut embargoed = vec![false; n];
        for window in embargo_windows(info, &test, embargo_width(self.pct_embargo, n)) {
            for i in window {
                embargoed[i] |= !test[i];
            }
        }

        let where_true = |mask: &[bool]| -> Vec<usize> {
            mask.iter().enumerate().filter(|(_, m)| **m).map(|(i, _)| i).collect()
        };
        let train_indices: Vec<usize> =
            (0..n).filter(|&i| !test[i] && !purged[i] && !embargoed[i]).collect();
        let test_indices = where_true(&test);
        let overlap_count_after_purge = train_indices
            .iter()
            .filter(|&&tr| test_indices.iter().any(|&te| intervals_overlap(info[tr], info[te])))
            .count();

        PurgedSplit {
            diagnostics: PurgedSplitDiagnostics {
                split_id,
                test_ranges: test_blocks.to_vec(),
                purged_indices: where_true(&purged),
                embargo_indices: where_true(&embargoed),
                overlap_count_after_purge,
            },
            train_indices,
            test_indices,
        }
    }
}

/// Unpurged k-fold: `n_splits` contiguous test folds, and every other sample trains.
///
/// This is the baseline AFML §7.3 warns against. It exists to measure leakage, for example
/// with [`count_train_test_overlaps`], not to validate models.
///
/// # Errors
/// [`CrossValidationError::InvalidSplits`] unless `2 <= n_splits <= n_samples`.
///
/// ```
/// use chrono::{Duration, NaiveDate};
/// use openquant::cross_validation::{count_train_test_overlaps, naive_kfold_splits};
///
/// # fn main() -> Result<(), openquant::cross_validation::CrossValidationError> {
/// let open = NaiveDate::from_ymd_opt(2024, 1, 2).unwrap().and_hms_opt(9, 0, 0).unwrap();
/// let info_sets: Vec<_> =
///     (0..40).map(|i| (open + Duration::hours(i), open + Duration::hours(i + 3))).collect();
///
/// // Unpurged, the third fold trains on 13-15 and 24-26, whose labels overlap the test labels.
/// let (train, test) = &naive_kfold_splits(40, 5)?[2];
/// assert_eq!(*test, (16..24).collect::<Vec<usize>>());
/// assert_eq!(train.len(), 32);
/// assert_eq!(count_train_test_overlaps(&info_sets, train, test)?, 6);
/// # Ok(())
/// # }
/// ```
pub fn naive_kfold_splits(
    n_samples: usize,
    n_splits: usize,
) -> Result<Vec<TrainTestSplit>, CrossValidationError> {
    if n_splits < 2 || n_splits > n_samples {
        return Err(CrossValidationError::InvalidSplits { n_splits, n_samples });
    }
    Ok(contiguous_fold_bounds(n_samples, n_splits)
        .into_iter()
        .map(|(start, stop)| {
            let train = (0..n_samples).filter(|i| *i < start || *i >= stop).collect();
            (train, (start..stop).collect())
        })
        .collect())
}

/// Counts the training samples whose information set intersects (as closed intervals) the
/// information set of at least one test sample.
///
/// Use it with [`naive_kfold_splits`] to measure the leak that purging removes (AFML §7.3); on
/// a [`PurgedKFold`] split it is always 0.
///
/// # Errors
/// [`CrossValidationError::IndexOutOfRange`] when an index is not a position in `info_sets`.
///
/// ```
/// use chrono::{Duration, NaiveDate};
/// use openquant::cross_validation::{count_train_test_overlaps, CrossValidationError};
///
/// let t = |h: i64| NaiveDate::from_ymd_opt(2024, 1, 2).unwrap().and_hms_opt(0, 0, 0).unwrap()
///     + Duration::hours(h);
/// let info_sets = vec![(t(0), t(2)), (t(1), t(3)), (t(4), t(5))];
///
/// // Sample 0 overlaps test sample 1 (closed intervals); sample 2 does not.
/// assert_eq!(count_train_test_overlaps(&info_sets, &[0, 2], &[1]), Ok(1));
/// assert_eq!(
///     count_train_test_overlaps(&info_sets, &[0], &[3]),
///     Err(CrossValidationError::IndexOutOfRange { index: 3, len: 3 })
/// );
/// ```
pub fn count_train_test_overlaps(
    info_sets: &[(NaiveDateTime, NaiveDateTime)],
    train_indices: &[usize],
    test_indices: &[usize],
) -> Result<usize, CrossValidationError> {
    let len = info_sets.len();
    if let Some(&index) = train_indices.iter().chain(test_indices).find(|&&i| i >= len) {
        return Err(CrossValidationError::IndexOutOfRange { index, len });
    }
    Ok(train_indices
        .iter()
        .filter(|&&tr| {
            test_indices.iter().any(|&te| intervals_overlap(info_sets[tr], info_sets[te]))
        })
        .count())
}

/// The embargo width h = ⌈`pct_embargo` · `n_samples`⌉ (AFML §7.4.2).
///
/// Snippet 7.3 truncates (`int(n * pct)`); rounding up guarantees that any positive
/// `pct_embargo` embargoes at least one sample.
pub(crate) fn embargo_width(pct_embargo: f64, n_samples: usize) -> usize {
    (pct_embargo * n_samples as f64).ceil() as usize
}

/// The embargo windows of a split (AFML §7.4.2, Snippet 7.3), one per test block.
///
/// A test block is a maximal run of adjacent `true` entries in `test_mask`. Only samples that
/// follow a block are embargoed: only later features can contain prices from the test window.
/// The window starts where the purge ends, at the first sample after the block whose
/// information set starts after the latest end among the block's information sets (Snippet
/// 7.3's `maxT1Idx`), and covers the next `width` samples, clipped to the sample count.
///
/// The windows may include test samples of a later block or samples another block purged;
/// callers decide what to do with those. Shared by [`PurgedKFold`] and
/// [`crate::backtesting_engine`], so both split the same way.
pub(crate) fn embargo_windows(
    info_sets: &[(NaiveDateTime, NaiveDateTime)],
    test_mask: &[bool],
    width: usize,
) -> Vec<std::ops::Range<usize>> {
    let n = test_mask.len();
    let mut windows = Vec::new();
    if width == 0 {
        return windows;
    }
    let mut i = 0;
    while i < n {
        if !test_mask[i] {
            i += 1;
            continue;
        }
        let mut block_end = info_sets[i].1;
        while i < n && test_mask[i] {
            block_end = block_end.max(info_sets[i].1);
            i += 1;
        }
        let mut resume = i;
        while resume < n && info_sets[resume].0 <= block_end {
            resume += 1;
        }
        windows.push(resume..resume.saturating_add(width).min(n));
    }
    windows
}

fn intervals_overlap(a: (NaiveDateTime, NaiveDateTime), b: (NaiveDateTime, NaiveDateTime)) -> bool {
    a.0 <= b.1 && b.0 <= a.1
}

/// `[start, stop)` of each of `n_splits` contiguous folds; the first `n_samples % n_splits`
/// folds hold one extra sample.
fn contiguous_fold_bounds(n_samples: usize, n_splits: usize) -> Vec<(usize, usize)> {
    let mut bounds = Vec::with_capacity(n_splits);
    let mut current = 0;
    for fold in 0..n_splits {
        let size = n_samples / n_splits + usize::from(fold < n_samples % n_splits);
        bounds.push((current, current + size));
        current += size;
    }
    bounds
}

/// Joins ascending half-open ranges that touch into one.
fn merge_adjacent(ranges: impl Iterator<Item = (usize, usize)>) -> Vec<(usize, usize)> {
    let mut out: Vec<(usize, usize)> = Vec::new();
    for (start, stop) in ranges {
        match out.last_mut() {
            Some(last) if last.1 == start => last.1 = stop,
            _ => out.push((start, stop)),
        }
    }
    out
}

/// C(n, k) for `k <= n`, or `None` on overflow.
fn n_choose_k(n: usize, k: usize) -> Option<usize> {
    let k = k.min(n - k);
    let mut acc: usize = 1;
    for i in 0..k {
        // acc * (n - i) is divisible by i + 1 at every step.
        acc = acc.checked_mul(n - i)? / (i + 1);
    }
    Some(acc)
}

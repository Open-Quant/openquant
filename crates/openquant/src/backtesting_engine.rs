//! Walk-forward, purged cross-validation and combinatorial purged cross-validation (CPCV)
//! backtests (AFML Chapters 11 and 12).
//!
//! The engine builds the splits for three backtest modes, purges and embargoes them, runs a
//! caller-supplied evaluator on each, and summarises the out-of-sample returns it gets back.
//! It does not fit models, size positions or see prices.
//!
//! - [`run_walk_forward`] (§12.2): expanding-window training before each test block; one path
//!   over the later part of the sample.
//! - [`run_cross_validation`] (§12.3): contiguous test folds, training on everything else; one
//!   path over the whole sample.
//! - [`run_cpcv`] (§12.4): every choice of `test_groups` of `n_groups` contiguous groups;
//!   φ[N, k] = k/N · C(N, k) full-length paths ([`cpcv_path_count`]), and so a distribution of
//!   performance rather than one number.
//!
//! Every run also requires a [`BacktestRunConfig`]: a provenance note, the number of trials
//! behind this configuration, and five [`BacktestSafeguards`] strings recording how the run
//! controls the pitfalls of §11.4. The engine only checks they are non-blank; it returns them
//! with the results in [`BacktestDiagnostics`], where `trials_count` is what a deflated
//! Sharpe ratio needs.
//!
//! Conventions:
//!
//! - One sample per entry of [`BacktestData::returns`] and [`BacktestData::label_spans`], in
//!   time order. `label_spans[i] = (start, end)` is the span of sample `i`'s label. `returns`
//!   is only validated and used to count samples; performance comes from the evaluator.
//! - The evaluator receives a [`SplitDefinition`] and must return **one out-of-sample return
//!   per test index, in order**. Only CPCV checks the length; the other modes only summarise.
//! - **Purging** (§7.4.1): a training sample is removed if its label span overlaps (closed
//!   intervals) the span of *any* test sample, compared pair by pair.
//! - **Embargo** (§7.4.2, Snippet 7.3): `h = ceil(pct_embargo * n_samples)` further training
//!   samples are removed after each test block, counting from where the purge ends. Samples
//!   before a block are never embargoed, so the embargo removes nothing in walk-forward mode.
//!   In CPCV, adjacent test groups form one block. The rule is shared with
//!   [`crate::cross_validation::PurgedKFold`].
//! - `sharpe` in [`FoldPerformance`] and [`CpcvPathPerformance`] is a **t-statistic**,
//!   `mean / std * sqrt(n)` with the sample standard deviation and `n` the number of returns;
//!   it is not annualised and grows with `n`. It is 0 when the deviation is 0.
//! - CPCV paths share most of their returns, so their spread is not a confidence interval.
//! - Cost grows quickly: purging compares every training span with every test span, and CPCV
//!   runs C(N, k) splits.
//!
//! ```
//! use chrono::{Duration, NaiveDate};
//! use openquant::backtesting_engine::{
//!     run_cpcv, BacktestData, BacktestError, BacktestRunConfig, BacktestSafeguards,
//!     CpcvConfig, SplitDefinition,
//! };
//!
//! # fn main() -> Result<(), BacktestError> {
//! // Twelve daily samples; each label spans one day, so neighbours share an endpoint.
//! let open = NaiveDate::from_ymd_opt(2024, 1, 2).unwrap().and_hms_opt(0, 0, 0).unwrap();
//! let returns: Vec<f64> = (0..12).map(|i| [0.01, 0.02, 0.03][i % 3]).collect();
//! let label_spans: Vec<_> =
//!     (0..12).map(|i| (open + Duration::days(i), open + Duration::days(i + 1))).collect();
//! let data = BacktestData { returns: returns.clone(), label_spans };
//!
//! let note = |s: &str| s.to_string();
//! let run = BacktestRunConfig {
//!     mode_provenance: note("docs example"),
//!     trials_count: 1,
//!     safeguards: BacktestSafeguards {
//!         survivorship_bias_control: note("synthetic"),
//!         look_ahead_control: note("synthetic"),
//!         data_mining_control: note("one configuration"),
//!         cost_assumption: note("none"),
//!         multiple_testing_control: note("single trial"),
//!     },
//! };
//!
//! // A "strategy" whose out-of-sample return is the sample's own return.
//! let evaluator = |split: &SplitDefinition| -> Result<Vec<f64>, BacktestError> {
//!     Ok(split.test_indices.iter().map(|&i| returns[i]).collect())
//! };
//!
//! // Four groups of three, tested two at a time: C(4, 2) = 6 splits, 2/4 * 6 = 3 paths.
//! let config = CpcvConfig { n_groups: 4, test_groups: 2, pct_embargo: 0.05 };
//! let result = run_cpcv(&data, &run, &config, evaluator)?;
//! assert_eq!((result.splits.len(), result.path_count), (6, 3));
//!
//! // Split 0 tests groups 0 and 1 (samples 0-5). Sample 6 shares an endpoint with sample 5 and
//! // is purged; the embargo of ceil(0.05 * 12) = 1 sample then removes 7.
//! let split = &result.splits[0];
//! assert_eq!(split.test_groups, vec![0, 1]);
//! assert_eq!((split.purged_count, split.embargo_count), (1, 1));
//! assert_eq!(split.train_indices, vec![8, 9, 10, 11]);
//!
//! // Each path covers every sample once, so here each has the series' mean return.
//! assert_eq!(result.path_assignments[0].split_for_group, vec![0, 0, 1, 2]);
//! for path in &result.path_distribution {
//!     assert_eq!(path.observations, 12);
//!     assert!((path.mean_return - 0.02).abs() < 1e-12);
//! }
//! # Ok(())
//! # }
//! ```
#![deny(missing_docs)]

use crate::cross_validation::{embargo_width, embargo_windows};
use chrono::NaiveDateTime;
use std::collections::HashMap;

/// Errors returned by the backtest runners.
///
/// Several variants guard internal invariants of CPCV path construction and are not reachable
/// through the public API; they say so. An error returned by the evaluator is passed through
/// unchanged, whatever its variant.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum BacktestError {
    /// A failure reported by the caller's split evaluator, passed through verbatim.
    #[error("{0}")]
    Evaluator(String),
    /// A label span in [`BacktestData::label_spans`] ends before it starts.
    #[error("label span end must be >= start")]
    LabelSpanEndBeforeStart,
    /// A CPCV group is tested by a number of splits other than φ. Internal invariant; not
    /// reachable through the public API.
    #[error("group {group} has {found} occurrences, expected {expected}")]
    GroupOccurrences {
        /// The group index.
        group: usize,
        /// How many splits test it.
        found: usize,
        /// The expected count, φ.
        expected: usize,
    },
    /// The named input is empty (`"returns"`) or blank after trimming whitespace
    /// (`"mode_provenance"` or a [`BacktestSafeguards`] field name).
    #[error("{0} cannot be empty")]
    Empty(&'static str),
    /// `returns` and `label_spans` have different lengths.
    #[error("returns and label_spans length mismatch")]
    ReturnsLabelSpansLengthMismatch,
    /// A parameter violates its requirement, for example `pct_embargo` outside `[0, 1)` or a
    /// zero `test_size`.
    #[error("{name} must be {requirement}")]
    Invalid {
        /// The parameter's name.
        name: &'static str,
        /// The violated requirement, such as `"> 0"`.
        requirement: &'static str,
    },
    /// Purging left a walk-forward split with no training samples.
    #[error("walk-forward produced an empty train split")]
    EmptyWalkForwardTrainSplit,
    /// `min_train_size >= n_samples`, so walk-forward has nothing to test.
    #[error("walk-forward produced no splits")]
    NoWalkForwardSplits,
    /// Purging and embargo left a cross-validation split with no training samples.
    #[error("cross-validation produced an empty train split")]
    EmptyCrossValidationTrainSplit,
    /// Purging and embargo left a CPCV split with no training samples.
    #[error("CPCV produced an empty train split")]
    EmptyCpcvTrainSplit,
    /// The evaluator returned no values for a split.
    #[error("split evaluator returned empty returns")]
    EmptySplitReturns,
    /// The evaluator returned a NaN or infinite value.
    #[error("split evaluator returned non-finite returns")]
    NonFiniteSplitReturns,
    /// `test_groups >= n_groups` in CPCV.
    #[error("test_groups must be < n_groups")]
    TestGroupsNotBelowGroups,
    /// More folds or groups were requested than there are samples.
    #[error("n_folds cannot exceed number of samples")]
    TooManyFolds,
    /// `k > n` in a binomial coefficient. Internal invariant; not reachable through the
    /// public API.
    #[error("k cannot exceed n")]
    CombinationSizeTooLarge,
    /// C(n_groups, test_groups) fits in `u128` but not in `usize`.
    #[error("combination count overflowed usize")]
    CombinationCountOverflow,
    /// A split tests a group outside `0..n_groups`. Internal invariant; not reachable through
    /// the public API.
    #[error("split references out-of-range test group")]
    TestGroupOutOfRange,
    /// A path assignment does not have one split per group. Internal invariant; not reachable
    /// through the public API.
    #[error("invalid path assignment length")]
    InvalidPathAssignmentLength,
    /// A path refers to a split id that does not exist. Internal invariant; not reachable
    /// through the public API.
    #[error("path references unknown split")]
    UnknownSplitInPath,
    /// A path takes a group from a split that does not test it. Internal invariant; not
    /// reachable through the public API.
    #[error("path assignment references split not containing group")]
    PathSplitMissingGroup,
    /// A split's returns are missing when paths are assembled. Internal invariant; not
    /// reachable through the public API.
    #[error("missing split returns for CPCV path construction")]
    MissingSplitReturns,
    /// In CPCV, the evaluator returned a number of values different from the split's
    /// `test_indices.len()`.
    #[error("split return count must match split test indices length")]
    SplitReturnCountMismatch,
    /// A group's test index has no return when paths are assembled. Internal invariant; not
    /// reachable through the public API.
    #[error("split returns missing group test index")]
    MissingGroupTestIndex,
}

/// How a backtest controls the five pitfalls of AFML §11.4, as free text.
///
/// Every field must be non-blank; the engine checks nothing else. The record is a record, not
/// a control: it travels with the result in [`BacktestDiagnostics::safeguards`].
#[derive(Debug, Clone, PartialEq)]
pub struct BacktestSafeguards {
    /// How the universe avoids survivorship bias (e.g. includes delisted names).
    pub survivorship_bias_control: String,
    /// How features and labels avoid look-ahead (e.g. point-in-time data, purging).
    pub look_ahead_control: String,
    /// How the strategy was chosen without mining the backtest.
    pub data_mining_control: String,
    /// The transaction-cost assumptions.
    pub cost_assumption: String,
    /// How the number of trials is accounted for (e.g. a deflated Sharpe ratio).
    pub multiple_testing_control: String,
}

impl BacktestSafeguards {
    /// Checks that every field is non-blank.
    ///
    /// # Errors
    ///
    /// [`BacktestError::Empty`] naming the first field that is empty or whitespace only, in
    /// declaration order.
    ///
    /// ```
    /// use openquant::backtesting_engine::{BacktestError, BacktestSafeguards};
    ///
    /// let mut safeguards = BacktestSafeguards {
    ///     survivorship_bias_control: "delisted names included".into(),
    ///     look_ahead_control: "point-in-time fundamentals".into(),
    ///     data_mining_control: "hypothesis fixed in advance".into(),
    ///     cost_assumption: "5 bp per side".into(),
    ///     multiple_testing_control: "deflated Sharpe ratio".into(),
    /// };
    /// assert_eq!(safeguards.validate(), Ok(()));
    /// safeguards.cost_assumption = "   ".into();
    /// assert_eq!(safeguards.validate(), Err(BacktestError::Empty("cost_assumption")));
    /// ```
    pub fn validate(&self) -> Result<(), BacktestError> {
        if self.survivorship_bias_control.trim().is_empty() {
            return Err(BacktestError::Empty("survivorship_bias_control"));
        }
        if self.look_ahead_control.trim().is_empty() {
            return Err(BacktestError::Empty("look_ahead_control"));
        }
        if self.data_mining_control.trim().is_empty() {
            return Err(BacktestError::Empty("data_mining_control"));
        }
        if self.cost_assumption.trim().is_empty() {
            return Err(BacktestError::Empty("cost_assumption"));
        }
        if self.multiple_testing_control.trim().is_empty() {
            return Err(BacktestError::Empty("multiple_testing_control"));
        }
        Ok(())
    }
}

/// The samples to backtest: one return and one label span per sample, in time order.
#[derive(Debug, Clone, PartialEq)]
pub struct BacktestData {
    /// One finite value per sample. Only validated and used to count samples; the reported
    /// performance comes from the evaluator.
    pub returns: Vec<f64>,
    /// `(start, end)` of each sample's label, `start <= end`, used for purging (closed
    /// intervals) and to place the embargo.
    pub label_spans: Vec<(NaiveDateTime, NaiveDateTime)>,
}

impl BacktestData {
    /// Checks the data every runner requires.
    ///
    /// Time order of `label_spans` is not checked.
    ///
    /// # Errors
    ///
    /// - [`BacktestError::Empty`]`("returns")` when there are no samples.
    /// - [`BacktestError::ReturnsLabelSpansLengthMismatch`] when the lengths differ.
    /// - [`BacktestError::Invalid`] with `name = "returns"` when a return is NaN or infinite.
    /// - [`BacktestError::LabelSpanEndBeforeStart`] when a span ends before it starts.
    pub fn validate(&self) -> Result<(), BacktestError> {
        if self.returns.is_empty() {
            return Err(BacktestError::Empty("returns"));
        }
        if self.returns.len() != self.label_spans.len() {
            return Err(BacktestError::ReturnsLabelSpansLengthMismatch);
        }
        if self.returns.iter().any(|r| !r.is_finite()) {
            return Err(BacktestError::Invalid { name: "returns", requirement: "finite" });
        }
        for (start, end) in &self.label_spans {
            if end < start {
                return Err(BacktestError::LabelSpanEndBeforeStart);
            }
        }
        Ok(())
    }
}

/// The backtest mode a result came from (AFML Chapter 12).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BacktestMode {
    /// Walk-forward (§12.2): [`run_walk_forward`].
    WalkForward,
    /// Purged k-fold cross-validation (§12.3): [`run_cross_validation`].
    CrossValidation,
    /// Combinatorial purged cross-validation (§12.4): [`run_cpcv`].
    CombinatorialPurgedCrossValidation,
}

/// One split, as passed to the evaluator and returned in the result.
#[derive(Debug, Clone, PartialEq)]
pub struct SplitDefinition {
    /// Position of the split, from 0. In CPCV, the position in lexicographic order of
    /// `test_groups`, matching [`crate::cross_validation::CpcvSplit::split_id`].
    pub split_id: usize,
    /// Sorted training indices, after purging and embargo.
    pub train_indices: Vec<usize>,
    /// Sorted test indices. The evaluator returns one value per entry, in this order.
    pub test_indices: Vec<usize>,
    /// In CPCV, the ascending groups tested. In walk-forward and cross-validation, a single
    /// entry equal to `split_id` (the fold number, not a group of a fixed partition).
    pub test_groups: Vec<usize>,
    /// Training samples removed because their label overlaps a test label.
    pub purged_count: usize,
    /// Training samples removed by the embargo that the purge had not already removed.
    pub embargo_count: usize,
}

/// Summary of the out-of-sample returns the evaluator produced for one split.
#[derive(Debug, Clone, PartialEq)]
pub struct FoldPerformance {
    /// The [`SplitDefinition::split_id`] summarised.
    pub split_id: usize,
    /// t-statistic `mean_return / std_return * sqrt(observations)`; not annualised, and 0
    /// when `std_return` is 0.
    pub sharpe: f64,
    /// Arithmetic mean of the returns.
    pub mean_return: f64,
    /// Sample standard deviation (divisor `n - 1`); 0 for a single return.
    pub std_return: f64,
    /// Number of returns.
    pub observations: usize,
}

/// Totals of what purging and embargo removed across all splits of a run.
#[derive(Debug, Clone, PartialEq)]
pub struct AntiLeakageDiagnostics {
    /// Always `true`: every mode purges by label span.
    pub uses_label_span_purging: bool,
    /// `pct_embargo > 0`. In walk-forward mode the embargo removes nothing even when `true`.
    pub uses_embargo: bool,
    /// Sum of [`SplitDefinition::purged_count`] over the splits.
    pub total_purged: usize,
    /// Sum of [`SplitDefinition::embargo_count`] over the splits.
    pub total_embargoed: usize,
}

/// The run record returned with every result (AFML §11.4, §11.6).
#[derive(Debug, Clone, PartialEq)]
pub struct BacktestDiagnostics {
    /// The mode that produced the result.
    pub mode: BacktestMode,
    /// Copied from [`BacktestRunConfig::mode_provenance`].
    pub mode_provenance: String,
    /// Copied from [`BacktestRunConfig::trials_count`]; the trial count a deflated Sharpe
    /// ratio needs.
    pub trials_count: usize,
    /// Number of splits evaluated.
    pub split_count: usize,
    /// The embargo fraction used.
    pub pct_embargo: f64,
    /// Copied from [`BacktestRunConfig::safeguards`].
    pub safeguards: BacktestSafeguards,
    /// What purging and embargo removed.
    pub anti_leakage: AntiLeakageDiagnostics,
}

/// Parameters of [`run_walk_forward`] (AFML §12.2).
///
/// Test blocks start at `min_train_size`, `min_train_size + step_size`, ...; each covers
/// `test_size` samples (the last is clipped to the data) and trains on every earlier sample.
/// `step_size < test_size` tests some samples twice; `step_size > test_size` leaves gaps.
#[derive(Debug, Clone, PartialEq)]
pub struct WalkForwardConfig {
    /// Index of the first test sample; must be `> 0`.
    pub min_train_size: usize,
    /// Samples per test block; must be `> 0`.
    pub test_size: usize,
    /// Samples between the starts of consecutive test blocks; must be `> 0`.
    pub step_size: usize,
    /// Embargo fraction in `[0, 1)`. It has no effect in walk-forward mode, where all
    /// training data precedes the test block.
    pub pct_embargo: f64,
}

/// Parameters of [`run_cross_validation`] (AFML §12.3).
#[derive(Debug, Clone, PartialEq)]
pub struct CrossValidationConfig {
    /// Number of contiguous test folds; `2 <= n_splits <= n_samples`.
    pub n_splits: usize,
    /// Embargo fraction of the whole sample count, in `[0, 1)`, rounded up.
    pub pct_embargo: f64,
}

/// Parameters of [`run_cpcv`] (AFML §12.4).
#[derive(Debug, Clone, PartialEq)]
pub struct CpcvConfig {
    /// Number of contiguous groups `N`; `2 <= n_groups <= n_samples`.
    pub n_groups: usize,
    /// Number of groups tested per split `k`; `1 <= test_groups < n_groups`.
    pub test_groups: usize,
    /// Embargo fraction of the whole sample count, in `[0, 1)`, rounded up.
    pub pct_embargo: f64,
}

/// The run record every runner requires (AFML §11.4, §11.6).
///
/// Validated before any split is built; see the runners' `# Errors`.
#[derive(Debug, Clone, PartialEq)]
pub struct BacktestRunConfig {
    /// Free-text description of where this configuration came from; must be non-blank.
    pub mode_provenance: String,
    /// Number of trials (configurations tried) behind this result; must be `> 0`.
    pub trials_count: usize,
    /// How the run controls the §11.4 pitfalls.
    pub safeguards: BacktestSafeguards,
}

/// Which split supplies each group's returns on one CPCV path (AFML §12.4, Figure 12.1).
#[derive(Debug, Clone, PartialEq)]
pub struct CpcvPathAssignment {
    /// Position of the path, from 0 to φ − 1.
    pub path_id: usize,
    /// `split_for_group[g]` is the `split_id` of the `path_id`-th split that tests group `g`.
    /// Equal to [`crate::cross_validation::CpcvPath::split_for_fold`].
    pub split_for_group: Vec<usize>,
}

/// Summary of the returns along one CPCV path, which covers every sample once.
#[derive(Debug, Clone, PartialEq)]
pub struct CpcvPathPerformance {
    /// The [`CpcvPathAssignment::path_id`] summarised.
    pub path_id: usize,
    /// t-statistic `mean_return / std_return * sqrt(observations)`; not annualised, and 0
    /// when `std_return` is 0.
    pub sharpe: f64,
    /// Arithmetic mean of the path's returns.
    pub mean_return: f64,
    /// Sample standard deviation (divisor `n - 1`); 0 for a single return.
    pub std_return: f64,
    /// Number of returns on the path, the sample count.
    pub observations: usize,
}

/// Output of [`run_walk_forward`].
#[derive(Debug, Clone, PartialEq)]
pub struct WalkForwardResult {
    /// One summary per split, in split order.
    pub folds: Vec<FoldPerformance>,
    /// The splits evaluated.
    pub splits: Vec<SplitDefinition>,
    /// The run record.
    pub diagnostics: BacktestDiagnostics,
}

/// Output of [`run_cross_validation`].
#[derive(Debug, Clone, PartialEq)]
pub struct CrossValidationResult {
    /// One summary per fold, in fold order.
    pub folds: Vec<FoldPerformance>,
    /// The splits evaluated.
    pub splits: Vec<SplitDefinition>,
    /// The run record.
    pub diagnostics: BacktestDiagnostics,
}

/// Output of [`run_cpcv`].
#[derive(Debug, Clone, PartialEq)]
pub struct CpcvResult {
    /// One summary per split, in split order.
    pub folds: Vec<FoldPerformance>,
    /// The C(N, k) splits evaluated.
    pub splits: Vec<SplitDefinition>,
    /// φ[N, k], the number of paths.
    pub path_count: usize,
    /// Which split supplies each group on each path.
    pub path_assignments: Vec<CpcvPathAssignment>,
    /// One summary per path.
    pub path_distribution: Vec<CpcvPathPerformance>,
    /// The run record.
    pub diagnostics: BacktestDiagnostics,
}

/// The number of CPCV backtest paths, φ[N, k] = k/N · C(N, k) (AFML §12.4.1).
///
/// # Errors
///
/// - [`BacktestError::Invalid`] when `n_groups < 2` or `test_groups == 0`.
/// - [`BacktestError::TestGroupsNotBelowGroups`] when `test_groups >= n_groups`.
/// - [`BacktestError::CombinationCountOverflow`] when C(N, k) fits in `u128` but not in
///   `usize`.
///
/// # Panics
///
/// The binomial coefficient is computed as a `u128` product of `k` numerator terms, which
/// overflows for large inputs (for example `cpcv_path_count(200, 100)`): that panics in debug
/// builds and wraps silently in release. The final `C(N, k) * k` can likewise overflow `usize`.
///
/// ```
/// use openquant::backtesting_engine::cpcv_path_count;
///
/// // N = 6, k = 2: 15 splits, each group tested 5 times, 5 paths.
/// assert_eq!(cpcv_path_count(6, 2).unwrap(), 5);
/// assert_eq!(cpcv_path_count(10, 5).unwrap(), 126);
/// assert!(cpcv_path_count(6, 6).is_err());
/// ```
pub fn cpcv_path_count(n_groups: usize, test_groups: usize) -> Result<usize, BacktestError> {
    validate_cpcv_params(n_groups, test_groups)?;
    let total = n_choose_k(n_groups, test_groups)?;
    Ok((total * test_groups) / n_groups)
}

/// Runs a walk-forward backtest with an expanding training window (AFML §12.2).
///
/// Test blocks are laid out by `config` (see [`WalkForwardConfig`]); each trains on every
/// earlier sample, purged of labels that overlap the test labels. The embargo removes nothing
/// here, since only samples after a test block are embargoed. The evaluator is called once per
/// split, in order, and its returns are summarised; their length is not checked against
/// `test_indices`.
///
/// # Errors
///
/// - From validating `data` ([`BacktestData::validate`]) and `run`: [`BacktestError::Empty`]
///   for a blank `mode_provenance` or safeguard field, and [`BacktestError::Invalid`] for
///   `trials_count == 0`.
/// - [`BacktestError::Invalid`] when `pct_embargo` is not in `[0, 1)` (including NaN) or
///   `min_train_size`, `test_size` or `step_size` is 0.
/// - [`BacktestError::EmptyWalkForwardTrainSplit`] when purging empties a training set.
/// - [`BacktestError::NoWalkForwardSplits`] when `min_train_size >= n_samples`.
/// - [`BacktestError::EmptySplitReturns`] or [`BacktestError::NonFiniteSplitReturns`] for
///   the evaluator's output, and any error the evaluator returns, unchanged.
///
/// # Panics
///
/// `start + test_size` and `start + step_size` are unchecked, so a `test_size` or
/// `step_size` near `usize::MAX` overflows (a panic in debug builds).
///
/// ```
/// use chrono::{Duration, NaiveDate};
/// use openquant::backtesting_engine::{
///     run_walk_forward, BacktestData, BacktestError, BacktestRunConfig, BacktestSafeguards,
///     SplitDefinition, WalkForwardConfig,
/// };
///
/// # fn main() -> Result<(), BacktestError> {
/// let open = NaiveDate::from_ymd_opt(2024, 1, 2).unwrap().and_hms_opt(0, 0, 0).unwrap();
/// let returns: Vec<f64> = (0..12).map(|i| [0.01, 0.02, 0.03][i % 3]).collect();
/// let label_spans: Vec<_> =
///     (0..12).map(|i| (open + Duration::days(i), open + Duration::days(i + 1))).collect();
/// let data = BacktestData { returns: returns.clone(), label_spans };
/// # let note = |s: &str| s.to_string();
/// # let run = BacktestRunConfig {
/// #     mode_provenance: note("docs example"),
/// #     trials_count: 1,
/// #     safeguards: BacktestSafeguards {
/// #         survivorship_bias_control: note("synthetic"),
/// #         look_ahead_control: note("synthetic"),
/// #         data_mining_control: note("one configuration"),
/// #         cost_assumption: note("none"),
/// #         multiple_testing_control: note("single trial"),
/// #     },
/// # };
///
/// let config = WalkForwardConfig { min_train_size: 6, test_size: 3, step_size: 3, pct_embargo: 0.5 };
/// let result = run_walk_forward(&data, &run, &config, |s: &SplitDefinition| {
///     Ok(s.test_indices.iter().map(|&i| returns[i]).collect())
/// })?;
///
/// // Tests 6-8 and 9-11. The last training label before each block touches it and is purged;
/// // the embargo removes nothing.
/// assert_eq!(result.splits.len(), 2);
/// assert_eq!(result.splits[0].train_indices, vec![0, 1, 2, 3, 4]);
/// assert_eq!(result.splits[1].train_indices, (0..8).collect::<Vec<usize>>());
/// assert_eq!(result.diagnostics.anti_leakage.total_purged, 2);
/// assert_eq!(result.diagnostics.anti_leakage.total_embargoed, 0);
/// // Each block holds 0.01, 0.02, 0.03: mean 0.02, deviation 0.01, t-statistic 2 * sqrt(3).
/// assert!((result.folds[0].sharpe - 2.0 * 3f64.sqrt()).abs() < 1e-9);
/// # Ok(())
/// # }
/// ```
pub fn run_walk_forward<E>(
    data: &BacktestData,
    run: &BacktestRunConfig,
    config: &WalkForwardConfig,
    mut evaluator: E,
) -> Result<WalkForwardResult, BacktestError>
where
    E: FnMut(&SplitDefinition) -> Result<Vec<f64>, BacktestError>,
{
    data.validate()?;
    run.validate(BacktestMode::WalkForward)?;
    validate_embargo(config.pct_embargo)?;
    if config.min_train_size == 0 {
        return Err(BacktestError::Invalid { name: "min_train_size", requirement: "> 0" });
    }
    if config.test_size == 0 {
        return Err(BacktestError::Invalid { name: "test_size", requirement: "> 0" });
    }
    if config.step_size == 0 {
        return Err(BacktestError::Invalid { name: "step_size", requirement: "> 0" });
    }

    let n_samples = data.returns.len();
    let mut split_defs = Vec::new();
    let mut start = config.min_train_size;
    let mut split_id = 0;

    while start < n_samples {
        let stop = (start + config.test_size).min(n_samples);
        let test_indices: Vec<usize> = (start..stop).collect();
        if test_indices.is_empty() {
            break;
        }
        let initial_train: Vec<usize> = (0..start).collect();
        let (train_indices, purged_count, embargo_count) = apply_purge_and_embargo(
            &initial_train,
            &test_indices,
            &data.label_spans,
            config.pct_embargo,
            n_samples,
        );
        if train_indices.is_empty() {
            return Err(BacktestError::EmptyWalkForwardTrainSplit);
        }
        split_defs.push(SplitDefinition {
            split_id,
            train_indices,
            test_indices,
            test_groups: vec![split_id],
            purged_count,
            embargo_count,
        });
        split_id += 1;
        start += config.step_size;
    }

    if split_defs.is_empty() {
        return Err(BacktestError::NoWalkForwardSplits);
    }

    let folds = evaluate_splits(&split_defs, &mut evaluator)?;
    let diagnostics =
        build_diagnostics(BacktestMode::WalkForward, run, config.pct_embargo, &split_defs);
    Ok(WalkForwardResult { folds, splits: split_defs, diagnostics })
}

/// Runs a purged k-fold cross-validation backtest (AFML §12.3).
///
/// The samples are cut into `n_splits` contiguous folds (the first `n_samples % n_splits`
/// hold one extra sample). Each fold is tested once, training on every other sample after
/// purging and embargo. The evaluator is called once per fold, in order, and its returns are
/// summarised; their length is not checked against `test_indices`.
///
/// # Errors
///
/// - From validating `data` ([`BacktestData::validate`]) and `run`: [`BacktestError::Empty`]
///   for a blank `mode_provenance` or safeguard field, and [`BacktestError::Invalid`] for
///   `trials_count == 0`.
/// - [`BacktestError::Invalid`] when `pct_embargo` is not in `[0, 1)` or `n_splits < 2`.
/// - [`BacktestError::TooManyFolds`] when `n_splits > n_samples`.
/// - [`BacktestError::EmptyCrossValidationTrainSplit`] when purging and embargo empty a
///   training set.
/// - [`BacktestError::EmptySplitReturns`] or [`BacktestError::NonFiniteSplitReturns`] for
///   the evaluator's output, and any error the evaluator returns, unchanged.
///
/// ```
/// use chrono::{Duration, NaiveDate};
/// use openquant::backtesting_engine::{
///     run_cross_validation, BacktestData, BacktestError, BacktestRunConfig, BacktestSafeguards,
///     CrossValidationConfig, SplitDefinition,
/// };
///
/// # fn main() -> Result<(), BacktestError> {
/// let open = NaiveDate::from_ymd_opt(2024, 1, 2).unwrap().and_hms_opt(0, 0, 0).unwrap();
/// let returns: Vec<f64> = (0..12).map(|i| [0.01, 0.02, 0.03][i % 3]).collect();
/// let label_spans: Vec<_> =
///     (0..12).map(|i| (open + Duration::days(i), open + Duration::days(i + 1))).collect();
/// let data = BacktestData { returns: returns.clone(), label_spans };
/// # let note = |s: &str| s.to_string();
/// # let run = BacktestRunConfig {
/// #     mode_provenance: note("docs example"),
/// #     trials_count: 1,
/// #     safeguards: BacktestSafeguards {
/// #         survivorship_bias_control: note("synthetic"),
/// #         look_ahead_control: note("synthetic"),
/// #         data_mining_control: note("one configuration"),
/// #         cost_assumption: note("none"),
/// #         multiple_testing_control: note("single trial"),
/// #     },
/// # };
///
/// let config = CrossValidationConfig { n_splits: 3, pct_embargo: 0.1 };
/// let result = run_cross_validation(&data, &run, &config, |s: &SplitDefinition| {
///     Ok(s.test_indices.iter().map(|&i| returns[i]).collect())
/// })?;
///
/// // The middle fold tests 4-7. Samples 3 and 8 touch it and are purged; an embargo of
/// // ceil(0.1 * 12) = 2 then removes 9 and 10. Nothing before the fold is embargoed.
/// let split = &result.splits[1];
/// assert_eq!(split.test_indices, vec![4, 5, 6, 7]);
/// assert_eq!((split.purged_count, split.embargo_count), (2, 2));
/// assert_eq!(split.train_indices, vec![0, 1, 2, 11]);
/// # Ok(())
/// # }
/// ```
pub fn run_cross_validation<E>(
    data: &BacktestData,
    run: &BacktestRunConfig,
    config: &CrossValidationConfig,
    mut evaluator: E,
) -> Result<CrossValidationResult, BacktestError>
where
    E: FnMut(&SplitDefinition) -> Result<Vec<f64>, BacktestError>,
{
    data.validate()?;
    run.validate(BacktestMode::CrossValidation)?;
    validate_embargo(config.pct_embargo)?;
    if config.n_splits < 2 {
        return Err(BacktestError::Invalid { name: "n_splits", requirement: ">= 2" });
    }

    let base_test_splits = contiguous_folds(data.returns.len(), config.n_splits)?;
    let mut split_defs = Vec::with_capacity(base_test_splits.len());
    for (split_id, test_indices) in base_test_splits.into_iter().enumerate() {
        let initial_train: Vec<usize> =
            (0..data.returns.len()).filter(|idx| !test_indices.contains(idx)).collect();
        let (train_indices, purged_count, embargo_count) = apply_purge_and_embargo(
            &initial_train,
            &test_indices,
            &data.label_spans,
            config.pct_embargo,
            data.returns.len(),
        );
        if train_indices.is_empty() {
            return Err(BacktestError::EmptyCrossValidationTrainSplit);
        }
        split_defs.push(SplitDefinition {
            split_id,
            train_indices,
            test_indices,
            test_groups: vec![split_id],
            purged_count,
            embargo_count,
        });
    }

    let folds = evaluate_splits(&split_defs, &mut evaluator)?;
    let diagnostics =
        build_diagnostics(BacktestMode::CrossValidation, run, config.pct_embargo, &split_defs);
    Ok(CrossValidationResult { folds, splits: split_defs, diagnostics })
}

/// Runs a combinatorial purged cross-validation backtest (AFML §12.4, §12.4.2).
///
/// The samples are cut into `n_groups` contiguous groups, and one split is built for each of
/// the C(N, k) ways to test `test_groups` of them, in lexicographic order. Each split trains on
/// the other groups after purging (pair by pair against every test label) and embargo (after
/// each run of adjacent test groups). The evaluator is called once per split and must return
/// one value per test index, in order. Path `j` then takes each group's returns from the
/// `j`-th split that tests it, giving φ[N, k] paths that each cover every sample once. See the
/// [module docs](crate::backtesting_engine) for an example.
///
/// The paths are rearrangements of the same predictions and are not independent.
///
/// # Errors
///
/// - From validating `data` ([`BacktestData::validate`]) and `run`: [`BacktestError::Empty`]
///   for a blank `mode_provenance` or safeguard field, and [`BacktestError::Invalid`] for
///   `trials_count == 0`.
/// - [`BacktestError::Invalid`] when `n_groups < 2`, `test_groups == 0` or `pct_embargo` is
///   not in `[0, 1)`; [`BacktestError::TestGroupsNotBelowGroups`] when
///   `test_groups >= n_groups`.
/// - [`BacktestError::TooManyFolds`] when `n_groups > n_samples`.
/// - [`BacktestError::EmptyCpcvTrainSplit`] when purging and embargo empty a training set.
/// - [`BacktestError::EmptySplitReturns`] or [`BacktestError::NonFiniteSplitReturns`] for
///   the evaluator's output, and any error the evaluator returns, unchanged.
/// - [`BacktestError::SplitReturnCountMismatch`] when the evaluator returns a number of values
///   different from `test_indices.len()`.
/// - [`BacktestError::CombinationCountOverflow`], from [`cpcv_path_count`].
///
/// All splits are enumerated and evaluated before the path count is computed, so the cost is
/// C(N, k) evaluator calls: `(10, 2)` is 45 model fits, `(16, 8)` is 12,870.
pub fn run_cpcv<E>(
    data: &BacktestData,
    run: &BacktestRunConfig,
    config: &CpcvConfig,
    mut evaluator: E,
) -> Result<CpcvResult, BacktestError>
where
    E: FnMut(&SplitDefinition) -> Result<Vec<f64>, BacktestError>,
{
    data.validate()?;
    run.validate(BacktestMode::CombinatorialPurgedCrossValidation)?;
    validate_cpcv_params(config.n_groups, config.test_groups)?;
    validate_embargo(config.pct_embargo)?;

    let group_index_map = contiguous_folds(data.returns.len(), config.n_groups)?;
    let group_combinations = combinations(config.n_groups, config.test_groups);
    let mut split_defs = Vec::with_capacity(group_combinations.len());

    for (split_id, groups) in group_combinations.iter().enumerate() {
        let mut test_indices = Vec::new();
        for g in groups {
            test_indices.extend(group_index_map[*g].iter().copied());
        }
        test_indices.sort_unstable();

        let initial_train: Vec<usize> =
            (0..data.returns.len()).filter(|idx| !test_indices.contains(idx)).collect();
        let (train_indices, purged_count, embargo_count) = apply_purge_and_embargo(
            &initial_train,
            &test_indices,
            &data.label_spans,
            config.pct_embargo,
            data.returns.len(),
        );

        if train_indices.is_empty() {
            return Err(BacktestError::EmptyCpcvTrainSplit);
        }

        split_defs.push(SplitDefinition {
            split_id,
            train_indices,
            test_indices,
            test_groups: groups.clone(),
            purged_count,
            embargo_count,
        });
    }

    let (folds, split_returns) = evaluate_splits_with_returns(&split_defs, &mut evaluator)?;
    let path_count = cpcv_path_count(config.n_groups, config.test_groups)?;
    let path_assignments = build_cpcv_path_assignments(config.n_groups, &split_defs, path_count)?;
    let path_distribution = build_path_distribution(
        config.n_groups,
        &group_index_map,
        &path_assignments,
        &split_defs,
        &split_returns,
    )?;

    let diagnostics = build_diagnostics(
        BacktestMode::CombinatorialPurgedCrossValidation,
        run,
        config.pct_embargo,
        &split_defs,
    );

    Ok(CpcvResult {
        folds,
        splits: split_defs,
        path_count,
        path_assignments,
        path_distribution,
        diagnostics,
    })
}

fn evaluate_splits<E>(
    splits: &[SplitDefinition],
    evaluator: &mut E,
) -> Result<Vec<FoldPerformance>, BacktestError>
where
    E: FnMut(&SplitDefinition) -> Result<Vec<f64>, BacktestError>,
{
    let mut out = Vec::with_capacity(splits.len());
    for split in splits {
        let split_returns = evaluator(split)?;
        let perf = summarize_returns(split.split_id, &split_returns)?;
        out.push(perf);
    }
    Ok(out)
}

/// Per-fold performance plus the out-of-sample returns keyed by split id.
type SplitEvaluation = (Vec<FoldPerformance>, HashMap<usize, Vec<f64>>);

fn evaluate_splits_with_returns<E>(
    splits: &[SplitDefinition],
    evaluator: &mut E,
) -> Result<SplitEvaluation, BacktestError>
where
    E: FnMut(&SplitDefinition) -> Result<Vec<f64>, BacktestError>,
{
    let mut out = Vec::with_capacity(splits.len());
    let mut split_returns = HashMap::with_capacity(splits.len());
    for split in splits {
        let returns = evaluator(split)?;
        let perf = summarize_returns(split.split_id, &returns)?;
        out.push(perf);
        split_returns.insert(split.split_id, returns);
    }
    Ok((out, split_returns))
}

fn summarize_returns(split_id: usize, returns: &[f64]) -> Result<FoldPerformance, BacktestError> {
    if returns.is_empty() {
        return Err(BacktestError::EmptySplitReturns);
    }
    if returns.iter().any(|r| !r.is_finite()) {
        return Err(BacktestError::NonFiniteSplitReturns);
    }
    let n = returns.len();
    let mean = returns.iter().sum::<f64>() / n as f64;
    let variance = if n > 1 {
        returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / (n as f64 - 1.0)
    } else {
        0.0
    };
    let std = variance.sqrt();
    let sharpe = if std > 0.0 { mean / std * (n as f64).sqrt() } else { 0.0 };
    Ok(FoldPerformance { split_id, sharpe, mean_return: mean, std_return: std, observations: n })
}

fn build_diagnostics(
    mode: BacktestMode,
    run: &BacktestRunConfig,
    pct_embargo: f64,
    splits: &[SplitDefinition],
) -> BacktestDiagnostics {
    let total_purged = splits.iter().map(|s| s.purged_count).sum::<usize>();
    let total_embargoed = splits.iter().map(|s| s.embargo_count).sum::<usize>();
    BacktestDiagnostics {
        mode,
        mode_provenance: run.mode_provenance.clone(),
        trials_count: run.trials_count,
        split_count: splits.len(),
        pct_embargo,
        safeguards: run.safeguards.clone(),
        anti_leakage: AntiLeakageDiagnostics {
            uses_label_span_purging: true,
            uses_embargo: pct_embargo > 0.0,
            total_purged,
            total_embargoed,
        },
    }
}

impl BacktestRunConfig {
    fn validate(&self, mode: BacktestMode) -> Result<(), BacktestError> {
        if self.mode_provenance.trim().is_empty() {
            return Err(BacktestError::Empty("mode_provenance"));
        }
        if self.trials_count == 0 {
            return Err(BacktestError::Invalid { name: "trials_count", requirement: "> 0" });
        }
        self.safeguards.validate()?;
        match mode {
            BacktestMode::WalkForward
            | BacktestMode::CrossValidation
            | BacktestMode::CombinatorialPurgedCrossValidation => Ok(()),
        }
    }
}

fn validate_embargo(pct_embargo: f64) -> Result<(), BacktestError> {
    if !(0.0..1.0).contains(&pct_embargo) {
        return Err(BacktestError::Invalid { name: "pct_embargo", requirement: "in [0,1)" });
    }
    Ok(())
}

fn validate_cpcv_params(n_groups: usize, test_groups: usize) -> Result<(), BacktestError> {
    if n_groups < 2 {
        return Err(BacktestError::Invalid { name: "n_groups", requirement: ">= 2" });
    }
    if test_groups == 0 {
        return Err(BacktestError::Invalid { name: "test_groups", requirement: "> 0" });
    }
    if test_groups >= n_groups {
        return Err(BacktestError::TestGroupsNotBelowGroups);
    }
    Ok(())
}

fn contiguous_folds(n_samples: usize, n_folds: usize) -> Result<Vec<Vec<usize>>, BacktestError> {
    if n_folds == 0 {
        return Err(BacktestError::Invalid { name: "n_folds", requirement: "> 0" });
    }
    if n_folds > n_samples {
        return Err(BacktestError::TooManyFolds);
    }

    let mut fold_sizes = vec![n_samples / n_folds; n_folds];
    for size in fold_sizes.iter_mut().take(n_samples % n_folds) {
        *size += 1;
    }

    let mut current = 0;
    let mut out = Vec::with_capacity(n_folds);
    for fold_size in fold_sizes {
        let indices: Vec<usize> = (current..(current + fold_size)).collect();
        out.push(indices);
        current += fold_size;
    }
    Ok(out)
}

fn overlaps(lhs: (NaiveDateTime, NaiveDateTime), rhs: (NaiveDateTime, NaiveDateTime)) -> bool {
    lhs.0 <= rhs.1 && rhs.0 <= lhs.1
}

fn apply_purge_and_embargo(
    initial_train: &[usize],
    test_indices: &[usize],
    label_spans: &[(NaiveDateTime, NaiveDateTime)],
    pct_embargo: f64,
    n_samples: usize,
) -> (Vec<usize>, usize, usize) {
    let mut train_mask = vec![false; n_samples];
    for idx in initial_train {
        train_mask[*idx] = true;
    }

    let mut purged_count = 0;
    for idx in initial_train {
        let mut should_purge = false;
        for test_idx in test_indices {
            if overlaps(label_spans[*idx], label_spans[*test_idx]) {
                should_purge = true;
                break;
            }
        }
        if should_purge {
            train_mask[*idx] = false;
            purged_count += 1;
        }
    }

    // Embargo (AFML 7.4.2, Snippet 7.3): only training samples that FOLLOW a test block are
    // embargoed, and the count starts where the purge ends: at the first sample after the block
    // whose label starts after the block's latest label end. Samples before a test block are
    // never embargoed. A CPCV split has one block per run of adjacent test groups. The rule is
    // shared with `cross_validation::PurgedKFold`.
    let mut test_mask = vec![false; n_samples];
    for idx in test_indices {
        test_mask[*idx] = true;
    }
    let mut embargo_count = 0;
    let width = embargo_width(pct_embargo, n_samples);
    for window in embargo_windows(label_spans, &test_mask, width) {
        for e in window {
            if train_mask[e] {
                train_mask[e] = false;
                embargo_count += 1;
            }
        }
    }

    let train_indices: Vec<usize> = train_mask
        .iter()
        .enumerate()
        .filter_map(|(idx, keep)| if *keep { Some(idx) } else { None })
        .collect();

    (train_indices, purged_count, embargo_count)
}

fn n_choose_k(n: usize, k: usize) -> Result<usize, BacktestError> {
    if k > n {
        return Err(BacktestError::CombinationSizeTooLarge);
    }
    let k_eff = k.min(n - k);
    let mut numerator: u128 = 1;
    let mut denominator: u128 = 1;
    for i in 0..k_eff {
        numerator *= (n - i) as u128;
        denominator *= (i + 1) as u128;
    }
    let comb = numerator / denominator;
    usize::try_from(comb).map_err(|_| BacktestError::CombinationCountOverflow)
}

fn combinations(n: usize, k: usize) -> Vec<Vec<usize>> {
    let mut out = Vec::new();
    let mut current = Vec::with_capacity(k);
    combinations_recursive(0, n, k, &mut current, &mut out);
    out
}

fn combinations_recursive(
    start: usize,
    n: usize,
    k: usize,
    current: &mut Vec<usize>,
    out: &mut Vec<Vec<usize>>,
) {
    if current.len() == k {
        out.push(current.clone());
        return;
    }
    for i in start..n {
        current.push(i);
        combinations_recursive(i + 1, n, k, current, out);
        current.pop();
    }
}

fn build_cpcv_path_assignments(
    n_groups: usize,
    splits: &[SplitDefinition],
    path_count: usize,
) -> Result<Vec<CpcvPathAssignment>, BacktestError> {
    let mut group_occurrences: Vec<Vec<usize>> = vec![Vec::new(); n_groups];
    for split in splits {
        for g in &split.test_groups {
            if *g >= n_groups {
                return Err(BacktestError::TestGroupOutOfRange);
            }
            group_occurrences[*g].push(split.split_id);
        }
    }

    for (group_idx, occurrences) in group_occurrences.iter().enumerate() {
        if occurrences.len() != path_count {
            return Err(BacktestError::GroupOccurrences {
                group: group_idx,
                found: occurrences.len(),
                expected: path_count,
            });
        }
    }

    let mut assignments = Vec::with_capacity(path_count);
    for path_id in 0..path_count {
        let mut split_for_group = Vec::with_capacity(n_groups);
        for occurrences in group_occurrences.iter().take(n_groups) {
            split_for_group.push(occurrences[path_id]);
        }
        assignments.push(CpcvPathAssignment { path_id, split_for_group });
    }
    Ok(assignments)
}

fn build_path_distribution(
    n_groups: usize,
    group_index_map: &[Vec<usize>],
    assignments: &[CpcvPathAssignment],
    splits: &[SplitDefinition],
    split_returns: &HashMap<usize, Vec<f64>>,
) -> Result<Vec<CpcvPathPerformance>, BacktestError> {
    let mut out = Vec::with_capacity(assignments.len());
    for assignment in assignments {
        if assignment.split_for_group.len() != n_groups {
            return Err(BacktestError::InvalidPathAssignmentLength);
        }

        let mut path_returns = Vec::new();
        for (group_id, split_id) in assignment.split_for_group.iter().enumerate() {
            let split = splits
                .iter()
                .find(|s| s.split_id == *split_id)
                .ok_or(BacktestError::UnknownSplitInPath)?;
            if !split.test_groups.contains(&group_id) {
                return Err(BacktestError::PathSplitMissingGroup);
            }
            let split_path_returns =
                split_returns.get(split_id).ok_or(BacktestError::MissingSplitReturns)?;
            if split_path_returns.len() != split.test_indices.len() {
                return Err(BacktestError::SplitReturnCountMismatch);
            }
            let return_by_index: HashMap<usize, f64> = split
                .test_indices
                .iter()
                .copied()
                .zip(split_path_returns.iter().copied())
                .collect();

            for idx in &group_index_map[group_id] {
                if split.test_indices.contains(idx) {
                    let r = return_by_index.get(idx).ok_or(BacktestError::MissingGroupTestIndex)?;
                    path_returns.push(*r);
                }
            }
        }

        let perf = summarize_returns(assignment.path_id, &path_returns)?;
        out.push(CpcvPathPerformance {
            path_id: assignment.path_id,
            sharpe: perf.sharpe,
            mean_return: perf.mean_return,
            std_return: perf.std_return,
            observations: perf.observations,
        });
    }
    Ok(out)
}

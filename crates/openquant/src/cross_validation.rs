use chrono::NaiveDateTime;
use itertools::Itertools;

/// Errors returned by the cross-validation splitters and diagnostics.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum CrossValidationError {
    /// `n_splits` is below 2 or above the number of samples.
    #[error("n_splits must be between 2 and the number of samples ({n_samples}), got {n_splits}")]
    InvalidSplits { n_splits: usize, n_samples: usize },
    /// A required input is empty.
    #[error("{0} cannot be empty")]
    Empty(&'static str),
    /// `n_samples` passed to a split method differs from the number of information sets.
    #[error("Dataset length must match samples_info_sets")]
    DatasetLengthMismatch,
    /// `pct_embargo` is not a finite number in `[0, 1)`.
    #[error("pct_embargo must be finite and in [0, 1), got {pct_embargo}")]
    InvalidEmbargo { pct_embargo: f64 },
    /// An information set ends before it starts.
    #[error("samples_info_sets[{index}] ends before it starts")]
    InvalidInfoSet { index: usize },
    /// `n_test_splits` for CPCV is not in `[1, n_splits)`.
    #[error("n_test_splits must be between 1 and {n_splits} - 1, got {n_test_splits}")]
    InvalidTestSplits { n_test_splits: usize, n_splits: usize },
    /// The number of CPCV splits, C(n_splits, n_test_splits), does not fit in `usize`.
    #[error("C({n_splits}, {n_test_splits}) CPCV splits overflow usize")]
    TooManySplits { n_splits: usize, n_test_splits: usize },
    /// A sample index is outside the information sets.
    #[error("sample index {index} is out of range for {len} information sets")]
    IndexOutOfRange { index: usize, len: usize },
}

/// Simple classifier interface for cross-validation.
pub trait SimpleClassifier {
    fn fit(&mut self, x: &[Vec<f64>], y: &[f64], sample_weight: Option<&[f64]>);
    fn predict_proba(&self, x: &[Vec<f64>]) -> Vec<f64>;
    fn predict(&self, x: &[Vec<f64>]) -> Vec<f64> {
        self.predict_proba(x).into_iter().map(|p| if p >= 0.5 { 1.0 } else { 0.0 }).collect()
    }
}

#[derive(Clone, Copy)]
pub enum Scoring {
    Accuracy,
    NegLogLoss,
    F1,
}

pub fn ml_cross_val_score<C: SimpleClassifier>(
    classifier: &mut C,
    x: &[Vec<f64>],
    y: &[f64],
    sample_weight: Option<&[f64]>,
    splits: &[(Vec<usize>, Vec<usize>)],
    scoring: Scoring,
) -> Vec<f64> {
    let mut scores = Vec::new();
    for (train_idx, test_idx) in splits {
        let x_train: Vec<Vec<f64>> = train_idx.iter().map(|i| x[*i].clone()).collect();
        let y_train: Vec<f64> = train_idx.iter().map(|i| y[*i]).collect();
        let sw_train: Option<Vec<f64>> =
            sample_weight.map(|sw| train_idx.iter().map(|i| sw[*i]).collect());

        classifier.fit(&x_train, &y_train, sw_train.as_deref());
        let x_test: Vec<Vec<f64>> = test_idx.iter().map(|i| x[*i].clone()).collect();
        let y_test: Vec<f64> = test_idx.iter().map(|i| y[*i]).collect();
        let probs = classifier.predict_proba(&x_test);
        let preds: Vec<f64> = probs.iter().map(|p| if *p >= 0.5 { 1.0 } else { 0.0 }).collect();

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
    scores
}

/// Remove training intervals that overlap with test intervals.
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
    /// Non-test samples inside an embargo window (§7.4.2), whether or not also purged.
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

/// Purged k-fold cross-validation with an embargo (AFML Snippet 7.3).
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
    /// starts.
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

        // Embargo ceil(pct_embargo * n) samples on both sides of each block, counted from the
        // block's edges. This differs from Snippet 7.3; see the docs page and issue #134.
        let mut embargoed = vec![false; n];
        let embargo = (self.pct_embargo * n as f64).ceil() as usize;
        if embargo > 0 {
            for &(start, stop) in test_blocks {
                let before = start.saturating_sub(embargo)..start;
                let after = stop..stop.saturating_add(embargo).min(n);
                for i in before.chain(after) {
                    embargoed[i] |= !test[i];
                }
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
/// # Errors
/// [`CrossValidationError::IndexOutOfRange`] when an index is not a position in `info_sets`.
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

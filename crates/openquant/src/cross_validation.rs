use chrono::NaiveDateTime;
use itertools::Itertools;
use std::collections::HashSet;

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
            if intervals_overlap((*start, *end), (*test_start, *test_end)) {
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PurgedSplitDiagnostics {
    pub split_id: usize,
    pub test_ranges: Vec<(usize, usize)>,
    pub purged_indices: Vec<usize>,
    pub embargo_indices: Vec<usize>,
    pub overlap_count_after_purge: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PurgedSplit {
    pub train_indices: Vec<usize>,
    pub test_indices: Vec<usize>,
    pub diagnostics: PurgedSplitDiagnostics,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CpcvPath {
    pub path_id: usize,
    pub test_fold_ids: Vec<usize>,
    pub split: PurgedSplit,
}

pub struct PurgedKFold {
    n_splits: usize,
    samples_info_sets: Vec<(NaiveDateTime, NaiveDateTime)>,
    pct_embargo: f64,
}

impl PurgedKFold {
    pub fn new(
        n_splits: usize,
        samples_info_sets: Vec<(NaiveDateTime, NaiveDateTime)>,
        pct_embargo: f64,
    ) -> Result<Self, String> {
        if n_splits < 2 {
            return Err("n_splits must be at least 2".into());
        }
        if samples_info_sets.is_empty() {
            return Err("samples_info_sets cannot be empty".into());
        }
        if n_splits > samples_info_sets.len() {
            return Err("n_splits cannot exceed sample count".into());
        }
        if !(0.0..1.0).contains(&pct_embargo) {
            return Err("pct_embargo must be in [0.0, 1.0)".into());
        }
        for (idx, (start, end)) in samples_info_sets.iter().enumerate() {
            if start > end {
                return Err(format!("invalid information set at index {idx}: start > end"));
            }
        }

        Ok(Self { n_splits, samples_info_sets, pct_embargo })
    }

    pub fn split(&self, n_samples: usize) -> Result<Vec<(Vec<usize>, Vec<usize>)>, String> {
        let splits = self.split_with_diagnostics(n_samples)?;
        Ok(splits.into_iter().map(|s| (s.train_indices, s.test_indices)).collect())
    }

    pub fn split_with_diagnostics(&self, n_samples: usize) -> Result<Vec<PurgedSplit>, String> {
        self.validate_n_samples(n_samples)?;
        let folds = contiguous_fold_bounds(n_samples, self.n_splits);

        let mut out = Vec::with_capacity(folds.len());
        for (split_id, (start, stop)) in folds.iter().enumerate() {
            let test_indices: Vec<usize> = (*start..*stop).collect();
            out.push(self.build_split(split_id, test_indices)?);
        }
        Ok(out)
    }

    pub fn cpcv_paths(
        &self,
        n_samples: usize,
        n_test_splits: usize,
    ) -> Result<Vec<CpcvPath>, String> {
        self.validate_n_samples(n_samples)?;
        if n_test_splits == 0 || n_test_splits >= self.n_splits {
            return Err("n_test_splits must be in [1, n_splits)".into());
        }

        let folds = contiguous_fold_bounds(n_samples, self.n_splits);
        let mut paths = Vec::new();

        for (path_id, test_fold_ids) in (0..self.n_splits).combinations(n_test_splits).enumerate() {
            let mut test_indices = Vec::new();
            for fold_id in &test_fold_ids {
                let (start, stop) = folds[*fold_id];
                test_indices.extend(start..stop);
            }
            let split = self.build_split(path_id, test_indices)?;
            paths.push(CpcvPath { path_id, test_fold_ids, split });
        }

        Ok(paths)
    }

    fn validate_n_samples(&self, n_samples: usize) -> Result<(), String> {
        if n_samples != self.samples_info_sets.len() {
            return Err("dataset length must match samples_info_sets".into());
        }
        Ok(())
    }

    fn build_split(
        &self,
        split_id: usize,
        test_indices: Vec<usize>,
    ) -> Result<PurgedSplit, String> {
        if test_indices.is_empty() {
            return Err("test_indices cannot be empty".into());
        }

        let n_samples = self.samples_info_sets.len();
        let test_set: HashSet<usize> = test_indices.iter().copied().collect();
        let test_intervals: Vec<(NaiveDateTime, NaiveDateTime)> =
            test_indices.iter().map(|idx| self.samples_info_sets[*idx]).collect();

        let mut purged = Vec::new();
        for idx in 0..n_samples {
            if test_set.contains(&idx) {
                continue;
            }
            let candidate = self.samples_info_sets[idx];
            let overlaps = test_intervals
                .iter()
                .any(|test_interval| intervals_overlap(candidate, *test_interval));
            if overlaps {
                purged.push(idx);
            }
        }

        let embargo = ((self.pct_embargo * n_samples as f64).ceil()) as usize;
        let mut embargoed = HashSet::new();
        if embargo > 0 {
            for (_start, stop) in contiguous_ranges(&test_indices) {
                let embargo_stop = (stop + embargo).min(n_samples);
                for idx in stop..embargo_stop {
                    if !test_set.contains(&idx) {
                        embargoed.insert(idx);
                    }
                }
            }
        }

        let purged_set: HashSet<usize> = purged.iter().copied().collect();
        let train_indices: Vec<usize> = (0..n_samples)
            .filter(|idx| {
                !test_set.contains(idx) && !purged_set.contains(idx) && !embargoed.contains(idx)
            })
            .collect();

        let overlap_count_after_purge =
            count_train_test_overlaps(&self.samples_info_sets, &train_indices, &test_indices);

        let mut embargo_indices: Vec<usize> = embargoed.into_iter().collect();
        embargo_indices.sort_unstable();

        Ok(PurgedSplit {
            train_indices,
            test_indices: test_indices.clone(),
            diagnostics: PurgedSplitDiagnostics {
                split_id,
                test_ranges: contiguous_ranges(&test_indices),
                purged_indices: purged,
                embargo_indices,
                overlap_count_after_purge,
            },
        })
    }
}

pub fn naive_kfold_splits(
    n_samples: usize,
    n_splits: usize,
) -> Result<Vec<(Vec<usize>, Vec<usize>)>, String> {
    if n_samples == 0 {
        return Err("n_samples must be > 0".into());
    }
    if n_splits < 2 {
        return Err("n_splits must be at least 2".into());
    }
    if n_splits > n_samples {
        return Err("n_splits cannot exceed n_samples".into());
    }

    let folds = contiguous_fold_bounds(n_samples, n_splits);
    let mut out = Vec::with_capacity(folds.len());
    for (start, stop) in folds {
        let test_indices: Vec<usize> = (start..stop).collect();
        let train_indices: Vec<usize> =
            (0..n_samples).filter(|i| *i < start || *i >= stop).collect();
        out.push((train_indices, test_indices));
    }
    Ok(out)
}

pub fn count_train_test_overlaps(
    info_sets: &[(NaiveDateTime, NaiveDateTime)],
    train_indices: &[usize],
    test_indices: &[usize],
) -> usize {
    let mut overlaps = 0;
    for train_idx in train_indices {
        let train_interval = info_sets[*train_idx];
        if test_indices
            .iter()
            .any(|test_idx| intervals_overlap(train_interval, info_sets[*test_idx]))
        {
            overlaps += 1;
        }
    }
    overlaps
}

fn intervals_overlap(a: (NaiveDateTime, NaiveDateTime), b: (NaiveDateTime, NaiveDateTime)) -> bool {
    a.0 <= b.1 && b.0 <= a.1
}

fn contiguous_fold_bounds(n_samples: usize, n_splits: usize) -> Vec<(usize, usize)> {
    let mut fold_sizes = vec![n_samples / n_splits; n_splits];
    for size in fold_sizes.iter_mut().take(n_samples % n_splits) {
        *size += 1;
    }

    let mut current = 0;
    let mut bounds = Vec::with_capacity(n_splits);
    for fold_size in fold_sizes {
        let next = current + fold_size;
        bounds.push((current, next));
        current = next;
    }
    bounds
}

fn contiguous_ranges(indices: &[usize]) -> Vec<(usize, usize)> {
    if indices.is_empty() {
        return Vec::new();
    }

    let mut sorted = indices.to_vec();
    sorted.sort_unstable();

    let mut ranges = Vec::new();
    let mut start = sorted[0];
    let mut prev = sorted[0];

    for idx in sorted.into_iter().skip(1) {
        if idx == prev + 1 {
            prev = idx;
            continue;
        }
        ranges.push((start, prev + 1));
        start = idx;
        prev = idx;
    }
    ranges.push((start, prev + 1));
    ranges
}

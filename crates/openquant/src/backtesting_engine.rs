//! Backtesting engine with walk-forward, purged CV, and CPCV support.
//!
//! AFML Chapter 11 framing is implemented through explicit diagnostics that
//! force provenance and bias-control metadata into each run, while Chapter 12
//! split logic is represented through WF/CV/CPCV mode-specific pathways.

use chrono::NaiveDateTime;
use std::collections::HashMap;

#[derive(Debug, Clone, PartialEq)]
pub struct BacktestSafeguards {
    pub survivorship_bias_control: String,
    pub look_ahead_control: String,
    pub data_mining_control: String,
    pub cost_assumption: String,
    pub multiple_testing_control: String,
}

impl BacktestSafeguards {
    pub fn validate(&self) -> Result<(), String> {
        if self.survivorship_bias_control.trim().is_empty() {
            return Err("survivorship_bias_control cannot be empty".to_string());
        }
        if self.look_ahead_control.trim().is_empty() {
            return Err("look_ahead_control cannot be empty".to_string());
        }
        if self.data_mining_control.trim().is_empty() {
            return Err("data_mining_control cannot be empty".to_string());
        }
        if self.cost_assumption.trim().is_empty() {
            return Err("cost_assumption cannot be empty".to_string());
        }
        if self.multiple_testing_control.trim().is_empty() {
            return Err("multiple_testing_control cannot be empty".to_string());
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct BacktestData {
    pub returns: Vec<f64>,
    pub label_spans: Vec<(NaiveDateTime, NaiveDateTime)>,
}

impl BacktestData {
    pub fn validate(&self) -> Result<(), String> {
        if self.returns.is_empty() {
            return Err("returns cannot be empty".to_string());
        }
        if self.returns.len() != self.label_spans.len() {
            return Err("returns and label_spans length mismatch".to_string());
        }
        if self.returns.iter().any(|r| !r.is_finite()) {
            return Err("returns must be finite".to_string());
        }
        for (start, end) in &self.label_spans {
            if end < start {
                return Err("label span end must be >= start".to_string());
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BacktestMode {
    WalkForward,
    CrossValidation,
    CombinatorialPurgedCrossValidation,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SplitDefinition {
    pub split_id: usize,
    pub train_indices: Vec<usize>,
    pub test_indices: Vec<usize>,
    pub test_groups: Vec<usize>,
    pub purged_count: usize,
    pub embargo_count: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FoldPerformance {
    pub split_id: usize,
    pub sharpe: f64,
    pub mean_return: f64,
    pub std_return: f64,
    pub observations: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AntiLeakageDiagnostics {
    pub uses_label_span_purging: bool,
    pub uses_embargo: bool,
    pub total_purged: usize,
    pub total_embargoed: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BacktestDiagnostics {
    pub mode: BacktestMode,
    pub mode_provenance: String,
    pub trials_count: usize,
    pub split_count: usize,
    pub pct_embargo: f64,
    pub safeguards: BacktestSafeguards,
    pub anti_leakage: AntiLeakageDiagnostics,
}

#[derive(Debug, Clone, PartialEq)]
pub struct WalkForwardConfig {
    pub min_train_size: usize,
    pub test_size: usize,
    pub step_size: usize,
    pub pct_embargo: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CrossValidationConfig {
    pub n_splits: usize,
    pub pct_embargo: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CpcvConfig {
    pub n_groups: usize,
    pub test_groups: usize,
    pub pct_embargo: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BacktestRunConfig {
    pub mode_provenance: String,
    pub trials_count: usize,
    pub safeguards: BacktestSafeguards,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CpcvPathAssignment {
    pub path_id: usize,
    pub split_for_group: Vec<usize>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CpcvPathPerformance {
    pub path_id: usize,
    pub sharpe: f64,
    pub mean_return: f64,
    pub std_return: f64,
    pub observations: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct WalkForwardResult {
    pub folds: Vec<FoldPerformance>,
    pub splits: Vec<SplitDefinition>,
    pub diagnostics: BacktestDiagnostics,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CrossValidationResult {
    pub folds: Vec<FoldPerformance>,
    pub splits: Vec<SplitDefinition>,
    pub diagnostics: BacktestDiagnostics,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CpcvResult {
    pub folds: Vec<FoldPerformance>,
    pub splits: Vec<SplitDefinition>,
    pub path_count: usize,
    pub path_assignments: Vec<CpcvPathAssignment>,
    pub path_distribution: Vec<CpcvPathPerformance>,
    pub diagnostics: BacktestDiagnostics,
}

pub fn cpcv_path_count(n_groups: usize, test_groups: usize) -> Result<usize, String> {
    validate_cpcv_params(n_groups, test_groups)?;
    let total = n_choose_k(n_groups, test_groups)?;
    Ok((total * test_groups) / n_groups)
}

pub fn run_walk_forward<E>(
    data: &BacktestData,
    run: &BacktestRunConfig,
    config: &WalkForwardConfig,
    mut evaluator: E,
) -> Result<WalkForwardResult, String>
where
    E: FnMut(&SplitDefinition) -> Result<Vec<f64>, String>,
{
    data.validate()?;
    run.validate(BacktestMode::WalkForward)?;
    validate_embargo(config.pct_embargo)?;
    if config.min_train_size == 0 {
        return Err("min_train_size must be > 0".to_string());
    }
    if config.test_size == 0 {
        return Err("test_size must be > 0".to_string());
    }
    if config.step_size == 0 {
        return Err("step_size must be > 0".to_string());
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
            return Err("walk-forward produced an empty train split".to_string());
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
        return Err("walk-forward produced no splits".to_string());
    }

    let folds = evaluate_splits(&split_defs, &mut evaluator)?;
    let diagnostics =
        build_diagnostics(BacktestMode::WalkForward, run, config.pct_embargo, &split_defs);
    Ok(WalkForwardResult { folds, splits: split_defs, diagnostics })
}

pub fn run_cross_validation<E>(
    data: &BacktestData,
    run: &BacktestRunConfig,
    config: &CrossValidationConfig,
    mut evaluator: E,
) -> Result<CrossValidationResult, String>
where
    E: FnMut(&SplitDefinition) -> Result<Vec<f64>, String>,
{
    data.validate()?;
    run.validate(BacktestMode::CrossValidation)?;
    validate_embargo(config.pct_embargo)?;
    if config.n_splits < 2 {
        return Err("n_splits must be >= 2".to_string());
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
            return Err("cross-validation produced an empty train split".to_string());
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

pub fn run_cpcv<E>(
    data: &BacktestData,
    run: &BacktestRunConfig,
    config: &CpcvConfig,
    mut evaluator: E,
) -> Result<CpcvResult, String>
where
    E: FnMut(&SplitDefinition) -> Result<Vec<f64>, String>,
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
            return Err("CPCV produced an empty train split".to_string());
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
) -> Result<Vec<FoldPerformance>, String>
where
    E: FnMut(&SplitDefinition) -> Result<Vec<f64>, String>,
{
    let mut out = Vec::with_capacity(splits.len());
    for split in splits {
        let split_returns = evaluator(split)?;
        let perf = summarize_returns(split.split_id, &split_returns)?;
        out.push(perf);
    }
    Ok(out)
}

fn evaluate_splits_with_returns<E>(
    splits: &[SplitDefinition],
    evaluator: &mut E,
) -> Result<(Vec<FoldPerformance>, HashMap<usize, Vec<f64>>), String>
where
    E: FnMut(&SplitDefinition) -> Result<Vec<f64>, String>,
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

fn summarize_returns(split_id: usize, returns: &[f64]) -> Result<FoldPerformance, String> {
    if returns.is_empty() {
        return Err("split evaluator returned empty returns".to_string());
    }
    if returns.iter().any(|r| !r.is_finite()) {
        return Err("split evaluator returned non-finite returns".to_string());
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
    fn validate(&self, mode: BacktestMode) -> Result<(), String> {
        if self.mode_provenance.trim().is_empty() {
            return Err("mode_provenance cannot be empty".to_string());
        }
        if self.trials_count == 0 {
            return Err("trials_count must be > 0".to_string());
        }
        self.safeguards.validate()?;
        match mode {
            BacktestMode::WalkForward
            | BacktestMode::CrossValidation
            | BacktestMode::CombinatorialPurgedCrossValidation => Ok(()),
        }
    }
}

fn validate_embargo(pct_embargo: f64) -> Result<(), String> {
    if !(0.0..1.0).contains(&pct_embargo) {
        return Err("pct_embargo must be in [0,1)".to_string());
    }
    Ok(())
}

fn validate_cpcv_params(n_groups: usize, test_groups: usize) -> Result<(), String> {
    if n_groups < 2 {
        return Err("n_groups must be >= 2".to_string());
    }
    if test_groups == 0 {
        return Err("test_groups must be > 0".to_string());
    }
    if test_groups >= n_groups {
        return Err("test_groups must be < n_groups".to_string());
    }
    Ok(())
}

fn contiguous_folds(n_samples: usize, n_folds: usize) -> Result<Vec<Vec<usize>>, String> {
    if n_folds == 0 {
        return Err("n_folds must be > 0".to_string());
    }
    if n_folds > n_samples {
        return Err("n_folds cannot exceed number of samples".to_string());
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
    let mut initial_train_mask = vec![false; n_samples];
    for idx in initial_train {
        train_mask[*idx] = true;
        initial_train_mask[*idx] = true;
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

    let embargo_width = (pct_embargo * n_samples as f64).ceil() as usize;
    let mut embargoed = vec![false; n_samples];
    if embargo_width > 0 {
        for test_idx in test_indices {
            let start = test_idx.saturating_sub(embargo_width);
            let stop = (*test_idx + embargo_width + 1).min(n_samples);
            for idx in start..stop {
                if initial_train_mask[idx] {
                    embargoed[idx] = true;
                }
            }
        }
        for idx in 0..n_samples {
            if embargoed[idx] {
                train_mask[idx] = false;
            }
        }
    }

    let train_indices: Vec<usize> = train_mask
        .iter()
        .enumerate()
        .filter_map(|(idx, keep)| if *keep { Some(idx) } else { None })
        .collect();
    let embargo_count = embargoed.into_iter().filter(|v| *v).count();

    (train_indices, purged_count, embargo_count)
}

fn n_choose_k(n: usize, k: usize) -> Result<usize, String> {
    if k > n {
        return Err("k cannot exceed n".to_string());
    }
    let k_eff = k.min(n - k);
    let mut numerator: u128 = 1;
    let mut denominator: u128 = 1;
    for i in 0..k_eff {
        numerator *= (n - i) as u128;
        denominator *= (i + 1) as u128;
    }
    let comb = numerator / denominator;
    usize::try_from(comb).map_err(|_| "combination count overflowed usize".to_string())
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
) -> Result<Vec<CpcvPathAssignment>, String> {
    let mut group_occurrences: Vec<Vec<usize>> = vec![Vec::new(); n_groups];
    for split in splits {
        for g in &split.test_groups {
            if *g >= n_groups {
                return Err("split references out-of-range test group".to_string());
            }
            group_occurrences[*g].push(split.split_id);
        }
    }

    for (group_idx, occurrences) in group_occurrences.iter().enumerate() {
        if occurrences.len() != path_count {
            return Err(format!(
                "group {group_idx} has {} occurrences, expected {path_count}",
                occurrences.len()
            ));
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
) -> Result<Vec<CpcvPathPerformance>, String> {
    let mut out = Vec::with_capacity(assignments.len());
    for assignment in assignments {
        if assignment.split_for_group.len() != n_groups {
            return Err("invalid path assignment length".to_string());
        }

        let mut path_returns = Vec::new();
        for (group_id, split_id) in assignment.split_for_group.iter().enumerate() {
            let split = splits
                .iter()
                .find(|s| s.split_id == *split_id)
                .ok_or_else(|| "path references unknown split".to_string())?;
            if !split.test_groups.contains(&group_id) {
                return Err("path assignment references split not containing group".to_string());
            }
            let split_path_returns = split_returns
                .get(split_id)
                .ok_or_else(|| "missing split returns for CPCV path construction".to_string())?;
            if split_path_returns.len() != split.test_indices.len() {
                return Err("split return count must match split test indices length".to_string());
            }
            let return_by_index: HashMap<usize, f64> = split
                .test_indices
                .iter()
                .copied()
                .zip(split_path_returns.iter().copied())
                .collect();

            for idx in &group_index_map[group_id] {
                if split.test_indices.contains(idx) {
                    let r = return_by_index
                        .get(idx)
                        .ok_or_else(|| "split returns missing group test index".to_string())?;
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

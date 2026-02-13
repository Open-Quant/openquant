//! Leakage-aware hyperparameter search utilities aligned to AFML Chapter 9.
//!
//! This module provides deterministic grid/randomized search wrappers on top of
//! `PurgedKFold`, with scoring options that preserve sample-weight semantics in
//! both fit and evaluation paths.

use std::collections::BTreeMap;

use chrono::NaiveDateTime;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use crate::cross_validation::{PurgedKFold, SimpleClassifier};

#[derive(Debug, Clone, PartialEq)]
pub enum HyperParamValue {
    Int(i64),
    Float(f64),
    Bool(bool),
}

impl HyperParamValue {
    pub fn as_i64(&self) -> Option<i64> {
        match self {
            Self::Int(v) => Some(*v),
            _ => None,
        }
    }

    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Self::Int(v) => Some(*v as f64),
            Self::Float(v) => Some(*v),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Self::Bool(v) => Some(*v),
            _ => None,
        }
    }
}

pub type ParamSet = BTreeMap<String, HyperParamValue>;

#[derive(Debug, Clone)]
pub enum RandomParamDistribution {
    Choice(Vec<HyperParamValue>),
    Uniform { low: f64, high: f64 },
    LogUniform { low: f64, high: f64 },
    IntRangeInclusive { low: i64, high: i64 },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SearchScoring {
    Accuracy,
    BalancedAccuracy,
    NegLogLoss,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SearchTrial {
    pub params: ParamSet,
    pub fold_scores: Vec<f64>,
    pub mean_score: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SearchResult {
    pub best_params: ParamSet,
    pub best_score: f64,
    pub trials: Vec<SearchTrial>,
}

pub fn sample_log_uniform<R: Rng + ?Sized>(
    low: f64,
    high: f64,
    rng: &mut R,
) -> Result<f64, String> {
    if low <= 0.0 || high <= 0.0 {
        return Err("log-uniform bounds must be strictly positive".to_string());
    }
    if low >= high {
        return Err("log-uniform low must be < high".to_string());
    }
    let log_low = low.ln();
    let log_high = high.ln();
    let draw = rng.gen_range(log_low..log_high);
    Ok(draw.exp())
}

pub fn classification_score(
    y_true: &[f64],
    probabilities: &[f64],
    sample_weight: Option<&[f64]>,
    scoring: SearchScoring,
) -> Result<f64, String> {
    if y_true.is_empty() {
        return Err("y_true cannot be empty".to_string());
    }
    if probabilities.len() != y_true.len() {
        return Err("probabilities/y_true length mismatch".to_string());
    }
    if let Some(sw) = sample_weight {
        if sw.len() != y_true.len() {
            return Err("sample_weight length mismatch".to_string());
        }
        if sw.iter().any(|w| *w < 0.0) {
            return Err("sample_weight cannot contain negative values".to_string());
        }
    }
    if probabilities.iter().any(|p| !p.is_finite() || *p < 0.0 || *p > 1.0) {
        return Err("probabilities must be finite and in [0,1]".to_string());
    }
    if y_true.iter().any(|y| (*y - 0.0).abs() > 1e-12 && (*y - 1.0).abs() > 1e-12) {
        return Err("y_true must contain only binary labels in {0,1}".to_string());
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
        return Err("sum of sample_weight must be > 0".to_string());
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
                return Err("balanced accuracy requires at least one labeled sample".to_string());
            }
            Ok(recalls.iter().sum::<f64>() / recalls.len() as f64)
        }
    }
}

pub fn expand_param_grid(
    param_grid: &BTreeMap<String, Vec<HyperParamValue>>,
) -> Result<Vec<ParamSet>, String> {
    if param_grid.is_empty() {
        return Err("param_grid cannot be empty".to_string());
    }
    for (name, values) in param_grid {
        if values.is_empty() {
            return Err(format!("param_grid entry '{name}' cannot be empty"));
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

pub struct SearchData<'a> {
    pub x: &'a [Vec<f64>],
    pub y: &'a [f64],
    pub sample_weight: Option<&'a [f64]>,
    pub samples_info_sets: &'a [(NaiveDateTime, NaiveDateTime)],
}

pub fn grid_search<C, F>(
    build_classifier: F,
    param_grid: &BTreeMap<String, Vec<HyperParamValue>>,
    data: SearchData<'_>,
    n_splits: usize,
    pct_embargo: f64,
    scoring: SearchScoring,
) -> Result<SearchResult, String>
where
    C: SimpleClassifier,
    F: Fn(&ParamSet) -> C,
{
    let params = expand_param_grid(param_grid)?;
    search_over_params(build_classifier, params, data, n_splits, pct_embargo, scoring)
}

pub fn randomized_search<C, F>(
    build_classifier: F,
    param_space: &BTreeMap<String, RandomParamDistribution>,
    n_iter: usize,
    seed: u64,
    data: SearchData<'_>,
    n_splits: usize,
    pct_embargo: f64,
    scoring: SearchScoring,
) -> Result<SearchResult, String>
where
    C: SimpleClassifier,
    F: Fn(&ParamSet) -> C,
{
    if param_space.is_empty() {
        return Err("param_space cannot be empty".to_string());
    }
    if n_iter == 0 {
        return Err("n_iter must be > 0".to_string());
    }

    let mut rng = StdRng::seed_from_u64(seed);
    let keys: Vec<String> = param_space.keys().cloned().collect();
    let mut params = Vec::with_capacity(n_iter);
    for _ in 0..n_iter {
        let mut draw = ParamSet::new();
        for key in &keys {
            let dist = param_space
                .get(key)
                .ok_or_else(|| format!("missing distribution for key '{key}'"))?;
            let value = sample_distribution(dist, &mut rng)?;
            draw.insert(key.clone(), value);
        }
        params.push(draw);
    }

    search_over_params(build_classifier, params, data, n_splits, pct_embargo, scoring)
}

fn sample_distribution<R: Rng + ?Sized>(
    dist: &RandomParamDistribution,
    rng: &mut R,
) -> Result<HyperParamValue, String> {
    match dist {
        RandomParamDistribution::Choice(values) => {
            if values.is_empty() {
                return Err("choice distribution cannot be empty".to_string());
            }
            let idx = rng.gen_range(0..values.len());
            Ok(values[idx].clone())
        }
        RandomParamDistribution::Uniform { low, high } => {
            if !low.is_finite() || !high.is_finite() || low >= high {
                return Err("uniform bounds must be finite and satisfy low < high".to_string());
            }
            Ok(HyperParamValue::Float(rng.gen_range(*low..*high)))
        }
        RandomParamDistribution::LogUniform { low, high } => {
            let v = sample_log_uniform(*low, *high, rng)?;
            Ok(HyperParamValue::Float(v))
        }
        RandomParamDistribution::IntRangeInclusive { low, high } => {
            if low > high {
                return Err("IntRangeInclusive requires low <= high".to_string());
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
) -> Result<SearchResult, String>
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
        .ok_or_else(|| "no trials produced".to_string())?;

    Ok(SearchResult { best_params: best.params, best_score: best.mean_score, trials })
}

fn validate_search_data(data: &SearchData<'_>, n_splits: usize) -> Result<(), String> {
    if data.x.is_empty() {
        return Err("x cannot be empty".to_string());
    }
    if data.y.is_empty() {
        return Err("y cannot be empty".to_string());
    }
    if data.x.len() != data.y.len() {
        return Err("x/y length mismatch".to_string());
    }
    if data.samples_info_sets.len() != data.x.len() {
        return Err("samples_info_sets length must match x length".to_string());
    }
    if n_splits < 2 {
        return Err("n_splits must be >= 2".to_string());
    }
    if let Some(sw) = data.sample_weight {
        if sw.len() != data.y.len() {
            return Err("sample_weight length mismatch".to_string());
        }
        if sw.iter().any(|w| *w < 0.0) {
            return Err("sample_weight cannot contain negative values".to_string());
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
) -> Result<Vec<f64>, String>
where
    C: SimpleClassifier,
    F: Fn(&ParamSet) -> C,
{
    let mut fold_scores = Vec::with_capacity(splits.len());

    for (train_idx, test_idx) in splits {
        if train_idx.is_empty() || test_idx.is_empty() {
            return Err("PurgedKFold generated an empty train/test fold".to_string());
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

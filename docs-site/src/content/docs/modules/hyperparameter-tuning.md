---
title: "hyperparameter_tuning"
description: "Leakage-aware grid/randomized hyper-parameter search with purged CV and weighted scoring."
status: generated
generated_from: src/data/moduleDocs.ts
last_generated: '2026-08-30'
banner:
  content: '<span class="doc-status doc-status--generated">Generated</span> Assembled automatically from <code>moduleDocs.ts</code>. No human has reviewed this page.'
audience:
  - quant-dev
  - platform-engineering
module: "hyperparameter_tuning"
api_surface: "rust-only"
risk_notes:
  - "Use Accuracy only when each prediction has similar economic value (equal bet sizing)."
  - "Prefer weighted NegLogLoss when probabilities drive position sizing or outcomes have different economic magnitude."
  - "BalancedAccuracy is useful for severe class imbalance, especially in meta-labeling where recall of positives matters."
rust_api:
  - "grid_search"
  - "randomized_search"
  - "expand_param_grid"
  - "sample_log_uniform"
  - "classification_score"
  - "SearchScoring"
  - "RandomParamDistribution"
sidebar:
  badge: Module
---

## Subject

**Sampling, Validation and ML Diagnostics**

## Why This Module Exists

AFML Chapter 9 recommends tuning under PurgedKFold, using randomized search for large spaces, and scoring with metrics aligned to trading objectives.

## Mathematical Foundations

### Purged CV Objective

$$\hat\theta=\arg\max_{\theta\in\Theta}\frac{1}{K}\sum_{k=1}^{K}\mathrm{Score}(f_\theta,\mathcal T_k^{train},\mathcal T_k^{test})$$

### Log-Uniform Draw

$$\log x\sim U(\log a,\log b),\; a>0,\;x\in(a,b)$$

### Weighted Neg Log Loss

$$-\frac{1}{\sum_i w_i}\sum_i w_i\left[y_i\log p_i + (1-y_i)\log(1-p_i)\right]$$

## Usage Examples

### Rust

#### Randomized search with PurgedKFold semantics

```rust
use chrono::{Duration, NaiveDateTime};
use openquant::cross_validation::SimpleClassifier;
use openquant::hyperparameter_tuning::{
    randomized_search, ParamSet, RandomParamDistribution, SearchData, SearchScoring,
};
use std::collections::BTreeMap;

// The search builds a fresh model from each sampled parameter set.
struct Logistic {
    c: f64,
}
impl SimpleClassifier for Logistic {
    fn fit(&mut self, _x: &[Vec<f64>], _y: &[f64], _sample_weight: Option<&[f64]>) {}
    fn predict_proba(&self, x: &[Vec<f64>]) -> Vec<f64> {
        x.iter().map(|row| 1.0 / (1.0 + (-self.c * row[0]).exp())).collect()
    }
}
let build_model =
    |params: &ParamSet| Logistic { c: params["C"].as_f64().unwrap_or(1.0) };

let mut space = BTreeMap::new();
space.insert("C".to_string(), RandomParamDistribution::LogUniform { low: 1e-2, high: 1e2 });
space.insert("gamma".to_string(), RandomParamDistribution::LogUniform { low: 1e-3, high: 1e1 });

let t0 = NaiveDateTime::parse_from_str("2024-01-02 00:00:00", "%Y-%m-%d %H:%M:%S")?;
let x: Vec<Vec<f64>> = (0..60).map(|i| vec![(i as f64 - 30.0) / 30.0]).collect();
let y: Vec<f64> = (0..60).map(|i| if i >= 30 { 1.0 } else { 0.0 }).collect();
let w = vec![1.0f64; 60];
// Label spans again — the search purges internally, so it needs them.
let info_sets: Vec<(NaiveDateTime, NaiveDateTime)> =
    (0..60).map(|i| (t0 + Duration::days(i), t0 + Duration::days(i + 2))).collect();

let result = randomized_search(
    build_model,
    &space,
    25,   // n_iter — parameter sets sampled
    42,   // seed
    SearchData { x: &x, y: &y, sample_weight: Some(&w), samples_info_sets: &info_sets },
    5,    // n_splits
    0.01, // pct_embargo
    SearchScoring::NegLogLoss,
)?;
println!("best score = {} with {:?}", result.best_score, result.best_params);
```

## API Reference

### Rust API

- `grid_search`
- `randomized_search`
- `expand_param_grid`
- `sample_log_uniform`
- `classification_score`
- `SearchScoring`
- `RandomParamDistribution`

## Implementation Notes

- Use Accuracy only when each prediction has similar economic value (equal bet sizing).
- Prefer weighted NegLogLoss when probabilities drive position sizing or outcomes have different economic magnitude.
- BalancedAccuracy is useful for severe class imbalance, especially in meta-labeling where recall of positives matters.

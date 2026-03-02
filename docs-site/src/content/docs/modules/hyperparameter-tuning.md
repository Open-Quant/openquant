---
title: "hyperparameter_tuning"
description: "Leakage-aware grid/randomized hyper-parameter search with purged CV and weighted scoring."
status: validated
last_validated: '2026-03-02'
audience:
  - quant-dev
  - platform-engineering
module: "hyperparameter_tuning"
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

## Key Public APIs

- `grid_search`
- `randomized_search`
- `expand_param_grid`
- `sample_log_uniform`
- `classification_score`
- `SearchScoring`
- `RandomParamDistribution`

## Mathematical Definitions

### Purged CV Objective

$$\hat\theta=\arg\max_{\theta\in\Theta}\frac{1}{K}\sum_{k=1}^{K}\mathrm{Score}(f_\theta,\mathcal T_k^{train},\mathcal T_k^{test})$$

### Log-Uniform Draw

$$\log x\sim U(\log a,\log b),\; a>0,\;x\in(a,b)$$

### Weighted Neg Log Loss

$$-\frac{1}{\sum_i w_i}\sum_i w_i\left[y_i\log p_i + (1-y_i)\log(1-p_i)\right]$$

## Implementation Examples

### Randomized search with PurgedKFold semantics

```rust
use std::collections::BTreeMap;
use openquant::hyperparameter_tuning::{
  randomized_search, RandomParamDistribution, SearchData, SearchScoring,
};

let mut space = BTreeMap::new();
space.insert("C".to_string(), RandomParamDistribution::LogUniform { low: 1e-2, high: 1e2 });
space.insert("gamma".to_string(), RandomParamDistribution::LogUniform { low: 1e-3, high: 1e1 });

let result = randomized_search(
  build_model,
  &space,
  25,
  42,
  SearchData { x: &x, y: &y, sample_weight: Some(&w), samples_info_sets: &info_sets },
  5,
  0.01,
  SearchScoring::NegLogLoss,
)?;
println!("best score = {}", result.best_score);
```

## Implementation Notes

- Use Accuracy only when each prediction has similar economic value (equal bet sizing).
- Prefer weighted NegLogLoss when probabilities drive position sizing or outcomes have different economic magnitude.
- BalancedAccuracy is useful for severe class imbalance, especially in meta-labeling where recall of positives matters.

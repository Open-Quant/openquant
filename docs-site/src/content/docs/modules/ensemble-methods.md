---
title: "ensemble_methods"
description: "Bias/variance diagnostics and practical bagging-vs-boosting ensemble utilities."
status: validated
last_validated: '2026-03-02'
audience:
  - quant-dev
  - platform-engineering
module: "ensemble_methods"
risk_notes:
  - "If base learners are highly correlated, bagging variance reduction is minimal even with many estimators."
  - "Sequential-bootstrap-style sampling is preferable under heavy label overlap and non-IID observations."
  - "Boosting is usually preferable for weak learners (bias reduction); bagging is usually preferable for unstable learners (variance reduction)."
rust_api:
  - "bias_variance_noise"
  - "bootstrap_sample_indices"
  - "sequential_bootstrap_sample_indices"
  - "aggregate_classification_vote"
  - "aggregate_classification_probability_mean"
  - "average_pairwise_prediction_correlation"
  - "bagging_ensemble_variance"
  - "recommend_bagging_vs_boosting"
sidebar:
  badge: Module
---

## Subject

**Sampling, Validation and ML Diagnostics**

## Why This Module Exists

AFML Chapter 6 emphasizes that ensemble gains depend on error decomposition and forecast dependence, not just estimator count.

## Mathematical Foundations

### Error Decomposition

$$\operatorname{MSE}=\operatorname{Bias}^2+\operatorname{Var}+\operatorname{Noise}$$

### Bagging Variance Under Average Correlation

$$\sigma^2_{bag}=\sigma^2\left(\rho+\frac{1-\rho}{N}\right)$$

### Majority Vote and Mean Probability

$$\hat y=\mathbf 1\left(\frac{1}{N}\sum_{m=1}^N \hat p_m \ge \tau\right),\quad \hat p=\frac{1}{N}\sum_{m=1}^N \hat p_m$$

## Usage Examples

### Rust

#### Assess Ensemble Variance and Recommendation

```rust
use openquant::ensemble_methods::{
  average_pairwise_prediction_correlation,
  bagging_ensemble_variance,
  recommend_bagging_vs_boosting,
};

let preds = vec![
  vec![0.51, 0.49, 0.52, 0.50],
  vec![0.50, 0.48, 0.53, 0.49],
  vec![0.52, 0.50, 0.51, 0.50],
];

let rho = average_pairwise_prediction_correlation(&preds)?;
let bag_var = bagging_ensemble_variance(1.0, rho, 20)?;
let decision = recommend_bagging_vs_boosting(0.54, rho, 0.75, 1.0, 20)?;

println!("rho={rho:.3}, var={bag_var:.3}, rec={:?}", decision.recommended);
```

#### Aggregate Bagged Classifier Outputs

```rust
use openquant::ensemble_methods::{
  aggregate_classification_vote,
  aggregate_classification_probability_mean,
};

let vote = aggregate_classification_vote(&[
  vec![1, 0, 1],
  vec![1, 1, 0],
  vec![0, 1, 1],
])?;

let (mean_prob, labels) = aggregate_classification_probability_mean(&[
  vec![0.9, 0.2, 0.6],
  vec![0.8, 0.3, 0.5],
  vec![0.7, 0.4, 0.4],
], 0.5)?;

assert_eq!(vote, vec![1, 1, 1]);
assert_eq!(labels, vec![1, 0, 1]);
assert_eq!(mean_prob.len(), 3);
```

## API Reference

### Rust API

- `bias_variance_noise`
- `bootstrap_sample_indices`
- `sequential_bootstrap_sample_indices`
- `aggregate_classification_vote`
- `aggregate_classification_probability_mean`
- `average_pairwise_prediction_correlation`
- `bagging_ensemble_variance`
- `recommend_bagging_vs_boosting`

## Implementation Notes

- If base learners are highly correlated, bagging variance reduction is minimal even with many estimators.
- Sequential-bootstrap-style sampling is preferable under heavy label overlap and non-IID observations.
- Boosting is usually preferable for weak learners (bias reduction); bagging is usually preferable for unstable learners (variance reduction).

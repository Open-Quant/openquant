---
title: "ensemble_methods"
description: "Bias/variance diagnostics and practical bagging-vs-boosting ensemble utilities."
status: generated
generated_from: src/data/moduleDocs.ts
last_generated: '2026-08-30'
banner:
  content: '<span class="doc-status doc-status--generated">Generated</span> Assembled automatically from <code>moduleDocs.ts</code>. No human has reviewed this page.'
audience:
  - quant-dev
  - platform-engineering
module: "ensemble_methods"
api_surface: "both"
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

## Concept Overview

The diagnostics behind the bagging-versus-boosting choice rather than another ensemble implementation. `bias_variance_noise` decomposes the error; `average_pairwise_prediction_correlation` measures how correlated your base learners actually are; `bagging_ensemble_variance` turns that rho into the variance a bagged ensemble can reach, sigma^2(rho + (1-rho)/N). The consequence AFML Chapter 6 draws is the useful one: as N grows the ensemble variance floors at sigma^2·rho, so with highly correlated learners more estimators buy nothing at all.

## When to Use

Use it before scaling an ensemble. If measured rho is 0.9, going from 20 to 200 estimators is wasted compute, and `recommend_bagging_vs_boosting` will say so from the numbers rather than from folklore. Reach for bagging when the base learner is unstable (variance-dominated) and boosting when it is weak (bias-dominated). Under heavy label overlap use `sequential_bootstrap_sample_indices` instead of the IID bootstrap, or the bags will be near-duplicates of each other.

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

### Python API

- `ensemble.bias_variance_noise`
- `ensemble.bootstrap_sample_indices`
- `ensemble.sequential_bootstrap_sample_indices`
- `ensemble.aggregate_regression_mean`
- `ensemble.aggregate_classification_vote`
- `ensemble.aggregate_classification_probability_mean`
- `ensemble.average_pairwise_prediction_correlation`
- `ensemble.bagging_ensemble_variance`
- `ensemble.recommend_bagging_vs_boosting`

### Rust API

- `bias_variance_noise`
- `bootstrap_sample_indices`
- `sequential_bootstrap_sample_indices`
- `aggregate_classification_vote`
- `aggregate_classification_probability_mean`
- `average_pairwise_prediction_correlation`
- `bagging_ensemble_variance`
- `recommend_bagging_vs_boosting`

## Risk Notes and Caveats

- If base learners are highly correlated, bagging variance reduction is minimal even with many estimators.
- Sequential-bootstrap-style sampling is preferable under heavy label overlap and non-IID observations.
- Boosting is usually preferable for weak learners (bias reduction); bagging is usually preferable for unstable learners (variance reduction).

## Related Modules

- [`sb-bagging`](/modules/sb-bagging/)
- [`sampling`](/modules/sampling/)
- [`sample-weights`](/modules/sample-weights/)
- [`cross-validation`](/modules/cross-validation/)
- [`feature-importance`](/modules/feature-importance/)

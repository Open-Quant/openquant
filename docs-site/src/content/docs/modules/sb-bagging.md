---
title: "sb_bagging"
description: "Sequentially bootstrapped bagging classifiers/regressors."
status: generated
generated_from: src/data/moduleDocs.ts
last_generated: '2026-08-30'
banner:
  content: '<span class="doc-status doc-status--generated">Generated</span> Assembled automatically from <code>moduleDocs.ts</code>. No human has reviewed this page.'
audience:
  - quant-dev
  - platform-engineering
module: "sb_bagging"
api_surface: "both"
risk_notes:
  - "Sequential bootstrap improves diversity under event overlap."
  - "Tune max_samples/max_features with out-of-sample monitoring."
rust_api:
  - "SequentiallyBootstrappedBaggingClassifier"
  - "SequentiallyBootstrappedBaggingRegressor"
  - "MaxSamples"
  - "MaxFeatures"
sidebar:
  badge: Module
---

## Subject

**Sampling, Validation and ML Diagnostics**

## Why This Module Exists

Combines ensemble variance reduction with overlap-aware sampling.

## Mathematical Foundations

### Bagging Predictor

$$\hat f(x)=\frac{1}{B}\sum_{b=1}^{B} f_b(x)$$

### Bootstrap Sampling

$$S_b\sim P_{seq}(u)$$

## Usage Examples

### Rust

#### Instantiate SB bagging classifier

```rust
use openquant::sb_bagging::SequentiallyBootstrappedBaggingClassifier;

// The single constructor argument is `random_state` — NOT the ensemble size.
// n_estimators defaults to 10 and has to be set explicitly.
let mut bag = SequentiallyBootstrappedBaggingClassifier::new(42);
bag.n_estimators = 100;
bag.oob_score = true;

println!("{} estimators, seed {}", bag.n_estimators, bag.random_state);
```

## API Reference

### Python API

- `sb_bagging.fit_predict_sb_classifier`
- `sb_bagging.fit_predict_sb_regressor`

### Rust API

- `SequentiallyBootstrappedBaggingClassifier`
- `SequentiallyBootstrappedBaggingRegressor`
- `MaxSamples`
- `MaxFeatures`

## Implementation Notes

- Sequential bootstrap improves diversity under event overlap.
- Tune max_samples/max_features with out-of-sample monitoring.

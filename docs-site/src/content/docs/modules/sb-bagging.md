---
title: "sb_bagging"
description: "Sequentially bootstrapped bagging classifiers/regressors."
status: validated
last_validated: '2026-03-02'
audience:
  - quant-dev
  - platform-engineering
module: "sb_bagging"
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

let bag = SequentiallyBootstrappedBaggingClassifier::new(100);
```

## API Reference

### Rust API

- `SequentiallyBootstrappedBaggingClassifier`
- `SequentiallyBootstrappedBaggingRegressor`
- `MaxSamples`
- `MaxFeatures`

## Implementation Notes

- Sequential bootstrap improves diversity under event overlap.
- Tune max_samples/max_features with out-of-sample monitoring.

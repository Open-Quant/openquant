---
title: "feature_importance"
description: "Feature ranking methods: MDI, MDA, and single-feature importance with PCA diagnostics."
status: validated
last_validated: '2026-03-02'
audience:
  - quant-dev
  - platform-engineering
module: "feature_importance"
risk_notes:
  - "Cross-validated MDA is preferred when leakage risk is high."
  - "Compare ranking stability across folds/time windows."
rust_api:
  - "mean_decrease_impurity"
  - "mean_decrease_accuracy"
  - "single_feature_importance"
  - "feature_pca_analysis"
sidebar:
  badge: Module
---

## Subject

**Sampling, Validation and ML Diagnostics**

## Why This Module Exists

Improves model interpretability and helps remove unstable or redundant features.

## Mathematical Foundations

### MDI

$$I_j=\sum_{t\in T_j} p(t)\Delta i(t)$$

### MDA

$$I_j=Score(X)-Score(X_{perm(j)})$$

## Usage Examples

### Rust

#### Run MDA with classifier

```rust
use openquant::feature_importance::mean_decrease_accuracy;

// Plug in your classifier implementing SimpleClassifier
let importance = mean_decrease_accuracy(&clf, &x, &y, 5)?;
```

## API Reference

### Rust API

- `mean_decrease_impurity`
- `mean_decrease_accuracy`
- `single_feature_importance`
- `feature_pca_analysis`

## Implementation Notes

- Cross-validated MDA is preferred when leakage risk is high.
- Compare ranking stability across folds/time windows.

---
title: "feature_importance"
description: "Feature ranking methods: MDI, MDA, and single-feature importance with PCA diagnostics."
status: generated
generated_from: src/data/moduleDocs.ts
last_generated: '2026-08-30'
banner:
  content: '<span class="doc-status doc-status--generated">Generated</span> Assembled automatically from <code>moduleDocs.ts</code>. No human has reviewed this page.'
audience:
  - quant-dev
  - platform-engineering
module: "feature_importance"
api_surface: "rust-only"
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
use openquant::cross_validation::{Scoring, SimpleClassifier};
use openquant::feature_importance::mean_decrease_accuracy;

// MDA works with any model implementing SimpleClassifier; this stand-in keeps
// the example self-contained.
struct MeanThreshold {
    threshold: f64,
}
impl SimpleClassifier for MeanThreshold {
    fn fit(&mut self, x: &[Vec<f64>], _y: &[f64], _sample_weight: Option<&[f64]>) {
        self.threshold = x.iter().map(|row| row[0]).sum::<f64>() / x.len() as f64;
    }
    fn predict_proba(&self, x: &[Vec<f64>]) -> Vec<f64> {
        x.iter().map(|row| if row[0] > self.threshold { 0.9 } else { 0.1 }).collect()
    }
}

let x: Vec<Vec<f64>> = (0..40).map(|i| vec![i as f64, (i % 7) as f64]).collect();
let y: Vec<f64> = (0..40).map(|i| if i >= 20 { 1.0 } else { 0.0 }).collect();
let feature_names = vec!["trend".to_string(), "noise".to_string()];

// MDA is measured out of sample, so it takes the *already-purged splits* — not a
// fold count. Feed it the output of PurgedKFold::split so the score is leak-free.
let splits = vec![
    ((0..20).collect::<Vec<usize>>(), (20..40).collect::<Vec<usize>>()),
    ((20..40).collect::<Vec<usize>>(), (0..20).collect::<Vec<usize>>()),
];

let mut model = MeanThreshold { threshold: 0.0 };
let importance = mean_decrease_accuracy(
    &mut model,
    &x,
    &y,
    &feature_names,
    &splits,
    None, // sample_weight — pass uniqueness weights from `sample_weights` in practice
    Scoring::Accuracy,
)?;

println!("trend: mean={:.4} std={:.4}", importance["trend"].mean, importance["trend"].std);
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

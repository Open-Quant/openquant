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
rust_api:
  - "mean_decrease_impurity"
  - "mean_decrease_accuracy"
  - "single_feature_importance"
  - "feature_pca_analysis"
sidebar:
  badge: Module
---

## Concept Overview

The three AFML Chapter 8 importance methods on the Rust side, each with a different blind spot. MDI is in-sample and tree-specific: it sums each feature's impurity decrease across splits, cheap but defeated by substitution, since two interchangeable features split the credit and both then look weak. MDA permutes a feature in the *test* fold and measures the score drop, so it is model-agnostic and out-of-sample but still substitution-prone. Single-feature importance trains on one feature at a time, immune to substitution but blind to interactions. `feature_pca_analysis` cross-checks the ranking against an unsupervised one.

## When to Use

Run at least two of the three: agreement between MDI and MDA is evidence, MDI alone is not. Prefer MDA when leakage risk is high, since it is the only one scored out of sample — and give it purged splits from `cross_validation`, not a fold count. Compare rankings across time windows before trusting them; a feature that is important in only one regime is a feature that will fail in the next.

## Mathematical Foundations

### MDI — Mean Decrease Impurity

$$I_j=\frac{1}{B}\sum_{b=1}^{B}\;\sum_{t\in T_j^{(b)}} p(t)\,\Delta i(t)$$

where $T_j^{(b)}$ are the nodes of tree $b$ that split on feature $j$, $p(t)$ the fraction of samples reaching node $t$, and $\Delta i(t)$ the impurity drop at that split. This is the tree-based definition: it is in-sample, computable only for tree ensembles, and `mean_decrease_impurity` takes the per-tree importance vectors a fitted forest already exposes. The Python `feature_diagnostics.mdi_importance` uses a different, linear-model estimator under the same acronym — see that page.

### MDA — Mean Decrease Accuracy

$$I_j=\frac{1}{K}\sum_{k=1}^{K}\big(S_k-S_{k,\text{perm}(j)}\big)$$

where $S_k$ is the out-of-sample score on purged fold $k$ and $S_{k,\text{perm}(j)}$ the same score after column $j$ is randomly permuted in the test set. Unlike MDI it is model-agnostic and out-of-sample, which is why `mean_decrease_accuracy` demands the CV splits rather than a fold count.

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

## Risk Notes and Caveats

- Cross-validated MDA is preferred when leakage risk is high.
- Compare ranking stability across folds/time windows.

## Related Modules

- [`feature-diagnostics`](/modules/feature-diagnostics/)
- [`cross-validation`](/modules/cross-validation/)
- [`sample-weights`](/modules/sample-weights/)
- [`codependence`](/modules/codependence/)
- [`fingerprint`](/modules/fingerprint/)

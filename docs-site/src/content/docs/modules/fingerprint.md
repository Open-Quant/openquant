---
title: "fingerprint"
description: "Model fingerprinting for linear, non-linear, and pairwise feature effects."
status: generated
generated_from: src/data/moduleDocs.ts
last_generated: '2026-08-30'
banner:
  content: '<span class="doc-status doc-status--generated">Generated</span> Assembled automatically from <code>moduleDocs.ts</code>. No human has reviewed this page.'
audience:
  - quant-dev
  - platform-engineering
module: "fingerprint"
api_surface: "rust-only"
risk_notes:
  - "Compare fingerprints across retrains for drift detection."
  - "Use pairwise effects to detect hidden interaction risk."
rust_api:
  - "RegressionModelFingerprint"
  - "ClassificationModelFingerprint"
  - "Effect"
  - "PairwiseEffect"
sidebar:
  badge: Module
---

## Subject

**Sampling, Validation and ML Diagnostics**

## Why This Module Exists

Quantifies behavior of fitted models beyond scalar accuracy metrics.

## Mathematical Foundations

### Partial Effect

$$f_j(x_j)=E_{X_{-j}}[f(X)|X_j=x_j]$$

### Pairwise Interaction

$$I_{ij}=f(x_i,x_j)-f_i(x_i)-f_j(x_j)$$

## Usage Examples

### Rust

#### Create regression fingerprint

```rust
use openquant::fingerprint::{RegressionModelFingerprint, RegressionPredictor};

// Fingerprinting is model-agnostic: anything that can predict will do.
struct LinearModel {
    beta: Vec<f64>,
}
impl RegressionPredictor for LinearModel {
    fn predict(&self, x: &[Vec<f64>]) -> Vec<f64> {
        x.iter()
            .map(|row| row.iter().zip(self.beta.iter()).map(|(v, b)| v * b).sum())
            .collect()
    }
}

let model = LinearModel { beta: vec![1.5, -0.5] };
let x: Vec<Vec<f64>> =
    (0..50).map(|i| vec![i as f64 / 50.0, ((i % 5) as f64) / 5.0]).collect();

// new() takes no arguments; the model and data go to fit(), which needs &mut self.
// num_values is the partial-dependence grid resolution.
let mut fingerprint = RegressionModelFingerprint::new();
fingerprint.fit(&model, &x, 10, Some(&[(0, 1)]))?;

// The accessor is get_effects(), returning (linear, non-linear, optional pairwise).
let (linear, non_linear, pairwise) = fingerprint.get_effects()?;
println!("linear={:?}", linear.norm);
println!("non_linear={:?}", non_linear.norm);
println!("pairwise={:?}", pairwise.map(|p| p.norm.clone()));
```

## API Reference

### Rust API

- `RegressionModelFingerprint`
- `ClassificationModelFingerprint`
- `Effect`
- `PairwiseEffect`

## Implementation Notes

- Compare fingerprints across retrains for drift detection.
- Use pairwise effects to detect hidden interaction risk.

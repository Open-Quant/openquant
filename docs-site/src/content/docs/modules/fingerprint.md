---
title: "fingerprint"
description: "Model fingerprinting for linear, non-linear, and pairwise feature effects."
status: generated
generated_from: src/data/moduleDocs.ts
last_generated: '2026-08-31'
audience:
  - quant-dev
  - platform-engineering
module: "fingerprint"
api_surface: "rust-only"
rust_api:
  - "RegressionModelFingerprint"
  - "ClassificationModelFingerprint"
  - "Effect"
  - "PairwiseEffect"
sidebar:
  badge: Module
---

## Concept Overview

Model fingerprinting decomposes a fitted model's behaviour into a linear effect, a non-linear effect and pairwise interaction effects per feature, by sweeping each feature across a grid and measuring how the prediction moves. The result describes *what the model learned* rather than how well it scored — two models with identical accuracy can have entirely different fingerprints, and only one of them may be relying on something that will still be there next quarter.

## When to Use

Use it after fitting and before deploying, and again on every retrain: comparing fingerprints across retrains is a drift signal that accuracy metrics do not give you. Use the pairwise effects to find interaction risk, since a large pairwise term means the model's response to one feature depends on another, which makes its extrapolation fragile. It works with any model — implement `RegressionPredictor` or `ClassificationPredictor` and pass it to `fit`.

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

## Risk Notes and Caveats

- Compare fingerprints across retrains for drift detection.
- Use pairwise effects to detect hidden interaction risk.

## Related Modules

- [`feature-importance`](/modules/feature-importance/)
- [`feature-diagnostics`](/modules/feature-diagnostics/)
- [`ensemble-methods`](/modules/ensemble-methods/)
- [`backtesting-engine`](/modules/backtesting-engine/)

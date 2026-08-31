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
use openquant::fingerprint::RegressionModelFingerprint;

let fp = RegressionModelFingerprint::new(&model, &x);
let effects = fp.linear_effects()?;
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

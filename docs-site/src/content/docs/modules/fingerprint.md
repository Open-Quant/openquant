---
title: "fingerprint"
description: "Model fingerprinting for linear, non-linear, and pairwise feature effects."
status: validated
last_validated: '2026-03-02'
audience:
  - quant-dev
  - platform-engineering
module: "fingerprint"
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

## Key Public APIs

- `RegressionModelFingerprint`
- `ClassificationModelFingerprint`
- `Effect`
- `PairwiseEffect`

## Mathematical Definitions

### Partial Effect

$$f_j(x_j)=E_{X_{-j}}[f(X)|X_j=x_j]$$

### Pairwise Interaction

$$I_{ij}=f(x_i,x_j)-f_i(x_i)-f_j(x_j)$$

## Implementation Examples

### Create regression fingerprint

```rust
use openquant::fingerprint::RegressionModelFingerprint;

let fp = RegressionModelFingerprint::new(&model, &x);
let effects = fp.linear_effects()?;
```

## Implementation Notes

- Compare fingerprints across retrains for drift detection.
- Use pairwise effects to detect hidden interaction risk.

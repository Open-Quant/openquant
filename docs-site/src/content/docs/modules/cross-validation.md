---
title: "cross_validation"
description: "Purged cross-validation utilities designed for label overlap and leakage control."
status: validated
last_validated: '2026-03-02'
audience:
  - quant-dev
  - platform-engineering
module: "cross_validation"
risk_notes:
  - "Always align event end-times when purging."
  - "Report variance across folds, not only mean score."
rust_api:
  - "ml_cross_val_score"
  - "ml_get_train_times"
  - "PurgedKFold"
  - "Scoring"
sidebar:
  badge: Module
---

## Subject

**Sampling, Validation and ML Diagnostics**

## Why This Module Exists

Time-dependent labels violate IID assumptions; purging/embargoing reduces leakage bias.

## Mathematical Foundations

### Purged Train Set

$$\mathcal{T}_{train}=\mathcal{T}\setminus(\mathcal{T}_{test}\oplus e)$$

### Embargo

$$e=\lfloor p\cdot T\rfloor$$

## Usage Examples

### Rust

#### Configure PurgedKFold

```rust
use openquant::cross_validation::PurgedKFold;

let cv = PurgedKFold::new(5, 0.01);
```

## API Reference

### Rust API

- `ml_cross_val_score`
- `ml_get_train_times`
- `PurgedKFold`
- `Scoring`

## Implementation Notes

- Always align event end-times when purging.
- Report variance across folds, not only mean score.

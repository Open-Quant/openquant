---
title: "ef3m"
description: "Moment-based mixture fitting utilities for two-normal components."
status: validated
last_validated: '2026-03-02'
audience:
  - quant-dev
  - platform-engineering
module: "ef3m"
risk_notes:
  - "Use as initialization for more expensive optimizers."
  - "Sensitive to higher-moment estimation noise."
rust_api:
  - "M2N"
  - "centered_moment"
  - "raw_moment"
  - "most_likely_parameters"
sidebar:
  badge: Module
---

## Subject

**Sampling, Validation and ML Diagnostics**

## Why This Module Exists

Provides robust parameter estimation for bimodal return mixtures when full MLE is heavy.

## Mathematical Foundations

### Raw Moment

$$m_k=E[X^k]$$

### Mixture Mean

$$\mu=p\mu_1+(1-p)\mu_2$$

## Usage Examples

### Rust

#### Estimate moments

```rust
use openquant::ef3m::centered_moment;

let moments = vec![0.0, 1.0, 0.1, 3.0];
let m3 = centered_moment(&moments, 3);
```

## API Reference

### Rust API

- `M2N`
- `centered_moment`
- `raw_moment`
- `most_likely_parameters`

## Implementation Notes

- Use as initialization for more expensive optimizers.
- Sensitive to higher-moment estimation noise.

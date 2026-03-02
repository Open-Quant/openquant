---
title: "filters"
description: "CUSUM and z-score event filters for event-driven sampling."
status: validated
last_validated: '2026-03-02'
audience:
  - quant-dev
  - platform-engineering
module: "filters"
risk_notes:
  - "Calibrate thresholds to target event frequency, not just sensitivity."
  - "Use identical filtering in train and live pipelines."
rust_api:
  - "cusum_filter_indices"
  - "cusum_filter_timestamps"
  - "z_score_filter_indices"
  - "Threshold"
sidebar:
  badge: Module
---

## Subject

**Event-Driven Data and Labeling**

## Why This Module Exists

Extracts informative events from noisy high-frequency sequences.

## Key Public APIs

- `cusum_filter_indices`
- `cusum_filter_timestamps`
- `z_score_filter_indices`
- `Threshold`

## Mathematical Definitions

### CUSUM

$$S_t=\max(0, S_{t-1}+r_t),\; trigger\;if\;|S_t|>h$$

### Z-score

$$z_t=\frac{x_t-\mu_t}{\sigma_t}$$

## Implementation Examples

### Run CUSUM over closes

```rust
use openquant::filters::{cusum_filter_indices, Threshold};

let close = vec![100.0, 100.1, 99.9, 100.2];
let idx = cusum_filter_indices(&close, Threshold::Scalar(0.02));
```

## Implementation Notes

- Calibrate thresholds to target event frequency, not just sensitivity.
- Use identical filtering in train and live pipelines.

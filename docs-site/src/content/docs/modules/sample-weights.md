---
title: "sample_weights"
description: "Sample weighting utilities for overlapping event structure."
status: validated
last_validated: '2026-03-02'
audience:
  - quant-dev
  - platform-engineering
module: "sample_weights"
risk_notes:
  - "Pair with sequential bootstrap for robust label sampling."
  - "Time-decay controls recency bias explicitly."
rust_api:
  - "get_weights_by_return"
  - "get_weights_by_time_decay"
sidebar:
  badge: Module
---

## Subject

**Event-Driven Data and Labeling**

## Why This Module Exists

Adjusts training influence to avoid overcounting dense overlapping labels.

## Key Public APIs

- `get_weights_by_return`
- `get_weights_by_time_decay`

## Mathematical Definitions

### Uniqueness Weight

$$w_i=\sum_t\frac{I_{t,i}}{\sum_j I_{t,j}}$$

### Time Decay

$$w_i=(\frac{i}{T})^\delta$$

## Implementation Examples

### Compute event weights

```rust
use openquant::sample_weights::get_weights_by_time_decay;

let w = get_weights_by_time_decay(&returns, 0.5);
```

## Implementation Notes

- Pair with sequential bootstrap for robust label sampling.
- Time-decay controls recency bias explicitly.

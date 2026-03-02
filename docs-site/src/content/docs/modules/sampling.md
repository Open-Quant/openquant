---
title: "sampling"
description: "Indicator matrix and sequential bootstrap tooling."
status: validated
last_validated: '2026-03-02'
audience:
  - quant-dev
  - platform-engineering
module: "sampling"
risk_notes:
  - "Indicator matrix quality drives bootstrap quality."
  - "Use average uniqueness as a diagnostics KPI."
rust_api:
  - "get_ind_matrix"
  - "seq_bootstrap"
  - "get_ind_mat_average_uniqueness"
  - "num_concurrent_events"
sidebar:
  badge: Module
---

## Subject

**Sampling, Validation and ML Diagnostics**

## Why This Module Exists

Produces less correlated training samples when labels overlap heavily in time.

## Key Public APIs

- `get_ind_matrix`
- `seq_bootstrap`
- `get_ind_mat_average_uniqueness`
- `num_concurrent_events`

## Mathematical Definitions

### Average Uniqueness

$$u_i=\frac{1}{|T_i|}\sum_{t\in T_i}\frac{1}{c_t}$$

### Sequential Draw Prob

$$P(i)\propto E[u_i \mid \mathcal{S}]$$

## Implementation Examples

### Run sequential bootstrap

```rust
use openquant::sampling::seq_bootstrap;

let ind = vec![vec![1,0,1], vec![0,1,1], vec![1,1,0]];
let idx = seq_bootstrap(&ind, Some(3), None);
```

## Implementation Notes

- Indicator matrix quality drives bootstrap quality.
- Use average uniqueness as a diagnostics KPI.

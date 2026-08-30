---
title: "cross_validation"
description: "Purged cross-validation utilities designed for label overlap and leakage control."
status: generated
generated_from: src/data/moduleDocs.ts
last_generated: '2026-08-30'
banner:
  content: '<span class="doc-status doc-status--generated">Generated</span> Assembled automatically from <code>moduleDocs.ts</code>. No human has reviewed this page.'
audience:
  - quant-dev
  - platform-engineering
module: "cross_validation"
api_surface: "rust-only"
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

$$\mathcal{T}_{\text{train}}=\mathcal{T}\setminus\{i:\;\exists j\in\mathcal{T}_{\text{test}},\;[t_{i,0},t_{i,1}]\cap[t_{j,0},t_{j,1}]\neq\varnothing\}\setminus\mathcal{E}$$

where $[t_{i,0},t_{i,1}]$ is observation $i$'s label span — the `samples_info_sets` entry `PurgedKFold::new` requires. *Purging* drops any training observation whose label lifetime overlaps a test label's; $\mathcal{E}$ is the embargo set below. Overlap, not adjacency, is what leaks: two observations sampled a month apart still share information if their labels resolve on the same bar.

### Embargo

$$e=\lfloor p\cdot T\rfloor,\qquad \mathcal{E}=\{i:\;\max(\mathcal{T}_{\text{test}})<i\le\max(\mathcal{T}_{\text{test}})+e\}$$

where $T$ is the total number of observations and $p$ the `pct_embargo` fraction (0.01 = 1%), so $e$ is an observation count. The embargo drops the $e$ observations immediately *after* each test fold, which catches serial correlation that purging alone misses because the label spans do not literally overlap.

## Usage Examples

### Rust

#### Configure PurgedKFold

```rust
use chrono::{Duration, NaiveDateTime};
use openquant::cross_validation::PurgedKFold;

let t0 = NaiveDateTime::parse_from_str("2024-01-02 00:00:00", "%Y-%m-%d %H:%M:%S")?;

// samples_info_sets is one (label_start, label_end) span per observation. It is
// mandatory: without label lifetimes there is nothing to purge against.
let samples_info_sets: Vec<(NaiveDateTime, NaiveDateTime)> = (0..100)
    .map(|i| (t0 + Duration::days(i), t0 + Duration::days(i + 3)))
    .collect();

// n_splits = 5 folds; pct_embargo = 0.01 drops a further 1% of the sample
// immediately after each test fold. new() validates and returns a Result.
let cv = PurgedKFold::new(5, samples_info_sets, 0.01)?;

let splits = cv.split(100)?;
println!("{} folds; fold 0 keeps {} training rows", splits.len(), splits[0].0.len());
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

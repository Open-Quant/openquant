---
title: "backtesting_engine"
description: "Backtesting core with walk-forward, purged CV, and combinatorial purged CV (CPCV) workflows."
status: validated
last_validated: '2026-03-02'
audience:
  - quant-dev
  - platform-engineering
module: "backtesting_engine"
risk_notes:
  - "Chapter 11: a backtest is a scenario sanity check; keep safeguards and assumptions attached to every run."
  - "Chapter 12: compare WF/CV/CPCV results by mode rather than averaging them into one statistic."
  - "CPCV output is a path distribution, enabling robust Sharpe diagnostics (e.g., quantiles) instead of point estimates."
rust_api:
  - "run_walk_forward"
  - "run_cross_validation"
  - "run_cpcv"
  - "cpcv_path_count"
  - "BacktestRunConfig"
  - "BacktestSafeguards"
  - "WalkForwardConfig"
  - "CrossValidationConfig"
  - "CpcvConfig"
sidebar:
  badge: Module
---

## Subject

**Sampling, Validation and ML Diagnostics**

## Why This Module Exists

AFML Chapters 11-12 require scenario-based validation with explicit anti-leakage controls, split provenance, and path-wise uncertainty rather than single-score reporting.

## Key Public APIs

- `run_walk_forward`
- `run_cross_validation`
- `run_cpcv`
- `cpcv_path_count`
- `BacktestRunConfig`
- `BacktestSafeguards`
- `WalkForwardConfig`
- `CrossValidationConfig`
- `CpcvConfig`

## Mathematical Definitions

### CPCV Path Count

$$\phi[N,k]=\binom{N}{k}\frac{k}{N}=\binom{N-1}{k-1}$$

### Purge + Embargo Train Set

$$\mathcal T_{train}^{*}=\mathcal T_{train}\setminus\{i: \exists j\in\mathcal T_{test},\;I_i\cap I_j\neq\varnothing\}\setminus\mathcal E(\mathcal T_{test},p)$$

### Per-Path Sharpe

$$S_{path}=\frac{\bar r_{path}}{\sigma_{path}}\sqrt{T_{path}}$$

## Implementation Examples

### Run CPCV and inspect Sharpe distribution

```rust
use openquant::backtesting_engine::{
  run_cpcv, BacktestData, BacktestRunConfig, BacktestSafeguards, CpcvConfig,
};

let result = run_cpcv(
  &data,
  &BacktestRunConfig {
    mode_provenance: "research_v3_with_costs".to_string(),
    trials_count: 24,
    safeguards: BacktestSafeguards {
      survivorship_bias_control: "point-in-time universe".to_string(),
      look_ahead_control: "lagged features".to_string(),
      data_mining_control: "frozen split protocol".to_string(),
      cost_assumption: "spread + slippage".to_string(),
      multiple_testing_control: "trial count logged".to_string(),
    },
  },
  &CpcvConfig { n_groups: 8, test_groups: 2, pct_embargo: 0.01 },
  |split| Ok(split.test_indices.iter().map(|i| pnl[*i]).collect()),
)?;

println!("phi = {}", result.path_count);
println!("path sharpe count = {}", result.path_distribution.len());
```

## Implementation Notes

- Chapter 11: a backtest is a scenario sanity check; keep safeguards and assumptions attached to every run.
- Chapter 12: compare WF/CV/CPCV results by mode rather than averaging them into one statistic.
- CPCV output is a path distribution, enabling robust Sharpe diagnostics (e.g., quantiles) instead of point estimates.

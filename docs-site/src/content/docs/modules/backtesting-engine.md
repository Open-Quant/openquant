---
title: "backtesting_engine"
description: "Backtesting core with walk-forward, purged CV, and combinatorial purged CV (CPCV) workflows."
status: generated
generated_from: src/data/moduleDocs.ts
last_generated: '2026-08-31'
audience:
  - quant-dev
  - platform-engineering
module: "backtesting_engine"
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

## Concept Overview

Three validation modes over one data contract: walk-forward, purged k-fold cross-validation, and combinatorial purged CV. CPCV is the one that justifies the extra cost — instead of a single backtest path it produces phi[N,k] = C(N-1, k-1) paths, so the output is a *distribution* of per-path Sharpe ratios you can take quantiles of rather than a point estimate you can fool yourself with. Every run carries a `BacktestSafeguards` record (survivorship, look-ahead, data-mining, cost and multiple-testing controls) so the assumptions travel attached to the number.

## When to Use

Use walk-forward when the question is "would this have worked as deployed"; use purged CV when you need many folds out of limited data; use CPCV when you are about to make a go/no-go decision and need to know how much of the reported Sharpe is path luck. All three require `label_spans` — the label lifetimes — not just observation timestamps, because that is what purging acts on. Compare the three modes against each other rather than averaging them into one statistic.

## Mathematical Foundations

### CPCV Path Count

$$\phi[N,k]=\binom{N}{k}\frac{k}{N}=\binom{N-1}{k-1}$$

### Purge + Embargo Train Set

$$\mathcal T_{train}^{*}=\mathcal T_{train}\setminus\{i: \exists j\in\mathcal T_{test},\;I_i\cap I_j\neq\varnothing\}\setminus\mathcal E(\mathcal T_{test},p)$$

### Per-Path Sharpe

$$S_{path}=\frac{\bar r_{path}}{\sigma_{path}}\sqrt{T_{path}}$$

## Usage Examples

### Rust

#### Run CPCV and inspect Sharpe distribution

```rust
use chrono::{Duration, NaiveDateTime};
use openquant::backtesting_engine::{
    run_cpcv, BacktestData, BacktestRunConfig, BacktestSafeguards, CpcvConfig,
};

let t0 = NaiveDateTime::parse_from_str("2024-01-02 00:00:00", "%Y-%m-%d %H:%M:%S")?;
let pnl: Vec<f64> = (0..240).map(|i| ((i % 7) as f64 - 3.0) / 1000.0).collect();

// Each observation carries the span its label was drawn over. That span — not the
// observation's timestamp — is what purging and the embargo act on.
let data = BacktestData {
    returns: pnl.clone(),
    label_spans: (0..240)
        .map(|i| (t0 + Duration::days(i), t0 + Duration::days(i + 2)))
        .collect(),
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

## API Reference

### Rust API

- `run_walk_forward`
- `run_cross_validation`
- `run_cpcv`
- `cpcv_path_count`
- `BacktestRunConfig`
- `BacktestSafeguards`
- `WalkForwardConfig`
- `CrossValidationConfig`
- `CpcvConfig`

## Risk Notes and Caveats

- Chapter 11: a backtest is a scenario sanity check; keep safeguards and assumptions attached to every run.
- Chapter 12: compare WF/CV/CPCV results by mode rather than averaging them into one statistic.
- CPCV output is a path distribution, enabling robust Sharpe diagnostics (e.g., quantiles) instead of point estimates.

## Related Modules

- [`cross-validation`](/modules/cross-validation/)
- [`sample-weights`](/modules/sample-weights/)
- [`backtest-statistics`](/modules/backtest-statistics/)
- [`synthetic-backtesting`](/modules/synthetic-backtesting/)
- [`hyperparameter-tuning`](/modules/hyperparameter-tuning/)

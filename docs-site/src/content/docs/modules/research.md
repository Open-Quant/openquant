---
title: "research"
description: "Synthetic dataset generation and flywheel research iteration with cost modeling and promotion gates."
status: validated
last_validated: '2026-03-02'
audience:
  - quant-dev
  - platform-engineering
module: "research"
api_surface: "python-only"
risk_notes:
  - "make_synthetic_futures_dataset is deterministic given seed — use for regression tests and reproducible notebooks."
  - "run_flywheel_iteration includes turnover estimation, transaction cost modeling, and net-of-cost Sharpe."
  - "Promotion gates check realized Sharpe, net Sharpe, and leakage guards before flagging a strategy as deployment-ready."
rust_api:
  - "make_synthetic_futures_dataset"
  - "run_flywheel_iteration"
  - "ResearchDataset"
sidebar:
  badge: Module
---

## Concept Overview

The research module implements the "research flywheel" pattern: a tight loop of hypothesis → synthetic test → cost estimation → promotion gate. It wraps the pipeline module with additional cost modeling (commissions, spread, slippage proportional to realized volatility) and strategy-readiness checks.

`make_synthetic_futures_dataset` generates a deterministic multi-asset futures dataset with realistic properties (seasonal patterns, correlated assets, noisy model forecasts). This lets you develop and test research workflows without real market data, and provides a stable baseline for regression testing.

`run_flywheel_iteration` runs the full pipeline, computes turnover and estimated transaction costs, calculates net-of-cost Sharpe, and evaluates promotion criteria. The result tells you whether a strategy variant passes minimum viability thresholds.

## When to Use

Use this for rapid strategy research iteration, especially during development when you don't have (or don't want to use) real market data. Also useful for CI regression tests and notebook tutorials.

**Prerequisites**: None for synthetic data. For real data, construct a ResearchDataset from your own prices and model forecasts.

## Key Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `n_bars` | `int` | Number of bars in synthetic dataset | 192 |
| `seed` | `int` | Random seed for reproducibility | 7 |
| `commission_bps` | `float` | Commission in basis points per turn | 1.5 |
| `spread_bps` | `float` | Spread cost in basis points | 2.0 |
| `min_net_sharpe` | `float` | Minimum net-of-cost Sharpe for promotion | 0.30 |

## Usage Examples

### Python

#### Synthetic research loop with cost-aware promotion

```python
from openquant.research import make_synthetic_futures_dataset, run_flywheel_iteration

# Generate deterministic synthetic multi-asset futures data
dataset = make_synthetic_futures_dataset(n_bars=192, seed=7)

# Run full pipeline + cost model + promotion checks
result = run_flywheel_iteration(dataset, config={
    "cusum_threshold": 0.001,
    "commission_bps": 1.5,
    "spread_bps": 2.0,
    "min_net_sharpe": 0.30,
})

# Cost breakdown
print(result["costs"])
# {'turnover': 12.3, 'net_sharpe': 0.42, 'estimated_total_cost': 0.018, ...}

# Promotion gate results
print(result["promotion"])
# {'passed_net_sharpe': True, 'promote_candidate': True, ...}

# Full summary DataFrame
print(result["summary"])
```

## Common Pitfalls

- Over-optimizing on synthetic data — the data generator has known dynamics; validate on real data before deployment.
- Ignoring cost estimates — gross Sharpe is misleading for high-turnover strategies.

## API Reference

### Python API

- `research.make_synthetic_futures_dataset`
- `research.run_flywheel_iteration`
- `research.ResearchDataset`

### Key Functions

- `make_synthetic_futures_dataset`
- `run_flywheel_iteration`
- `ResearchDataset`

## Implementation Notes

- make_synthetic_futures_dataset is deterministic given seed — use for regression tests and reproducible notebooks.
- run_flywheel_iteration includes turnover estimation, transaction cost modeling, and net-of-cost Sharpe.
- Promotion gates check realized Sharpe, net Sharpe, and leakage guards before flagging a strategy as deployment-ready.

## Related Modules

- [`pipeline`](/modules/pipeline/)

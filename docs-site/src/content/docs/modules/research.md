---
title: "research"
description: "Synthetic datasets, transaction-cost heuristics, research comparison grids, and promotion flags."
status: authored
last_authored: '2026-10-10'
audience:
  - quant-dev
  - platform-engineering
module: "research"
api_surface: "python-only"
python_api:
  - "research.make_synthetic_futures_dataset"
  - "research.run_flywheel_iteration"
  - "research.ResearchDataset"
  - "research.run_flywheel_grid"
  - "research.research_run_manifest"
citation:
  - "OpenQuant implementation: python/openquant/research.py."
  - "OpenQuant regression tests: python/tests/test_research_grid_and_screening.py."
sidebar:
  badge: Module
---

`research` supplies deterministic synthetic inputs and a cost-aware wrapper around
[pipeline](/modules/pipeline/). It is a Python workflow utility, not a model trainer
or evidence that a strategy is ready to trade. The formulas and gates below come
from the [implementation](https://github.com/Open-Quant/openquant/blob/main/python/openquant/research.py);
[regression tests](https://github.com/Open-Quant/openquant/blob/main/python/tests/test_research_grid_and_screening.py)
check annualization, net costs and grid ordering.

## Construct a dataset

`ResearchDataset` contains parallel per-bar lists: `timestamps`, `close`,
`model_probabilities`, `model_sides`, a row-major `asset_prices` matrix and its
`asset_names`. Its optional `periods_per_year` supplies the annualization convention.
Construct it from your own aligned inputs, or use `make_synthetic_futures_dataset`.

The generator defaults to 192 bars, seed 7 and assets `CL`, `NG`, `RB`, `GC`; it
requires at least 32 bars and two assets. It combines sinusoidal price components,
drift and seeded noise, with synthetic probability and side sequences. Bars are
one minute apart, starting at 2024-01-01 09:30, without a session calendar. Names
are illustrative: this does not model exchange sessions, contract rolls, tick sizes
or actual futures fills. It sets `periods_per_year=98280.0` using the pipeline's
252 × 390 convention.

## Run and inspect the cost accounting

```python
from openquant.research import make_synthetic_futures_dataset, run_flywheel_iteration

dataset = make_synthetic_futures_dataset(n_bars=64, seed=7)
result = run_flywheel_iteration(dataset, config={
    "commission_bps": 1.5,
    "spread_bps": 2.0,
    "min_net_sharpe": 0.30,
})
costs = result["costs"]
print("bars:", len(dataset.timestamps))
print("periods/year:", costs["periods_per_year"])
print("cost identity:", abs(costs["estimated_total_cost"]
    - costs["turnover"] * costs["cost_per_turn"]) < 1e-12)
print("net identity:", abs(costs["net_total_return"]
    - (costs["gross_total_return"] - costs["estimated_total_cost"])) < 1e-12)
print("summary rows:", result["summary"].height)
print("gate names:", sorted(result["promotion"]))
```

```text
bars: 64
periods/year: 98280.0
cost identity: True
net identity: True
summary rows: 1
gate names: ['passed_alignment_guard', 'passed_event_order_guard', 'passed_net_sharpe', 'passed_realized_sharpe', 'promote_candidate']
```

The result retains every pipeline output and adds `costs`, `promotion` and a
one-row `summary`. The example checks accounting relationships rather than claiming
that synthetic performance demonstrates an investment result.

## What the cost numbers mean

Let `s[t]` be the frame's position, `r[t]` its strategy return (including the initial
zero), and `bar_vol` the sample standard deviation of those returns. The implementation
uses:

- `turnover = sum(abs(s[t] - s[t-1]))` for bars after the first; initial entry is free.
- `cost_per_turn = (commission_bps + spread_bps) * 1e-4 + slippage_vol_mult * bar_vol * sqrt(252) * 1e-3`.
- `estimated_total_cost = turnover * cost_per_turn`.
- `gross_total_return = final_equity - 1`; `net_total_return` subtracts total estimated cost.
- Each net bar return subtracts its position-change cost. `net_sharpe` is the mean
  of these net returns divided by their sample standard deviation, times
  `sqrt(periods_per_year)`; zero deviation gives zero.

These are comparison heuristics, not a fill simulation. The slippage term retains
`sqrt(252)` even for intraday bars. The net total return subtracts aggregate costs
from gross compounded return; it is not a compounded net equity curve. Unlike
pipeline realized Sharpe, this net Sharpe formula does not subtract a risk-free rate.
The [net-Sharpe test](https://github.com/Open-Quant/openquant/blob/main/python/tests/test_research_grid_and_screening.py)
independently reconstructs the per-bar deductions.

| Configuration | Default | Effect |
| --- | --- | --- |
| `commission_bps`, `spread_bps` | 1.5, 2.0 | Linear cost per unit of turnover |
| `slippage_vol_mult` | 8.0 | Multiplier on the volatility heuristic |
| `min_realized_sharpe`, `min_net_sharpe` | 0.25, 0.30 | Inclusive promotion thresholds |
| `periods_per_year` | `None` | Config override, then dataset value, then inferred timestamp spacing |

Other pipeline settings pass through. Unknown config keys are ignored, so check
spelling rather than relying on an error for a misspelled parameter.

## Promotion and repeated trials

`promote_candidate` is the conjunction of realized-Sharpe and net-Sharpe thresholds,
strictly increasing timestamps (`passed_alignment_guard`) and sorted event indices.
These guards cannot detect look-ahead inside supplied predictions and perform no
out-of-sample test. A true value is a workflow flag, not deployment approval.

`run_flywheel_grid(dataset, configs)` repeats this computation on the same dataset.
Its leaderboard sorts passing candidates first, then descending net Sharpe and
realized Sharpe, with input index breaking ties. It returns `records`, a Polars
`leaderboard` and full `runs`. Ranking many configurations on one sample creates
selection bias; reserve independent evaluation data.

`research_run_manifest(config, dataset_meta=None)` adds a 12-character SHA-256
prefix of the sorted JSON configuration and optional supplied dataset metadata.
It does not hash price data or capture the environment automatically. See
[the notebook research workflow](/workflows/notebook-research-workflow/) for the
broader reproducibility process.

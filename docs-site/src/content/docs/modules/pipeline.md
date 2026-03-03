---
title: "pipeline"
description: "End-to-end AFML research pipeline: events → signals → portfolio → risk → backtest with leakage checks."
status: validated
last_validated: '2026-03-02'
audience:
  - quant-dev
  - platform-engineering
module: "pipeline"
api_surface: "python-only"
risk_notes:
  - "The pipeline enforces input alignment and event ordering as leakage guards."
  - "run_mid_frequency_pipeline_frames adds Polars DataFrames to the raw dict output."
  - "summarize_pipeline extracts key metrics into a single-row DataFrame for notebook display."
rust_api:
  - "run_mid_frequency_pipeline"
  - "run_mid_frequency_pipeline_frames"
  - "summarize_pipeline"
sidebar:
  badge: Module
---

## Concept Overview

The pipeline module orchestrates the full AFML research workflow in a single function call. It chains: CUSUM event detection → triple-barrier labeling → bet sizing → portfolio allocation → risk metrics → backtest statistics. Each stage passes its output to the next, and built-in leakage checks verify that inputs are aligned, events are chronologically ordered, and no forward-looking bias is present.

This is designed for rapid research iteration — change a parameter, re-run the pipeline, and compare the summary table. The `_frames` variant enriches output with Polars DataFrames for each stage, making notebook exploration ergonomic.

## When to Use

Use this when you want to run a complete AFML workflow without manually chaining individual modules. It's the fastest path from "I have prices and a model" to "I have a backtested strategy with risk metrics."

**Prerequisites**: Timestamps, close prices, model probability forecasts, and multi-asset price matrix.

**Alternatives**: Call individual modules (filters, labeling, bet_sizing, etc.) for more control over each stage.

## Key Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `cusum_threshold` | `float` | CUSUM event filter threshold | 0.001 |
| `num_classes` | `int` | Number of label classes for bet sizing | 2 |
| `step_size` | `float` | Bet size discretization step | 0.1 |
| `risk_free_rate` | `float` | Risk-free rate for Sharpe calculations | 0.0 |
| `confidence_level` | `float` | Confidence level for VaR/ES | 0.05 |

## Usage Examples

### Python

#### Run a complete research pipeline

```python
from openquant.pipeline import run_mid_frequency_pipeline_frames, summarize_pipeline

out = run_mid_frequency_pipeline_frames(
    timestamps=timestamps,
    close=close,
    model_probabilities=probabilities,
    asset_prices=asset_prices,
    model_sides=sides,
    asset_names=["CL", "NG", "RB", "GC"],
    cusum_threshold=0.001,
)

# Polars DataFrames for each stage
signals_df = out["frames"]["signals"]
backtest_df = out["frames"]["backtest"]
weights_df = out["frames"]["weights"]

# One-row summary with key metrics
summary = summarize_pipeline(out)
print(summary)
# portfolio_sharpe | realized_sharpe | value_at_risk | has_forward_look_bias
```

## Common Pitfalls

- Not checking leakage_checks in the output — the pipeline flags forward-look bias but doesn't stop execution.
- Using the raw dict output when DataFrames are more convenient — prefer run_mid_frequency_pipeline_frames.

## API Reference

### Python API

- `pipeline.run_mid_frequency_pipeline`
- `pipeline.run_mid_frequency_pipeline_frames`
- `pipeline.summarize_pipeline`

### Key Functions

- `run_mid_frequency_pipeline`
- `run_mid_frequency_pipeline_frames`
- `summarize_pipeline`

## Implementation Notes

- The pipeline enforces input alignment and event ordering as leakage guards.
- run_mid_frequency_pipeline_frames adds Polars DataFrames to the raw dict output.
- summarize_pipeline extracts key metrics into a single-row DataFrame for notebook display.

## Related Modules

- [`filters`](/modules/filters/)
- [`labeling`](/modules/labeling/)
- [`bet-sizing`](/modules/bet-sizing/)
- [`backtest-statistics`](/modules/backtest-statistics/)
- [`risk-metrics`](/modules/risk-metrics/)

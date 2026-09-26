---
title: "pipeline"
description: "End-to-end AFML research pipeline: events → signals → portfolio → risk → backtest, with ordering checks."
status: generated
generated_from: src/data/moduleDocs.ts
last_generated: '2026-09-26'
audience:
  - quant-dev
  - platform-engineering
module: "pipeline"
api_surface: "both"
rust_api:
  - "run_mid_frequency_pipeline"
  - "ResearchPipelineConfig"
  - "ResearchPipelineInput"
  - "ResearchPipelineOutput"
  - "LeakageChecks"
python_api:
  - "pipeline.run_mid_frequency_pipeline"
  - "pipeline.run_mid_frequency_pipeline_frames"
  - "pipeline.summarize_pipeline"
sidebar:
  badge: Module
---

## Concept Overview

The pipeline module orchestrates the full AFML research workflow in a single function call. It chains: CUSUM event detection → bet sizing from the model's probabilities → a max-Sharpe portfolio allocation → risk metrics → a single-asset backtest. No labeling or model fitting happens here: the model probabilities and sides are inputs, one per bar, and the signal is traded with a one-bar lag. The output also reports whether the timestamps and the event positions are in increasing order. The pipeline does not detect look-ahead in the probabilities you pass it; that is your responsibility.

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
| `risk_free_rate` | `float` | Annual risk-free rate, for both the max-Sharpe allocation and realized_sharpe | 0.0 |
| `periods_per_year` | `float` | Bars per year of close and rows per year of asset_prices; annualises realized_sharpe and the portfolio figures | 252.0 |
| `confidence_level` | `float` | Confidence level for VaR/ES | 0.05 |

## Usage Examples

### Python

#### Run a complete research pipeline

```python doc-check=skip
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
# portfolio_sharpe | realized_sharpe | value_at_risk | timestamps_increasing | ...
```

## Common Pitfalls

- Reading has_forward_look_bias as a test: it is a deprecated constant (always false). The pipeline cannot see look-ahead inside model_probabilities; fit them on data available at each bar's close.
- Unordered timestamps do not stop the run; check leakage_checks.timestamps_increasing.
- Leaving periods_per_year at 252 for intraday bars: realized_sharpe and the portfolio figures are then annual in units of 252 bars, not calendar years.
- Using the raw dict output when DataFrames are more convenient — prefer run_mid_frequency_pipeline_frames.

## Risk Notes and Caveats

- Mismatched input lengths are an error. leakage_checks reports two computed ordering checks, timestamps_increasing and event_indices_sorted; inputs_aligned (always true) and has_forward_look_bias (always false) are deprecated constants.
- run_mid_frequency_pipeline_frames and summarize_pipeline are Python-only helpers over the Rust run_mid_frequency_pipeline.
- run_mid_frequency_pipeline_frames adds Polars DataFrames to the raw dict output.
- summarize_pipeline extracts key metrics into a single-row DataFrame for notebook display.

## Related Modules

- [`filters`](/modules/filters/)
- [`labeling`](/modules/labeling/)
- [`bet-sizing`](/modules/bet-sizing/)
- [`backtest-statistics`](/modules/backtest-statistics/)
- [`risk-metrics`](/modules/risk-metrics/)

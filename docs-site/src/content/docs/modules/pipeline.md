---
title: "pipeline"
description: "CUSUM signals, a single-instrument backtest, separate portfolio allocation, and ordering checks."
status: authored
last_authored: '2026-10-10'
audience:
  - quant-dev
  - platform-engineering
module: "pipeline"
api_surface: "both"
rust_api:
  - "run_mid_frequency_pipeline"
  - "infer_periods_per_year"
  - "ResearchPipelineConfig"
  - "ResearchPipelineInput"
  - "ResearchPipelineOutput"
  - "LeakageChecks"
python_api:
  - "pipeline.run_mid_frequency_pipeline"
  - "pipeline.run_mid_frequency_pipeline_frames"
  - "pipeline.summarize_pipeline"
  - "pipeline.infer_periods_per_year"
citation:
  - "OpenQuant implementation: python/openquant/pipeline.py."
  - "OpenQuant regression tests: python/tests/test_pipeline_api.py."
sidebar:
  badge: Module
---

The pipeline takes already-computed probabilities and sides, selects CUSUM events,
discretizes their bet sizes, and holds each event signal until the next event. It
also estimates a max-Sharpe allocation from `asset_prices`. **Those portfolio weights
are not used by the backtest**: the equity curve trades only `close`, using the
previous bar's signal. There is no label construction, model fitting, or rolling
portfolio rebalance here. These are the stages implemented in the
[Rust pipeline](https://github.com/Open-Quant/openquant/blob/main/crates/openquant/src/pipeline.rs).

## Inputs and outputs

Supply oldest-first timestamp strings in `"%Y-%m-%d %H:%M:%S"` format and one close,
probability and optional side per bar. `asset_prices` is a rectangular matrix with
one row per bar and one column per asset, matching `asset_names`. If sides are
omitted, event sides default to +1. Mismatched lengths raise; unordered timestamps
are reported rather than sorted. CUSUM accumulates log returns of positive close
prices, so its threshold is in log-return units ([filter source](https://github.com/Open-Quant/openquant/blob/main/crates/openquant/src/filters.rs)).
No selected events is an error.
See the [input validation and stage construction](https://github.com/Open-Quant/openquant/blob/main/crates/openquant/src/pipeline.rs).

| Result | Meaning |
| --- | --- |
| `events` | Selected indices, timestamps, probabilities and sides |
| `signals` | Event signals and a full per-bar signal timeline |
| `portfolio` | Allocation weights and estimated annual return, risk and Sharpe |
| `risk` | Realized strategy Sharpe, VaR, ES and conditional drawdown risk |
| `backtest` | One equity value per bar; one fewer strategy return |
| `leakage_checks` | Ordering flags, plus two deprecated compatibility constants |

The Python [`_frames` wrapper](https://github.com/Open-Quant/openquant/blob/main/python/openquant/pipeline.py)
adds `signals`, `events`, `backtest` and `weights` Polars frames under `frames`.
It prepends a zero return to the backtest frame to align returns with timestamps.
The event frame uses the event timestamp for both start and end; these are selected
points, not labeled holding intervals. `summarize_pipeline` returns one row of metrics
and guards. The [Python contract tests](https://github.com/Open-Quant/openquant/blob/main/python/tests/test_pipeline_api.py)
check these shapes and the unordered-timestamp behavior.

## Run a self-contained example

This uses deterministic synthetic inputs only to demonstrate the API contract;
its forecasts are not fitted or validated market predictions.

```python
from openquant.pipeline import run_mid_frequency_pipeline_frames, summarize_pipeline
from openquant.research import make_synthetic_futures_dataset

dataset = make_synthetic_futures_dataset(n_bars=64, seed=7)
out = run_mid_frequency_pipeline_frames(
    timestamps=dataset.timestamps,
    close=dataset.close,
    model_probabilities=dataset.model_probabilities,
    model_sides=dataset.model_sides,
    asset_prices=dataset.asset_prices,
    asset_names=dataset.asset_names,
)
print("bars:", out["frames"]["signals"].height)
print("raw returns:", len(out["backtest"]["strategy_returns"]))
print("frame returns:", out["frames"]["backtest"]["returns"].len())
print("assets:", out["frames"]["weights"]["asset"].to_list())
print("periods/year:", out["risk"]["periods_per_year"])
print("summary rows:", summarize_pipeline(out).height)
print("timestamps ordered:", out["leakage_checks"]["timestamps_increasing"])
```

```text
bars: 64
raw returns: 63
frame returns: 64
assets: ['CL', 'NG', 'RB', 'GC']
periods/year: 98280.0
summary rows: 1
timestamps ordered: True
```

## Parameters and interpretation

The [Python wrapper](https://github.com/Open-Quant/openquant/blob/main/python/openquant/pipeline.py)
defaults to `cusum_threshold=0.001`, `num_classes=2`, `step_size=0.1`,
`risk_free_rate=0.0` and `confidence_level=0.05`. The last is the lower-tail
probability for VaR/ES; the pipeline passes its complement to conditional drawdown
risk. `risk_free_rate` is annual and applies to both allocation and realized Sharpe.

Python's `periods_per_year=None` derives an annualization factor from the median
positive timestamp gap: 252 sessions of 390 minutes for intraday bars, 252 for daily
bars, and 365.25 divided by gap days for longer intervals. It falls back to 252 when
no positive gap is measurable. One-minute bars therefore use 98,280. Override this
for a different trading calendar, and ensure both price inputs share that frequency.
Rust `ResearchPipelineConfig` and the `_core` binding instead default to 252.
The [annualization tests](https://github.com/Open-Quant/openquant/blob/main/python/tests/test_pipeline_api.py)
exercise the distinction.

`timestamps_increasing` and `event_indices_sorted` are computed checks.
`inputs_aligned=True` and `has_forward_look_bias=False` are deprecated constants:
the latter does **not** certify that forecasts were available at trade time. The
portfolio estimate uses the supplied full sample. Keep training and evaluation
separate and inspect provenance before interpreting its metrics.

For component-level control, use [filters](/modules/filters/),
[bet sizing](/modules/bet-sizing/), [portfolio optimization](/modules/portfolio-optimization/)
and [risk metrics](/modules/risk-metrics/). For a cost heuristic around this same
backtest, see [research](/modules/research/).

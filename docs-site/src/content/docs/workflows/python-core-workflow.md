---
title: Python Core Workflow
description: A Python API tour of CSV ingestion and a separate synthetic pipeline, with Rust/Python conventions made explicit.
status: authored
last_authored: '2026-11-07'
audience:
  - quant-dev
  - platform-engineering
sidebar:
  order: 2
---

This script connects two API demonstrations: it loads a small CSV fixture and builds
bars, then starts a separate synthetic dataset for the convenience pipeline and its
promotion flags. The CSV bars do not feed that second example, and no predictive model
is trained here. The flags illustrate configured checks, not a validated research decision.
For one connected path through features, training folds and out-of-sample predictions,
use the [end-to-end research journey](/workflows/research-journey/).

You need the bindings built first ([Python Bindings
Setup](/setup/python-bindings/)). Run everything below from the
repository root:

```bash
uv run --no-sync --python .venv/bin/python python workflow.py
```

For the namespace-by-namespace symbol list, see the [Module Reference
Index](/modules/), under "By language surface". This page explains the loop
rather than reprinting it.

## The script

```python
"""OpenQuant research loop: ingest -> bars -> pipeline -> costs -> decision."""

from openquant import bars, data, research

# --- Stage 1: ingest and clean -------------------------------------------
frame, quality = data.load_ohlcv(
    "python/tests/fixtures/ohlcv_us_equities.csv",
    return_report=True,
)
print(f"stage 1  {frame.height} rows, {quality['symbol_count']} symbols, "
      f"{quality['rows_removed_by_deduplication']} duplicate row(s) dropped")
print(f"stage 1  columns: {frame.columns}")

# --- Stage 2: event-driven bars ------------------------------------------
dollar_bars = bars.build_dollar_bars(frame, dollar_value_per_bar=1_000_000.0)
print(f"stage 2  {dollar_bars.height} dollar bars")
print(f"stage 2  diagnostics: {bars.bar_diagnostics(dollar_bars)}")

# --- Stages 3-4: the research loop ---------------------------------------
# The fixture above is six rows; the loop needs a real series, so use the
# deterministic synthetic generator for the rest.
dataset = research.make_synthetic_futures_dataset(n_bars=480, seed=7)
result = research.run_flywheel_iteration(dataset)

print(f"stage 3  {len(result['events']['timestamps'])} CUSUM events")
print(f"stage 3  leakage checks: {result['leakage_checks']}")

risk = result["risk"]
print(f"stage 4  realized_sharpe {risk['realized_sharpe']:+.4f}   "
      f"VaR(5%) {risk['value_at_risk']:+.6f}   ES(5%) {risk['expected_shortfall']:+.6f}")

portfolio = result["portfolio"]
weights = dict(zip(portfolio["asset_names"], (round(w, 3) for w in portfolio["weights"])))
print(f"stage 4  weights {weights}   portfolio_sharpe {portfolio['portfolio_sharpe']:.4f}")

costs = result["costs"]
print(f"stage 4  turnover {costs['turnover']:.2f}   cost {costs['estimated_total_cost']:.6f}")
print(f"stage 4  gross {costs['gross_total_return']:+.6f} -> "
      f"net {costs['net_total_return']:+.6f}   net_sharpe {costs['net_sharpe']:+.4f}")

print("stage 4  promotion gates:")
for gate, passed in result["promotion"].items():
    print(f"           {gate:<26} {passed}")
```

## Output

```text
stage 1  5 rows, 2 symbols, 1 duplicate row(s) dropped
stage 1  columns: ['ts', 'symbol', 'open', 'high', 'low', 'close', 'volume', 'adj_close']
stage 2  4 dollar bars
stage 2  diagnostics: {'n_bars': 4.0, 'lag1_return_autocorr': 0.0, 'lag1_sq_return_autocorr': 0.0, 'return_std': 0.0}
stage 3  207 CUSUM events
stage 3  leakage checks: {'inputs_aligned': True, 'timestamps_increasing': True, 'event_indices_sorted': True, 'has_forward_look_bias': False}
stage 4  realized_sharpe -6.4725   VaR(5%) -0.000201   ES(5%) -0.000330
stage 4  weights {'CL': 0.237, 'NG': 0.313, 'RB': 0.215, 'GC': 0.234}   portfolio_sharpe 36.1214
stage 4  turnover 9.40   cost 0.003441
stage 4  gross -0.001254 -> net -0.004695   net_sharpe -23.8478
stage 4  promotion gates:
           passed_realized_sharpe     False
           passed_net_sharpe          False
           passed_alignment_guard     True
           passed_event_order_guard   True
           promote_candidate          False
```

:::caution[Annualised figures are not performance]
These numbers come from a few hours of synthetic one-minute bars, and the
max-Sharpe allocation is fitted and scored on the same data (in-sample).
Annualising at 390 bars a day (98,280 a year) multiplies a per-bar Sharpe
ratio by √98,280 ≈ 313, so a small per-bar edge, or plain noise, becomes a
very large annual number. For an honest estimate, evaluate out of sample on
real data with purged CV or CPCV and deflate the Sharpe ratio for the
number of trials ([CPCV and the deflated Sharpe
ratio](/runbooks/cpcv-deflated-sharpe/)).
:::

Four things in that output are worth pausing on.

**480 bars produced 207 events.** CUSUM at a 0.1% threshold fires on
roughly 43% of bars here. On real data that ratio is your event-rate
knob, and it trades statistical power against label overlap.

**`portfolio_sharpe` 36.1 sits next to `realized_sharpe` -6.47.** These
answer different questions and are not comparable. The first is the
allocator's in-sample optimum across the four assets, scored on the same
sample it was fitted to and so flattering by construction; the second is
what the traded strategy returned. Confusing them is the easiest way to
believe a dead strategy is alive.

**Cost is larger than gross return.** `turnover` 9.4 × the per-turn
charge gives 0.0034, against a gross return of -0.0013. The strategy was
already losing; costs made it lose almost four times as much.

**Both ordering guards passed and the candidate was still rejected.** These
checks verify increasing timestamps and sorted event indices. They do not audit
feature availability or model training; the legacy alignment/look-ahead fields
are constants, as explained below.

## What each stage is doing, and why

### Stage 1 — ingest, and get the report

`load_ohlcv` does three things that matter before any modelling:
canonicalises the header (`Date` → `ts`, `Ticker` → `symbol`, `Adj Close`
→ `adj_close`, and about a dozen more aliases in `data.py`), sorts by
symbol and timestamp, and de-duplicates — by default keeping the *last surviving
input row* for a repeated `(symbol, ts)`. Stable sorting preserves input order
among tied keys; it does not establish which vendor revision was available at a
historical decision time.

Pass `return_report=True` whenever the data is not yours. The
deduplication is silent otherwise. Keeping a later restatement can introduce
look-ahead just as easily as discarding the wrong row. Resolve point-in-time
availability upstream, before cleaning: neither `dedupe_keep="first"` nor
`"last"` can establish it from the bar timestamp alone. Preserve revision
provenance separately because the cleaner discards extra columns.
`data.data_quality_report(df)` gives you the same diagnostics for a frame
you already hold. `data.align_calendar(df, interval="1d")` inserts missing grid
rows with null prices/volume and `is_missing_bar=True`; it does not impute them.
Its grid is anchored at each symbol's first timestamp and includes weekends and
holidays, rather than following an exchange calendar. Request its alignment
report to inspect input rows that fall off that grid.

### Stage 2 — bars sampled on activity

`bars.build_dollar_bars` (and `build_tick_bars`, `build_volume_bars`)
close a bar when a *quantity* threshold is met rather than when the clock
advances. Dollar bars are usually the best default: they are the least
sensitive to splits and to secular growth in share price, so bar
statistics stay comparable over long histories.
`bars.bar_diagnostics` reports the lag-1 autocorrelation you are trying
to remove by sampling this way.

:::caution[Memory]
`bars` crosses the PyO3 boundary with plain Python lists — one of
timestamp *strings* — so peak memory is a large multiple of the Arrow
frame you started from. Process complete symbols separately if that reduces
the data held at once; independent time slices reset aggregation state and
are not generally equivalent to one call. See
[Troubleshooting](/setup/troubleshooting/#python-runs-out-of-memory-building-bars-from-a-large-dataset).
:::

### Stage 3 — the pipeline, and its leakage checks

`run_mid_frequency_pipeline_frames` runs CUSUM sampling → bet sizing →
backtest → portfolio and risk in one call (no labeling: the model
probabilities are an input), returning nested
dicts under `events`, `signals`, `portfolio`, `risk`, `backtest`,
`leakage_checks`, plus polars frames under `frames`. The `_frames`
variant is the one to use interactively;
`run_mid_frequency_pipeline` returns the same content without the
DataFrames.

Read `leakage_checks` before you read anything else.
`timestamps_increasing` and `event_indices_sorted` are computed from the
data, and two gates in `promotion` below depend on them. `inputs_aligned`
and `has_forward_look_bias` are deprecated constants (`True` and `False`):
mismatched lengths raise instead, and the pipeline cannot detect
look-ahead in the model probabilities you pass it.

The parameters worth knowing:

| Parameter | Default | Effect |
|---|---|---|
| `cusum_threshold` | `0.001` | Event rate. Lower gives more events and more overlap. |
| `num_classes` | `2` | The null probability `get_signal` tests against, `1/num_classes`. |
| `step_size` | `0.1` | Position quantisation. Larger suppresses churn at the cost of tracking. |
| `risk_free_rate` | `0.0` | Annual rate, subtracted in both the allocation and `realized_sharpe` (as `risk_free_rate / periods_per_year` per bar). |
| `periods_per_year` | `None` | Bars a year. Annualises `realized_sharpe`, `realized_vol`, `net_sharpe` and the portfolio figures. `None` derives it from the timestamps (252 for daily bars; 390 × 252 = 98,280 for one-minute bars on a 6.5-hour session); the synthetic dataset carries 98,280. The Rust config and `openquant._core` default to 252. |
| `confidence_level` | `0.05` | Tail level for VaR, ES and CDaR. |

### Stage 4 — costs, and the promotion decision

This is what separates `research.run_flywheel_iteration` from calling the
pipeline directly: it charges the strategy for trading.

`costs` is built from turnover times a per-turn cost of commission +
spread + a volatility-scaled slippage term (`commission_bps` 1.5,
`spread_bps` 2.0, `slippage_vol_mult` 8.0 by default). `net_sharpe` is
recomputed after that charge. Gross and net will differ, sometimes by
enough to change the sign of the decision — which is the point.

These are comparison heuristics, not a full execution model. The slippage term
uses per-bar strategy-return volatility times a fixed `sqrt(252)`, even when
`periods_per_year` differs. `gross_total_return` is final gross equity minus one;
`net_total_return` subtracts the summed cost estimate from it rather than
recompounding the cost-adjusted returns. `net_sharpe` does use those per-bar
cost-adjusted returns, but does **not** subtract `risk_free_rate`, unlike the
pipeline's `realized_sharpe`. Keep those definitions explicit when comparing
the fields or choosing thresholds.

`promotion` is then four boolean gates, all of which must hold for
`promote_candidate`:

| Gate | Passes when |
|---|---|
| `passed_realized_sharpe` | realised Sharpe ≥ `min_realized_sharpe` (0.25) |
| `passed_net_sharpe` | net-of-cost Sharpe ≥ `min_net_sharpe` (0.30) |
| `passed_alignment_guard` | `leakage_checks["timestamps_increasing"]` |
| `passed_event_order_guard` | `leakage_checks["event_indices_sorted"]` |

Two of the four are ordering checks, not performance thresholds or a model-training
audit. The flag cannot establish that a candidate is ready for research promotion.

### Feature diagnostics, before you trust any of it

`openquant.feature_diagnostics` is the part most easily misused, so it is
worth naming what each function answers:

- `mdi_importance` — in-sample, computed from the fitted model. Cheap,
  and biased toward high-cardinality features.
- `mda_importance` — out-of-sample, by permutation. Takes
  `event_end_indices` (plus `n_splits`, `pct_embargo`) and builds *purged*
  splits from them. This is the one to trust when the two disagree.
  `event_end_indices` is **required**: omitting it raises, because a
  missing horizon used to yield degenerate one-row intervals that purged
  nothing. To accept embargo-only splits, opt out explicitly with
  `allow_unpurged=True` — the result then reports
  `cv["method"] == "kfold_embargo_only"` and `cv["purged"] is False`.
- `sfi_importance` — one feature at a time, so it is the only one immune
  to the substitution effect. Also purged; also slow, since it refits per
  feature.
- `substitution_effect_report` — tells you when correlated features are
  splitting credit and making all three of the above look flat.
- `orthogonalize_features_pca` — the usual response to that report.

MDI and MDA routinely rank features differently. That is information
about your features, not a bug.

## Carrying a workflow between Python and Rust

Compare the contracts at the layer you call, not just similarly named functions.
The [Rust walkthrough](/workflows/rust-core-workflow/) composes individual functions;
this page uses wrappers that add frames, defaults and cost estimates.

| Convention | What to preserve when moving between languages |
| --- | --- |
| Annualisation and risk-free rate | The pipeline takes an **annual** risk-free rate and divides by `periods_per_year` for realised Sharpe. `backtest_stats.sharpe_ratio` and Rust `backtest_statistics::sharpe_ratio` instead take a **per-return-period** rate. Python's public pipeline infers annualisation when omitted; the Rust config and compiled `_core` default to 252. Pass it explicitly for a comparison. |
| Matrix orientation | Pipeline `asset_prices` is observations × assets in both languages: Python nested rows or Rust `DMatrix`, with names in column order. This does not generalise to every API: sampling indicators are bars × labels, while ensemble aggregation expects models × observations. Check the module's shape contract. |
| Time and row alignment | Pipeline timestamps are strings in Python and `NaiveDateTime` in Rust. The Python CV wrapper converts datetimes to int64 nanoseconds; integer bar indices pass through unchanged. Keep both span columns in one unit, sort events by start and apply the same ordering to features, targets and weights. The splitter does not validate that ordering. |
| Label overlap | Purged CV treats label spans as closed intervals: touching endpoints overlap. Returned indices identify rows in the supplied order. This differs from the half-open `[start, end)` lifetime used by active-bet averaging. |
| Return stream and output shape | A triple-barrier label return is an event outcome, possibly overlapping other events. The pipeline's gross strategy returns are a per-bar account path using the previous bar's position, with `n-1` returns for `n` closes. The Python backtest frame prepends a zero for timestamp alignment. Neither event returns nor that padding should silently change the sample used for a Sharpe comparison. |

Use public Python wrappers when their contracts fit the task. Calling `_core` directly can
bypass conversions or default inference; matching a Rust function name is not enough to
establish parity. See [cross-validation](/modules/cross-validation/) for split shapes and
[backtest statistics](/modules/backtest-statistics/) for score definitions.

This example checks the pipeline's annual-rate convention against the lower-level Sharpe
function shared by the Python binding and Rust core:

```python
import math
from openquant import backtest_stats, pipeline, research

ds = research.make_synthetic_futures_dataset(n_bars=80, seed=7)
periods, annual_rf = 252.0 * 390.0, 0.04
out = pipeline.run_mid_frequency_pipeline_frames(
    ds.timestamps, ds.close, ds.model_probabilities, ds.asset_prices,
    model_sides=ds.model_sides, asset_names=ds.asset_names,
    periods_per_year=periods, risk_free_rate=annual_rf,
)
returns = out["backtest"]["strategy_returns"]  # no frame-alignment zero
per_period = backtest_stats.sharpe_ratio(returns, 1.0, annual_rf / periods)
annualised = backtest_stats.sharpe_ratio(returns, periods, annual_rf / periods)
assert math.isclose(annualised, per_period * math.sqrt(periods), rel_tol=1e-12)
assert math.isclose(annualised, out["risk"]["realized_sharpe"], rel_tol=1e-12)
assert len(returns) == len(ds.close) - 1
assert out["frames"]["backtest"].height == len(ds.close)
# Passing the annual rate directly to the low-level function changes the calculation.
assert not math.isclose(annualised, backtest_stats.sharpe_ratio(returns, periods, annual_rf))
```

## Where to go next

- [Rust Core Workflow](/workflows/rust-core-workflow/) — the same ground in Rust
- [Examples Catalog](/examples/catalog/) — runnable examples that ship in the repo
- [Module Reference Index](/modules/) — the full symbol lists, by namespace

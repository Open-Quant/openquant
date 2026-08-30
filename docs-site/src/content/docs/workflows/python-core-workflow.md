---
title: Python Core Workflow
description: One runnable Python script from raw OHLCV to a promotion decision, with its output.
status: reviewed
last_validated: '2026-08-30'
banner:
  content: '<span class="doc-status doc-status--reviewed">Reviewed</span> A human has read this page end to end. It has not been verified line by line against the code.'
audience:
  - quant-dev
  - platform-engineering
sidebar:
  order: 2
---

The Python surface is a research loop, not a wrapper: it goes from a CSV
to a *decision* — promote this candidate, or don't. This page is one
script that runs the whole loop, and its actual output.

You need the bindings built first ([Python Bindings
Setup](/setup/python-bindings/)). Run everything below from the
repository root:

```bash
uv run --python .venv/bin/python python workflow.py
```

For the namespace-by-namespace symbol list, see [API
Surfaces](/module-reference/api-surfaces/). This page explains the loop
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
stage 3  leakage checks: {'inputs_aligned': True, 'event_indices_sorted': True, 'has_forward_look_bias': False}
stage 4  realized_sharpe -0.3277   VaR(5%) -0.000201   ES(5%) -0.000330
stage 4  weights {'CL': 0.403, 'NG': 0.471, 'RB': 0.043, 'GC': 0.082}   portfolio_sharpe 2.0040
stage 4  turnover 9.40   cost 0.003441
stage 4  gross -0.001254 -> net -0.004695   net_sharpe -0.2951
stage 4  promotion gates:
           passed_realized_sharpe     False
           passed_net_sharpe          False
           passed_alignment_guard     True
           passed_event_order_guard   True
           promote_candidate          False
```

Four things in that output are worth pausing on.

**480 bars produced 207 events.** CUSUM at a 0.1% threshold fires on
roughly 43% of bars here. On real data that ratio is your event-rate
knob, and it trades statistical power against label overlap.

**`portfolio_sharpe` 2.00 sits next to `realized_sharpe` -0.33.** These
answer different questions and are not comparable. The first is the
allocator's in-sample optimum across the four assets; the second is what
the traded strategy returned. Confusing them is the easiest way to
believe a dead strategy is alive.

**Cost is larger than gross return.** `turnover` 9.4 × the per-turn
charge gives 0.0034, against a gross return of -0.0013. The strategy was
already losing; costs made it lose almost four times as much.

**Both leakage guards passed and the candidate was still rejected.** The
guards are necessary, not sufficient — they say the inputs were coherent,
not that the result is good.

## What each stage is doing, and why

### Stage 1 — ingest, and get the report

`load_ohlcv` does three things that matter before any modelling:
canonicalises the header (`Date` → `ts`, `Ticker` → `symbol`, `Adj Close`
→ `adj_close`, and about a dozen more aliases in `data.py`), sorts by
symbol and timestamp, and de-duplicates — keeping the *last* row for a
repeated `(symbol, ts)`, which is the right convention when a vendor
restates a bar.

Pass `return_report=True` whenever the data is not yours. The
deduplication is silent otherwise, and a silently dropped restatement is
how a look-ahead bug enters a backtest without anyone noticing.
`data.data_quality_report(df)` gives you the same diagnostics for a frame
you already hold, and `data.align_calendar(df, interval="1d")` fills the
grid when you need one bar per period per symbol.

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
frame you started from. On multi-million-row inputs, slice first. See
[Troubleshooting](/setup/troubleshooting/#python-runs-out-of-memory-building-bars-from-a-large-dataset).
:::

### Stage 3 — the pipeline, and its leakage checks

`run_mid_frequency_pipeline_frames` runs CUSUM sampling → labeling →
bet sizing → backtest → portfolio and risk in one call, returning nested
dicts under `events`, `signals`, `portfolio`, `risk`, `backtest`,
`leakage_checks`, plus polars frames under `frames`. The `_frames`
variant is the one to use interactively;
`run_mid_frequency_pipeline` returns the same content without the
DataFrames.

Read `leakage_checks` before you read anything else. `inputs_aligned`
and `event_indices_sorted` are the pipeline stating that it was handed
coherent inputs. A Sharpe computed on top of a failed alignment check is
not a number, and every gate in `promotion` below depends on these two
passing.

The parameters worth knowing:

| Parameter | Default | Effect |
|---|---|---|
| `cusum_threshold` | `0.001` | Event rate. Lower gives more events and more overlap. |
| `num_classes` | `2` | The null probability `get_signal` tests against, `1/num_classes`. |
| `step_size` | `0.1` | Position quantisation. Larger suppresses churn at the cost of tracking. |
| `risk_free_rate` | `0.0` | Sharpe numerator offset. |
| `confidence_level` | `0.05` | Tail level for VaR, ES and CDaR. |

### Stage 4 — costs, and the promotion decision

This is what separates `research.run_flywheel_iteration` from calling the
pipeline directly: it charges the strategy for trading.

`costs` is built from turnover times a per-turn cost of commission +
spread + a volatility-scaled slippage term (`commission_bps` 1.5,
`spread_bps` 2.0, `slippage_vol_mult` 8.0 by default). `net_sharpe` is
recomputed after that charge. Gross and net will differ, sometimes by
enough to change the sign of the decision — which is the point.

`promotion` is then four boolean gates, all of which must hold for
`promote_candidate`:

| Gate | Passes when |
|---|---|
| `passed_realized_sharpe` | realised Sharpe ≥ `min_realized_sharpe` (0.25) |
| `passed_net_sharpe` | net-of-cost Sharpe ≥ `min_net_sharpe` (0.30) |
| `passed_alignment_guard` | `leakage_checks["inputs_aligned"]` |
| `passed_event_order_guard` | `leakage_checks["event_indices_sorted"]` |

Two of the four are leakage guards, not performance thresholds. A
candidate cannot be promoted on returns alone.

### Feature diagnostics, before you trust any of it

`openquant.feature_diagnostics` is the part most easily misused, so it is
worth naming what each function answers:

- `mdi_importance` — in-sample, computed from the fitted model. Cheap,
  and biased toward high-cardinality features.
- `mda_importance` — out-of-sample, by permutation. Takes
  `event_end_indices` (plus `n_splits`, `pct_embargo`) and builds *purged*
  splits from them. This is the one to trust when the two disagree — but
  only if you actually pass `event_end_indices`; the default builds
  degenerate one-row intervals and purges nothing.
- `sfi_importance` — one feature at a time, so it is the only one immune
  to the substitution effect. Also purged; also slow, since it refits per
  feature.
- `substitution_effect_report` — tells you when correlated features are
  splitting credit and making all three of the above look flat.
- `orthogonalize_features_pca` — the usual response to that report.

MDI and MDA routinely rank features differently. That is information
about your features, not a bug.

## Where to go next

- [Rust Core Workflow](/workflows/rust-core-workflow/) — the same ground in Rust
- [Examples Catalog](/examples/catalog/) — runnable examples that ship in the repo
- [API Surfaces](/module-reference/api-surfaces/) — the full symbol lists

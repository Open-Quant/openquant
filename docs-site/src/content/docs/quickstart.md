---
title: Quickstart
description: Build OpenQuant from source, run a synthetic pipeline and interpret its output.
status: authored
last_authored: '2026-11-13'
audience:
  - quant-dev
  - platform-engineering
sidebar:
  order: 1
---

This is the human learning path. For repository automation, use the
[agent guide](/agents/) alongside it.

Build once, check one small component, then inspect a synthetic pipeline. Allow time for a native Rust
build; its duration depends on your machine and whether dependencies are cached.

You need a Rust toolchain, a linker, and `uv`. If you do not have them,
start at [Prerequisites](/setup/prerequisites/) and return here after installing the
tools for your platform.

## Step 1 — Clone

```bash
git clone https://github.com/Open-Quant/openquant.git
cd openquant
```

## Step 2 — Install

This guide builds the Python package and its PyO3 extension from the same
checkout. The distribution name is `pyopenquant`; the import name is `openquant`.
Do not `pip install openquant`: that name belongs to an unrelated project.
See [release status](/project/governance/#versioning-and-releases) if you want
to use a packaged release instead.

```bash
uv venv --python 3.13 .venv
uv sync --locked --group dev --no-install-project
uv run --no-sync --python .venv/bin/python maturin develop --release --manifest-path crates/pyopenquant/Cargo.toml

uv run --no-sync --python .venv/bin/python python -c "import openquant; print('openquant bindings import ok')"
```

This compiles the Rust extension, including `polars`, `nalgebra` and their
dependency trees. Later builds can reuse cached work. If you already have a source
checkout, use its environment and rebuild after changing Rust or binding code; restart
any running notebook kernel before importing the rebuilt extension.

The import check should print:

```
openquant bindings import ok
```

If it does not, [Troubleshooting](/setup/troubleshooting/) has the
import, toolchain and architecture checks to run next.

## Step 3 — Check one component

Start with a call small enough to verify by hand. Save this as `first_event.py`
in the repository root:

```python
import math
from openquant import filters

# Five synthetic prices, with log-return steps +0.006, +0.006, -0.006, -0.006.
close = [100.0 * math.exp(x) for x in [0.0, 0.006, 0.012, 0.006, 0.0]]
events = filters.cusum_filter_indices(close, 0.01)
assert events == [2, 4]
print("event row positions:", events)
```

```text
event row positions: [2, 4]
```

```bash
uv run --no-sync --python .venv/bin/python python first_event.py
```

The filter accumulates log-price changes, not price levels. Two upward steps sum
to 0.012 and exceed the 0.01 threshold at row 2; two downward steps exceed it in
the other direction at row 4. Positions are **zero-based**, so these refer to the
third and fifth input prices. Pass positive prices, not precomputed returns.

This verifies a real native call and a small expected result. It does not fit a
model, generate buy/sell instructions or validate a dataset. The [filter guide](/modules/filters/)
explains reset rules and changing thresholds. If you only need one library component,
continue with [Find an API by task](/module-reference/api-task-guide/).

## Step 4 — Inspect the pipeline

Save this as `first_run.py` in the repository root:

```python
from openquant.research import make_synthetic_futures_dataset, run_flywheel_iteration

# Deterministic synthetic multi-asset futures data: 192 one-minute bars,
# four instruments. No market data feed required.
dataset = make_synthetic_futures_dataset(n_bars=192, seed=7)

# Exercise events -> sizing -> backtest -> portfolio -> risk,
# then estimated trading costs and promotion gates. No model is fitted.
result = run_flywheel_iteration(dataset)

print("leakage checks:")
for k, v in result["leakage_checks"].items():
    print(f"  {k:<24} {v}")

print("\nsummary:")
summary = result["summary"].transpose(include_header=True, header_name="metric", column_names=["value"])
for metric, value in summary.iter_rows():
    print(f"  {metric:<26} {value:>12.6f}")

print("\npromotion gates:")
for k, v in result["promotion"].items():
    print(f"  {k:<26} {v}")
```

Run it:

```bash
uv run --no-sync --python .venv/bin/python python first_run.py
```

Here is what it printed on the machine this page was written on. The
dataset is seeded; the example checker compares the printed output. Floating-point
rounding can vary across builds, so use the stages and fields to diagnose a mismatch
before interpreting a small numerical difference as a model change:

```text
leakage checks:
  inputs_aligned           True
  timestamps_increasing    True
  event_indices_sorted     True
  has_forward_look_bias    False

summary:
  portfolio_sharpe              25.303744
  portfolio_return               6.956303
  portfolio_risk                 0.274912
  realized_sharpe               -2.701140
  value_at_risk                 -0.000209
  expected_shortfall            -0.000312
  conditional_drawdown_risk      0.002713
  inputs_aligned                 1.000000
  timestamps_increasing          1.000000
  event_indices_sorted           1.000000
  has_forward_look_bias          0.000000
  turnover                       3.600000
  realized_vol                   0.037197
  estimated_cost                 0.001314
  gross_total_return            -0.000197
  net_total_return              -0.001511
  net_sharpe                   -20.589471

promotion gates:
  passed_realized_sharpe     False
  passed_net_sharpe          False
  passed_alignment_guard     True
  passed_event_order_guard   True
  promote_candidate          False
```

**The candidate was rejected, and that is the correct result.** The
synthetic dataset's "model probabilities" are a sine wave, not a model.
A quickstart that printed a promoted strategy would be teaching you to
trust the wrong thing.

## Read the output in this order

| Look at… | What it answers | What it does not establish |
| --- | --- | --- |
| Ordering checks | Are timestamps and event indices ordered? | Whether a feature or supplied probability contains future information |
| `realized_sharpe`, gross and net return | What did this synthetic strategy calculation produce under its stated conventions? | Out-of-sample performance or executable order fills |
| `portfolio_sharpe` | How did the allocator score its in-sample synthetic asset mix? | The traded strategy's Sharpe ratio |
| `promote_candidate` | Did the configured flags pass? | Permission to trade or production readiness |

A successful first run means the script executes and you can explain these
outputs. A `False` promotion decision is not an installation failure. Do not tune
thresholds just to turn that flag green; that would change the question after
seeing the answer.

## What you just ran

`run_flywheel_iteration` connected these components: CUSUM event
sampling, probability-to-position sizing, a backtest, portfolio
allocation and risk metrics — then charged the result
for commission, spread and slippage, and applied promotion gates.

Three things in that output are worth understanding now, because they are
the habits the rest of the library is built around.

**The ordering checks come first.** `timestamps_increasing` and
`event_indices_sorted` are computed from the data; two of the four
`promotion` gates are these checks, not performance thresholds, so a
candidate cannot be promoted on returns alone. `inputs_aligned` and
`has_forward_look_bias` are deprecated constants: mismatched input lengths
raise an error instead, and the pipeline cannot see look-ahead inside the
model probabilities you hand it. That part is your job.

**Gross and net are different numbers.** `gross_total_return` is
-0.000197; `net_total_return` is -0.001511. The gap is `estimated_cost`,
which is `turnover` (3.6) times a per-turn charge built from commission,
spread and a volatility-scaled slippage term. `net_sharpe` is recomputed
after per-bar position-change charges. These are simplified cost estimates,
not simulated order fills: initial entry is not charged, `net_total_return`
subtracts total cost from gross compounded return, and `net_sharpe` uses the
per-bar net-return series. They are not two views of one compounded net equity
curve. The net Sharpe calculation does not subtract a risk-free rate; the gross
realized Sharpe does when you configure one.

**`portfolio_sharpe` is not your strategy's Sharpe.** It reads 25.3 above
while `realized_sharpe` is -2.70. They answer different questions:
`portfolio_sharpe` is the in-sample optimum of the allocator across the
four synthetic assets, and `realized_sharpe` is what the traded strategy
actually returned. Reading the first as the second is the single easiest
way to convince yourself you have found something.

**Every Sharpe ratio and volatility here is annualised for one-minute
bars.** The synthetic bars are one minute apart, so the loop annualises
with `dataset.periods_per_year` = 390 bars a session × 252 sessions =
98,280 bars a year. On your own data the factor is derived from the
timestamps; pass `config={"periods_per_year": ...}` to override it.

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

## Next

| If you want to… | Go here |
|---|---|
| Fit a baseline and trace out-of-sample predictions | [End-to-end research journey](/workflows/research-journey/) |
| Select parameters before one chronological holdout | [Model selection and holdout](/workflows/model-selection-holdout/) |
| Align asset histories and calculate named weights | [Multi-asset allocation](/workflows/multi-asset-allocation/) |
| Understand each stage of that loop | [Python Core Workflow](/workflows/python-core-workflow/) |
| Follow a separate Rust component walkthrough | [Rust Core Workflow](/workflows/rust-core-workflow/) |
| Check which data revisions were available at a cutoff | [Point-in-time data workflow](/workflows/point-in-time-data/) |
| Load and inspect your own CSV or Parquet | [Data loading and provenance](/modules/data/), then [Examples Catalog](/examples/catalog/) |
| Reproduce a research notebook and inspect its evidence | [Research runbooks](/runbooks/) and [Notebook research workflow](/workflows/notebook-research-workflow/) |
| Choose a Python wrapper or Rust API for a task | [Find an API by task](/module-reference/api-task-guide/) |
| Check a function's arguments and returned fields | [Python API reference](/api/python/) |
| Build and test the repo itself | [Local Build Setup](/setup/local-build/) |

For your own data, loading the file is only ingestion. Define features, labels and
training boundaries before generating model probabilities; the [research journey](/workflows/research-journey/)
shows that handoff. The convenience pipeline does not train a predictive model for you.

The repo's own test suite and documentation gates are *maintainer*
commands, not first-run commands. They live on [Local Build
Setup](/setup/local-build/); passing them verifies their covered checks, not the validity of your research design.

---
title: Quickstart
description: Install OpenQuant and get one real result out of it.
status: reviewed
last_validated: '2026-08-30'
audience:
  - quant-dev
  - platform-engineering
sidebar:
  order: 1
---

Three steps, ending in a number OpenQuant computed for you. Budget
**20–30 minutes**, nearly all of it the Rust compile in step 2.

You need a Rust toolchain, a linker, and `uv`. If you do not have them,
start at [Prerequisites](/setup/prerequisites/) and come back — it is
four commands.

## Step 1 — Clone

```bash
git clone https://github.com/Open-Quant/openquant.git
cd openquant
```

## Step 2 — Install

`openquant` is not on PyPI: the Python package sits on a compiled PyO3
extension, so installing it means building it.

```bash
uv venv --python 3.11 .venv

uv run --python .venv/bin/python --with maturin \
  maturin develop --manifest-path crates/pyopenquant/Cargo.toml

uv run --python .venv/bin/python python -c "import openquant; print('openquant bindings import ok')"
```

This compiles the whole Rust workspace — `polars`, `nalgebra` and their
dependency trees — so **expect 10–20 minutes on a cold cache**. Later
builds are incremental and take seconds.

The third command should print:

```
openquant bindings import ok
```

If it does not, [Troubleshooting](/setup/troubleshooting/) has the
failures worth knowing about, and the architecture mismatch one is the
most common on macOS.

## Step 3 — Get a result

Save this as `first_run.py` in the repository root:

```python
from openquant.research import make_synthetic_futures_dataset, run_flywheel_iteration

# Deterministic synthetic multi-asset futures data: 192 one-minute bars,
# four instruments. No market data feed required.
dataset = make_synthetic_futures_dataset(n_bars=192, seed=7)

# One full AFML iteration: events -> labels -> sizing -> backtest ->
# portfolio -> risk, then trading costs and promotion gates.
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
uv run --python .venv/bin/python python first_run.py
```

Here is what it printed on the machine this page was written on. The
dataset is seeded, so you should see the same numbers:

```text
leakage checks:
  inputs_aligned           True
  event_indices_sorted     True
  has_forward_look_bias    False

summary:
  portfolio_sharpe               8.319781
  portfolio_return               0.020480
  portfolio_risk                 0.002462
  realized_sharpe               -0.136778
  value_at_risk                 -0.000209
  expected_shortfall            -0.000312
  conditional_drawdown_risk      0.000768
  inputs_aligned                 1.000000
  event_indices_sorted           1.000000
  has_forward_look_bias          0.000000
  turnover                       3.600000
  realized_vol                   0.001884
  estimated_cost                 0.001314
  gross_total_return            -0.000197
  net_total_return              -0.001511
  net_sharpe                    -0.194432

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

## What you just ran

`run_flywheel_iteration` executed the full AFML loop: CUSUM event
sampling, triple-barrier labeling, probability-to-position sizing, a
backtest, portfolio allocation and risk metrics — then charged the result
for commission, spread and slippage, and applied promotion gates.

Three things in that output are worth understanding now, because they are
the habits the rest of the library is built around.

**The leakage checks come first.** `inputs_aligned` and
`event_indices_sorted` are the pipeline asserting it was handed coherent
inputs. Two of the four `promotion` gates are these checks, not
performance thresholds. A candidate cannot be promoted on returns alone.

**Gross and net are different numbers.** `gross_total_return` is
-0.000197; `net_total_return` is -0.001511. The gap is `estimated_cost`,
which is `turnover` (3.6) times a per-turn charge built from commission,
spread and a volatility-scaled slippage term. `net_sharpe` is recomputed
after that charge. For a mid-frequency strategy this gap is usually the
whole story.

**`portfolio_sharpe` is not your strategy's Sharpe.** It reads 8.32 above
while `realized_sharpe` is -0.14. They answer different questions:
`portfolio_sharpe` is the in-sample optimum of the allocator across the
four synthetic assets, and `realized_sharpe` is what the traded strategy
actually returned. Reading the first as the second is the single easiest
way to convince yourself you have found something.

## Next

| If you want to… | Go here |
|---|---|
| Understand each stage of that loop | [Python Core Workflow](/workflows/python-core-workflow/) |
| Do the same thing in Rust | [Rust Core Workflow](/workflows/rust-core-workflow/) |
| Run it on your own CSV | [Examples Catalog](/examples/catalog/) |
| Build and test the repo itself | [Local Build Setup](/setup/local-build/) |

The repo's own test suite and documentation gates are *maintainer*
commands, not first-run commands. They live on [Local Build
Setup](/setup/local-build/); running them proves the repository is
healthy, which is a different question from whether OpenQuant is useful
to you.

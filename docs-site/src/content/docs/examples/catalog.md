---
title: Examples Catalog
description: Runnable examples that ship in this repository, and what each one demonstrates.
status: draft
audience:
  - quant-dev
  - platform-engineering
sidebar:
  order: 1
---

Every example below is a file in this repository that you can execute. The
point of this page is that you should not have to trust a snippet: each
entry names the file, the command that runs it, and what it demonstrates.

All of them need the Python bindings built first — see
[Python Bindings Setup](/setup/python-bindings/). The Rust example needs
only `cargo`.

## Examples that ship in the repo

| File | Run it with | Demonstrates |
|---|---|---|
| `crates/openquant/examples/research_notebook_smoke.rs` | `cargo run -p openquant --example research_notebook_smoke` | The minimum Rust path: CUSUM event sampling → max-Sharpe allocation → VaR/ES/CDaR. ~30 lines, asserts its own invariants. |
| `notebooks/python/scripts/smoke_all.py` | `just notebook-smoke` | Executes all eight research notebooks headlessly. The broadest single check that the Python surface works. |
| `experiments/run_pipeline.py` | `just exp-run` | A config-driven pipeline run (`experiments/configs/futures_oil_baseline.toml`) that writes artifacts to `experiments/artifacts`. |
| `python/benchmarks/benchmark_pipeline.py` | `just py-bench` | Times the mid-frequency pipeline over 30 iterations at 2048 bars. |
| `python/benchmarks/benchmark_data_processing.py` | `just py-bench-data` | Ingestion and bar-building throughput at 200k rows × 4 symbols. Use this if you want a real memory profile before feeding it production data. |
| `python/tests/` | `just py-test` | Nine test modules that double as usage examples — `test_pipeline_api.py` and `test_bindings_contract.py` are the two worth reading first. |

The eight notebooks under `notebooks/python/` are numbered in reading
order, from `01_event_labeling_and_pipeline.ipynb` to
`08_algo_wheel_experiments.ipynb`, and
`06_afml_real_data_end_to_end.ipynb` is the one that runs on real data
rather than synthetic.

## Worked example: cleaning a messy OHLCV file

The repository ships a deliberately awful CSV at
`python/tests/fixtures/ohlcv_us_equities.csv` — non-canonical column
names, two symbols interleaved, rows out of chronological order, and a
duplicated `2024-01-02` bar for AAPL:

```
Date,Ticker,Open,High,Low,Close,Volume,Adj Close
2024-01-03,AAPL,186.10,187.00,185.50,186.30,5000,186.10
2024-01-01,AAPL,184.00,185.00,183.50,184.80,8000,184.70
2024-01-02,AAPL,185.00,186.20,184.70,185.90,7000,185.80
2024-01-02,AAPL,185.05,186.30,184.60,186.00,7100,185.90
2024-01-01,MSFT,370.10,372.00,369.90,371.50,6000,371.30
2024-01-03,MSFT,372.20,373.10,371.40,372.00,5500,371.90
```

`load_ohlcv` canonicalises the header (`Date` → `ts`, `Ticker` →
`symbol`, `Adj Close` → `adj_close`), sorts by symbol and timestamp, and
drops the duplicate — keeping the *last* occurrence by default, which is
the convention you want when a vendor restates a bar:

```python
from openquant import data

frame, report = data.load_ohlcv(
    "python/tests/fixtures/ohlcv_us_equities.csv",
    return_report=True,
)
print(frame)
print(report)
```

Ask for the report whenever the data is not yours. Silent deduplication
is how a survivorship or restatement bug gets into a backtest without
anyone noticing.

## Worked example: the whole research loop

The end-to-end loops are long enough to deserve their own pages, with the
stages explained rather than just listed:

- [Rust Core Workflow](/workflows/rust-core-workflow/) — one program: event
  sampling → triple-barrier labels → bet sizing → purged CV → risk and
  allocation, with its printed output.
- [Python Core Workflow](/workflows/python-core-workflow/) — the same
  ground in Python, ending in a promotion decision.
- [Notebook Research Workflow](/workflows/notebook-research-workflow/) —
  the notebook-driven version.

## Where to start

| If you want to… | Go here |
|---|---|
| See OpenQuant produce a number, once | [Quickstart](/quickstart/) |
| Understand the Rust API | [Rust Core Workflow](/workflows/rust-core-workflow/) |
| Understand the Python API | [Python Core Workflow](/workflows/python-core-workflow/) |
| Look up one module | [Modules by AFML chapter](/module-reference/by-afml-chapter/) |
| Build and test the repo | [Local Build Setup](/setup/local-build/) |

:::note[What is still missing]
This page indexes examples and works two of them. It does not yet have a
worked example for feature diagnostics (MDI/MDA/SFI and the substitution
effect), which is the part of the library most likely to be misused, nor
one for `backtesting_engine`'s CPCV path. That is why this page is still
`draft`.
:::

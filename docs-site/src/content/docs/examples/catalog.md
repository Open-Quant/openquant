---
title: Examples Catalog
description: Runnable examples that ship in this repository, and what each one demonstrates.
status: authored
last_authored: '2026-11-07'
audience:
  - quant-dev
  - platform-engineering
sidebar:
  order: 1
---

Every example below is a file in this repository that you can execute. The
point of this page is that you should not have to trust a snippet: each
entry names the file, the command that runs it, and what it demonstrates.

The point-in-time data example uses only the Python standard library. Other Python
commands need the bindings built in the selected environment — see
[Python Bindings Setup](/setup/python-bindings/). Rust examples use the pinned toolchain
and native build prerequisites described in [local setup](/setup/local-build/). Run
repository-relative commands from the repository root.

## Choose the output you need

- **A connected fitted example:** the [research journey](/workflows/research-journey/)
  keeps one synthetic history from observations through labels, fold-local training and
  held-out predictions, then saves a JSON report. It does not produce account P&L.
- **A configured pipeline artifact:** `experiments/run_pipeline.py` consumes a configuration
  and generates synthetic data and probability sequences. It demonstrates signal, allocation and cost
  APIs; it does not train a forecasting model or certify a candidate for trading.
- **Saved tables and figures:** use the notebook runner. API tours illustrate calls;
  research runbooks record a hypothesis and evaluation protocol.
- **A component regression or timing measurement:** use tests or benchmarks. A successful
  test or fast benchmark is different evidence from an out-of-sample research result.

Two further self-contained walkthroughs connect APIs around a specific decision:
[model selection and a chronological holdout](/workflows/model-selection-holdout/) fits
preprocessing inside each training fold, selects parameters and refits before one final
classification score; [multi-asset allocation](/workflows/multi-asset-allocation/) aligns
observations and asset order before estimating moments and naming weights. Neither
reports a realized trading backtest.

## Examples that ship in the repo

| File | Run it with | Demonstrates |
|---|---|---|
| `examples/point_in_time_data.py` | `python3 examples/point_in_time_data.py` | [Point-in-time data workflow](/workflows/point-in-time-data/): offline synthetic revision selection at a cutoff, compared with latest revisions; selected-content and policy identity without claiming provider authenticity. Uses only the standard library. |
| `examples/research_journey.py` | `uv run --no-sync --python .venv/bin/python python examples/research_journey.py --out /tmp/openquant-journey.json` | One synthetic history through bars, resolved events, fold-local weights and baseline predictions; JSON with provenance and exact splits. Choose a new output path because existing reports are not overwritten. |
| `crates/openquant/examples/research_notebook_smoke.rs` | `cargo run -p openquant --example research_notebook_smoke` | Rust component smoke checks for CUSUM, allocation and VaR/ES/CDaR on separate small synthetic arrays; asserts invariants, without training a model. |
| `notebooks/python/scripts/run_notebooks.py` | `just notebooks-run` | Executes every research notebook in a Jupyter kernel and saves their tables and figures; fails on any cell error. CI runs it and checks the committed outputs are current. |
| `notebooks/python/scripts/smoke_all.py` | `just notebook-smoke` | Plain-script mirror of the core notebook calls; a quick check that the Python surface works. |
| `experiments/run_pipeline.py` | `just exp-run` | A config-driven pipeline run (`experiments/configs/futures_oil_baseline.toml`) that writes artifacts to `experiments/artifacts`. |
| `python/benchmarks/benchmark_pipeline.py` | `just py-bench` | Times 30 flywheel calls at 2048 synthetic bars after one warm-up; dataset generation is outside the timer, while pipeline execution, cost estimates and summary creation are inside. Seeds are `42 + i`; output is timing text, not a saved research report. |
| `python/benchmarks/benchmark_data_processing.py` | `just py-bench-data` | Time and throughput of `openquant.data` loading (CSV and Parquet), cleaning, calendar alignment and quality reporting at 200k rows × 4 symbols. The committed results are on the [Performance](/project/performance/) page. |
| `python/tests/` | `just py-test` | Test modules, many of which double as usage examples — `test_pipeline_api.py` and `test_bindings_contract.py` are the two worth reading first. |

The thirteen notebooks under `notebooks/python/` are numbered in reading
order, from `01_event_labeling_and_pipeline.ipynb` to
`14_fracdiff_features_classifier.ipynb` (06 was removed; 11 replaces it). Notebooks 09 to 14 are
research runbooks, listed under [Runbooks](#runbooks) below and in the [research gallery](/runbooks/); 01 to 08 are
API tours that claim nothing (see the [notebook contract](/workflows/research-notebook-contract/)). They are committed with their outputs,
and their figures are exported to `docs-site/public/figures/notebooks/`.
All of them run on synthetic data: `09` to `14` read the SYNTHETIC `SYN_A`..`SYN_E`
sample through `openquant.data.fetch`, and run on real data only when you pass your own source.

### Figures from the executed notebooks

`just notebooks-run` writes each notebook figure here in both site themes. These two come from
`11_meta_labeling_triple_barrier.ipynb` and describe **SYNTHETIC** price paths with a planted
signal, not a real market.

<img class="dark:sl-hidden" src="/figures/notebooks/nb11-headline-equity-light.svg" alt="Cumulative net log return on a simulated planted-signal path for a moving-average crossover, the crossover filtered by a meta-labeling model, and an oracle filter that knows the hidden regime." />
<img class="light:sl-hidden" src="/figures/notebooks/nb11-headline-equity-dark.svg" alt="Cumulative net log return on a simulated planted-signal path for a moving-average crossover, the crossover filtered by a meta-labeling model, and an oracle filter that knows the hidden regime." />

<img class="dark:sl-hidden" src="/figures/notebooks/nb11-monte-carlo-light.svg" alt="Box plots over simulated paths of the precision gain and the net Sharpe ratio gain of meta-labeling over the primary model at three signal strengths, with the oracle's mean gain marked." />
<img class="light:sl-hidden" src="/figures/notebooks/nb11-monte-carlo-dark.svg" alt="Box plots over simulated paths of the precision gain and the net Sharpe ratio gain of meta-labeling over the primary model at three signal strengths, with the oracle's mean gain marked." />

### Runbooks

Studies that state a hypothesis up front and end with a promotion decision, each with a page under
Research. The [research gallery](/runbooks/) shows every runbook's hypothesis, figures, result and
decision on one page:

- [Fracdiff: stationarity versus memory](/runbooks/fracdiff-stationarity-memory/) —
  `09_fracdiff_stationarity_memory.ipynb`, the FFD d-sweep (AFML Fig. 5.5) on SYNTHETIC data,
  checked against simulated series of known memory.
- [HRP vs IVP and CLA out of sample](/runbooks/hrp-vs-ivp-cla-oos/) —
  `10_hrp_vs_ivp_cla_oos.ipynb`, AFML §16.6's Monte Carlo on SYNTHETIC data.
- [Triple-barrier labeling and meta-labeling](/runbooks/meta-labeling-triple-barrier/) —
  `11_meta_labeling_triple_barrier.ipynb`, primary vs meta-labeled bets in purged k-fold, net of
  costs, deflated by a trial registry, on SYNTHETIC paths with a planted signal and a no-signal
  control.
- [CPCV backtest with PSR and deflated Sharpe](/runbooks/cpcv-deflated-sharpe/) —
  `12_cpcv_deflated_sharpe.ipynb`, best-of-74 strategy selection judged by CPCV, PSR and DSR on
  a SYNTHETIC no-signal control and a planted signal.
- [Bet sizing from predicted probabilities](/runbooks/bet-sizing-from-probabilities/) —
  `13_bet_sizing_from_probabilities.ipynb`, the pre-registered test of runbook 11's post hoc
  finding: AFML Snippets 10.1-10.3 probability sizing against a flat 0.5 meta filter, with every
  sizing choice registered as a trial, on SYNTHETIC planted-signal paths and no-signal controls.
- [Fracdiff features against returns in a classifier](/runbooks/fracdiff-features-classifier/) —
  `14_fracdiff_features_classifier.ipynb`, runbook 09's open question: FFD features against
  returns and integer differences as purged-CV classifier inputs, net of costs, deflated by a
  trial registry, on SYNTHETIC paths with a planted signal and no-signal controls.

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
a row-order policy, not a point-in-time revision policy. A later vendor restatement may
contain information unavailable at the historical evaluation time; select the allowed
revision upstream before cleaning:

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

## Run a configuration grid and read its artifacts

The supplied grid varies event, sizing, cost and gate settings on one synthetic dataset.
Both `--config` and `--grid-config` are needed. This example uses the current interpreter
and a temporary output root, then checks the files actually written rather than relying
only on a successful exit code:

```python
import json
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
import polars as pl

with TemporaryDirectory() as directory:
    completed = subprocess.run([
        sys.executable, "experiments/run_pipeline.py",
        "--config", "experiments/configs/futures_oil_baseline.toml",
        "--grid-config", "experiments/configs/futures_oil_grid.toml",
        "--out", directory,
    ], check=True, capture_output=True, text=True)
    output = Path(completed.stdout.strip())
    manifest = json.loads((output / "run_manifest.json").read_text())
    leaderboard = pl.read_parquet(output / "leaderboard.parquet")
    runs = [path for path in output.iterdir() if path.is_dir()]
    assert manifest["mode"] == "grid"
    assert len(runs) == leaderboard.height == manifest["run_count"]
    assert set(leaderboard["run_name"]) == set(manifest["run_names"])
    for run in runs:
        child = json.loads((run / "run_manifest.json").read_text())
        assert child["dataset_hash"] == manifest["dataset_hash"]
        assert pl.read_parquet(run / "metrics.parquet").height == 1
        backtest = pl.read_parquet(run / "backtest.parquet")
        signals = pl.read_parquet(run / "signals.parquet")
        assert backtest["ts"].to_list() == signals["ts"].to_list()
        assert (run / "decision.md").is_file()
    print("saved configurations:", len(runs))
    print("shared dataset and aligned artifacts: verified")
```

```text
saved configurations: 4
shared dataset and aligned artifacts: verified
```

Use a new persistent `--out` directory to retain a study. Unlike the journey script's
exclusive report creation, this runner derives subdirectory names from configuration
hashes. It also rejects existing target directories or symlinks rather than overwriting
them; a different configuration may create a different child under the same root. Preserve the actual
configurations and dataset separately: a recorded hash is not the underlying data, and the
manifest's git revision does not prove a clean tree or matching loaded native extension.

The leaderboard ranks these same-sample results; the first row is not an independently
validated winner. `decision.md` reports threshold flags, and `backtest.parquet` records the
single-close-series strategy. Separate allocation weights in `weights.parquet` do not drive
that equity curve. Use the [connected journey](/workflows/research-journey/) for actual
fold-local fitting, and define an independent evaluation and trial inventory before
interpreting repeated searches as evidence.

Choose unique run names consisting of letters, digits, underscores or hyphens before
invoking the CLI. It validates names as nonempty single path components, rejecting
path separators, absolute paths, drive prefixes, and `.`/`..`. Duplicate names with
identical effective configurations target the same child directory and are rejected
before a grid runs. The [artifact auditor](https://github.com/Open-Quant/openquant/blob/main/skills/artifact-auditor/SKILL.md)
checks saved structure afterward; it is not a substitute for these write-time guards. See
[grid ordering and identity](/modules/research/#promotion-and-repeated-trials).

## Worked example: the whole research loop

Choose the workflow whose scope matches the task:

- [End-to-end research journey](/workflows/research-journey/) — actual fold-local
  fitting and held-out predictions from one coherent synthetic history, with a saved report.
- [Rust Core Workflow](/workflows/rust-core-workflow/) — one API program: event
  sampling → triple-barrier labels → bet sizing → purged CV → risk and
  allocation, with its printed output; supplied toy scores stand in for a model.
- [Python Core Workflow](/workflows/python-core-workflow/) — the same
  component APIs in Python, ending in workflow promotion flags. Its ingestion fixture and
  synthetic pipeline dataset are separate; it is not a fitted end-to-end backtest.
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

## Worked examples on other pages

This page executes the cleaning and grid-artifact examples above. Other examples live next to their methods.
The Python documentation gate executes runnable fences and compares following output blocks;
the Rust module gate compiles examples but does not run them. Walkthroughs and notebooks
need their own execution checks:

- **Feature diagnostics** (MDI, MDA and SFI on the same features, with
  purged folds): [Feature diagnostics](/modules/feature-diagnostics/#three-rankings-of-the-same-features),
  and the Rust side in [Feature importance](/modules/feature-importance/).
- **A CPCV backtest**: [Backtesting engine](/modules/backtesting-engine/#combinatorial-purged-cross-validation)
  for the splits and paths, and [runbook 12](/runbooks/cpcv-deflated-sharpe/) for
  CPCV path Sharpe ratios next to PSR and the deflated Sharpe ratio.

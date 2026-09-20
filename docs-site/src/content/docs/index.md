---
title: Overview
description: Rust implementations of the methods in Advances in Financial Machine Learning, with Python bindings.
template: splash
banner:
  content: '<a href="/openquant/quickstart/">New to OpenQuant? Start with the Quickstart guide →</a>'
hero:
  title: OpenQuant
  tagline: The methods from López de Prado's Advances in Financial Machine Learning, implemented in Rust and callable from Python.
  image:
    file: ../../assets/openquant-mark.svg
  actions:
    - text: Quickstart
      link: /openquant/quickstart/
      icon: right-arrow
    - text: Browse modules
      link: /openquant/modules/
      variant: minimal
status: reviewed
last_validated: '2026-09-19'
audience:
  - quant-dev
  - platform-engineering
---

OpenQuant is a library of the building blocks in *Advances in Financial Machine
Learning* (AFML): event-driven bars, CUSUM filters, triple-barrier and
meta-labels, sample weights, fractional differentiation, purged
cross-validation, backtest statistics, bet sizing, and hierarchical portfolio
construction. The numerical core is Rust; a PyO3 extension exposes it to Python.
Functions take plain sequences, and the data-loading layer works in Polars frames.

## Status

Pre-release. Version 0.1.0 is not on PyPI or crates.io, so installing means
building from source: about 10–20 minutes the first time, seconds afterwards. The
[Quickstart](/quickstart/) takes you from a clone to a printed result.

When it is published, the Python distribution will be `pyopenquant` and the
import name `openquant`. Do not `pip install openquant` — that name on PyPI
belongs to an unrelated project.

## What using it looks like

Sample events from a price series with a CUSUM filter, then label each event by
the first barrier its price path touches:

```python
import math, random
from datetime import datetime, timedelta
from openquant import filters, labeling

# 500 one-minute bars of a seeded random walk. No market data needed.
random.seed(7)
start = datetime(2024, 1, 2, 9, 30)
times = [(start + timedelta(minutes=i)).strftime("%Y-%m-%d %H:%M:%S") for i in range(500)]
close = [100.0]
for _ in range(499):
    close.append(close[-1] * math.exp(random.gauss(0, 0.001)))

# AFML ch. 2: emit an event only when cumulative drift exceeds 0.4%.
events = filters.cusum_filter_timestamps(close, times, 0.004)

# AFML ch. 3: barriers at +/-0.5%, and a vertical barrier 30 minutes out.
vertical = labeling.add_vertical_barrier(events, times, close, 0, 0, 30, 0)
labels = labeling.triple_barrier_labels(
    times, close, events, events, [0.005] * len(events),
    pt=1.0, sl=1.0, vertical_barrier_times=vertical,
)

print(f"{len(close)} bars -> {len(events)} events -> {len(labels)} labels")
print("first label:", labels[0])
print("label counts:", {b: sum(1 for row in labels if row[3] == b) for b in (-1, 1)})
```

The walk is seeded, so this prints the same thing every time:

```text
500 bars -> 42 events -> 41 labels
first label: ('2024-01-02 09:49:00', -0.001163909562213905, 0.005, -1, None)
label counts: {-1: 18, 1: 23}
```

Each label is `(event time, return at first touch, target, label, side)`. The
first event never reached either ±0.5% barrier: it ran into the 30-minute
vertical barrier with a return of −0.12%, so it is labelled by the sign of that
return. There are 41 labels for 42 events because the last event's vertical
barrier falls past the end of the data and neither horizontal barrier was
touched: its outcome is not known yet, so it gets no label instead of an invented
one. This example runs in CI on every change to the docs; if the API moves, the
build fails before this page goes stale.

## What is implemented

Every module is a Rust API. The Python column says whether the compiled
extension exposes it today.

| AFML chapters | Modules | Python |
| --- | --- | --- |
| 2 · Data structures | [data_structures](/modules/data-structures/), [filters](/modules/filters/) | yes |
| | [etf_trick](/modules/etf-trick/) | Rust only |
| 3 · Labeling | [labeling](/modules/labeling/) | yes |
| 4 · Sample weights | [sampling](/modules/sampling/), [sample_weights](/modules/sample-weights/), [sb_bagging](/modules/sb-bagging/) | yes |
| 5–6 · Features, ensembles | [fracdiff](/modules/fracdiff/), [ensemble_methods](/modules/ensemble-methods/) | yes |
| 7 · Cross-validation | [cross_validation](/modules/cross-validation/) | Rust only |
| 8–9 · Importance, tuning | [feature_importance](/modules/feature-importance/), [fingerprint](/modules/fingerprint/), [hyperparameter_tuning](/modules/hyperparameter-tuning/) | Rust only |
| 10 · Bet sizing | [bet_sizing](/modules/bet-sizing/) | yes |
| 11–13 · Backtesting | [backtesting_engine](/modules/backtesting-engine/) | Rust only |
| | [synthetic_backtesting](/modules/synthetic-backtesting/) | yes |
| 14–15 · Statistics, risk | [backtest_statistics](/modules/backtest-statistics/), [risk_metrics](/modules/risk-metrics/), [strategy_risk](/modules/strategy-risk/) | yes |
| 16 · Portfolio construction | [hrp](/modules/hrp/), [hcaa](/modules/hcaa/), [onc](/modules/onc/), [cla](/modules/cla/), [portfolio_optimization](/modules/portfolio-optimization/) | yes |
| 17–19 · Breaks, microstructure | [structural_breaks](/modules/structural-breaks/), [microstructural_features](/modules/microstructural-features/), [codependence](/modules/codependence/) | yes |
| 20–22 · HPC | [streaming_hpc](/modules/streaming-hpc/) | yes |
| | [hpc_parallel](/modules/hpc-parallel/), [combinatorial_optimization](/modules/combinatorial-optimization/) | Rust only |

The Rust-only rows are the gap that matters most for research: purged
cross-validation and the CPCV backtester are not callable from Python yet. The
[module index](/modules/) lists everything, including the pure-Python layers
(`data`, `pipeline`, `research`, `viz`), and
[By AFML Chapter](/module-reference/by-afml-chapter/) maps book sections to APIs.

## How far to trust it

- **Tests.** The Rust suite runs on every pull request. About half of its test
  files check results against fixtures derived from mlfinlab's tests; the rest
  use inline data with no external reference yet.
- **Page status.** Every page carries a status. Most module pages are still
  marked *generated* — assembled from templates and not yet read by a person
  against the book. Setup, quickstart and workflow pages are *reviewed*.
  [Coverage](/coverage/) has the current counts.
- **Examples.** Rust snippets on module pages are compiled in CI, and Python
  snippets are executed, unless a snippet is explicitly marked as illustrative.
- **Open work** is tracked in
  [GitHub issues](https://github.com/Open-Quant/openquant/issues).

## Timings

Criterion means from the regression baseline CI compares each pull request
against, recorded 2026-02-07. The hardware was not recorded and there is no
comparison against another library, so read these as orders of magnitude.

| Benchmark | Mean |
| --- | ---: |
| EWMA over 100,000 points | 4.7 ms |
| VaR, expected shortfall and CDaR on the synthetic ticker set | 8.0 ms |
| Sequential bootstrap, 2,000 samples × 600 draws | 384 ms |
| SADF structural-break statistic (`sm_power`) | 602 ms |

## Where to go next

- **To run something:** [Quickstart](/quickstart/), then the
  [Python workflow](/workflows/python-core-workflow/).
- **To work on the Rust core:** [Prerequisites](/setup/prerequisites/),
  [Local build](/setup/local-build/), [Rust workflow](/workflows/rust-core-workflow/).
- **To review methodology:**
  [Methodology and leakage controls](/governance/methodology-and-leakage-controls/).

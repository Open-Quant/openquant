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
    - text: Start with Python
      link: /openquant/quickstart/
      icon: right-arrow
    - text: Agent guide
      link: /openquant/agents/
      variant: minimal
status: authored
last_authored: '2026-11-05'
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

## Choose your path

<nav class="oq-paths not-content" aria-label="Documentation paths">
  <a class="oq-path" href="/openquant/quickstart/">
    <span class="oq-path-label">For humans</span>
    <strong>Run your first experiment</strong>
    <span>Install from source, run a small synthetic example, and learn what its results mean.</span>
    <span class="oq-path-action">Open the quickstart →</span>
  </a>
  <a class="oq-path" href="/openquant/agents/">
    <span class="oq-path-label">For agents</span>
    <strong>Work from verified sources</strong>
    <span>Find the right API, preserve research safeguards, and choose the checks your change needs.</span>
    <span class="oq-path-action">Open the agent guide →</span>
  </a>
</nav>

## Install and maturity

OpenQuant is pre-release. These docs use a source checkout so the examples,
Python wrappers and Rust extension come from the same revision. The
[Quickstart](/quickstart/) walks through installation and a first result;
[Local build](/setup/local-build/) is the contributor path.

The Python distribution is named **`pyopenquant`** in the package metadata;
the import is **`openquant`**. Do not install the unrelated PyPI project named
`openquant`. Check [release status](/project/governance/#versioning-and-releases)
before choosing a packaged release instead of the source workflow.

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
vertical = labeling.add_vertical_barrier(events, times, num_minutes=30)
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

## Find an implementation

Start with a task, then follow the linked module to its API and examples:

- **Prepare data and labels:** [bars](/modules/data-structures/),
  [filters](/modules/filters/) and [labeling](/modules/labeling/).
- **Validate without overlapping labels leaking across folds:**
  [purged cross-validation](/modules/cross-validation/) and
  [CPCV backtesting](/modules/backtesting-engine/), available through Python wrappers.
- **Evaluate a research result:** [evaluation](/modules/evaluation/),
  [cost-aware research](/modules/research/) and the [research runbooks](/runbooks/).
- **Allocate and measure risk:** [portfolio construction](/modules/portfolio-optimization/),
  [HRP](/modules/hrp/) and [risk metrics](/modules/risk-metrics/).

The [module index](/modules/) is the source of truth for the module list and
language surfaces. Use the [Python API](/api/python/) for exact import paths,
signatures and return values, or [Rust API](/api/rust/openquant/) for the core.
[By AFML chapter](/module-reference/by-afml-chapter/) connects the book to the code.

## What the checks establish

- **Examples:** the docs gate executes runnable Python snippets and checks their
  documented output. Rust examples on module pages are compile-checked, not executed.
  Explicitly skipped Python snippets are syntax-checked only.
- **Review status:** each page carries its own status. *Authored* is not a human
  review; *generated* is not a correctness claim. The
  [coverage dashboard](/coverage/) has the current counts and gaps.
- **Research limits:** synthetic results demonstrate the workflow, not trading
  performance. Read the [methodology and leakage controls](/project/governance/#methodology-and-leakage-controls)
  before interpreting a backtest.
- **Performance:** current measurements, hardware limitations and the regression
  policy live on the [performance page](/project/performance/).

## Where to go next

- **To run something:** [Quickstart](/quickstart/), then the
  [Python workflow](/workflows/python-core-workflow/).
- **To work on the Rust core:** [Prerequisites](/setup/prerequisites/),
  [Local build](/setup/local-build/), [Rust workflow](/workflows/rust-core-workflow/).
- **To review methodology:**
  [Methodology and leakage controls](/project/governance/#methodology-and-leakage-controls),
  then the [research gallery](/runbooks/).

- **To automate a task:** [Agent guide](/agents/), with source paths and validation gates.

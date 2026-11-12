---
title: Overview
description: Rust implementations of the methods in Advances in Financial Machine Learning, with Python bindings.
template: splash
banner:
  content: '<a href="/openquant/quickstart/">New to OpenQuant? Start with the Quickstart guide →</a>'
hero:
  title: '<span class="oq-eyebrow">Rust core · Python bindings</span>OpenQuant'
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
last_authored: '2026-11-12'
audience:
  - quant-dev
  - platform-engineering
---

Build research from explicit observations, labels and validation boundaries.
Use the Rust core directly or work through its Python bindings.

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

## Start with your task

| I want to… | Start here | What to keep |
| --- | --- | --- |
| Check that the library runs locally | [Quickstart](/quickstart/) | The synthetic result and the interpreter used to produce it |
| See where a model actually fits | [End-to-end research journey](/workflows/research-journey/) | One JSON report linking observations, events, training folds and predictions |
| Select parameters before a final chronological assessment | [Model selection and holdout](/workflows/model-selection-holdout/) | Development-only search, fold-local preprocessing, an explicit refit and the untouched holdout score |
| Select only revisions available at a research cutoff | [Point-in-time data](/workflows/point-in-time-data/) | Availability timestamps, selection policy and selected revisions; a hash does not establish provider truth |
| Load my own CSV or Parquet | [Data loading and provenance](/modules/data/), then the [CSV example](/examples/catalog/) | Cleaned data, quality report, source terms and dataset hash |
| Turn aligned observations into named portfolio weights | [Multi-asset allocation workflow](/workflows/multi-asset-allocation/) | Common timestamps, asset-column order, moment units and the final fitting date |
| Reproduce a documented experiment | [Research runbooks](/runbooks/) and [notebook execution](/workflows/notebook-research-workflow/) | The selected notebook, its outputs and the comparison limits |
| Find an exact call or port it to Rust | [Python API](/api/python/) or [Rust API](/api/rust/openquant/) | Argument order, return shape and the module's caller preconditions |
| Fix a build or import failure | [Troubleshooting](/setup/troubleshooting/) | The interpreter path, extension path and failing command |

The quickstart consumes synthetic probabilities. The research journey fits a fixed baseline
inside walk-forward folds; the model-selection walkthrough adds a parameter search and a
separate final holdout. Use the latter two to follow training boundaries. Their classification
results do not provide a trading P&L backtest.

## A research workflow, at a glance

Each stage has a different job. Keep the data, label intervals, validation splits
and evaluation assumptions explicit as you move between them.

<figure class="oq-workflow not-content">
  <ol class="oq-research-map" role="list" aria-label="Research workflow stages">
    <li>
      <span class="oq-step">01 →</span>
      <strong>Prepare</strong>
      <svg viewBox="0 0 160 48" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true" focusable="false"><path d="M8 42H152"/><path d="M18 34V24M36 34V14M54 34V20M72 34V8M90 34V16M108 34V4M126 34V12M144 34V6" class="oq-signal" stroke-width="7"/></svg>
      <p>Turn observations into bars and sample meaningful events.</p>
      <a href="/openquant/modules/data-structures/">Bars</a> · <a href="/openquant/modules/filters/">Filters</a>
    </li>
    <li>
      <span class="oq-step">02 →</span>
      <strong>Label</strong>
      <svg viewBox="0 0 160 48" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true" focusable="false"><path d="M8 8H152M8 40H152M128 4V44" stroke-dasharray="4 4"/><path d="m8 28 18-7 20 9 20-13 20 5 20-14" class="oq-signal"/><circle cx="106" cy="8" r="4" class="oq-signal"/></svg>
      <p>Define outcomes and retain their start and end times.</p>
      <a href="/openquant/modules/labeling/">Triple-barrier labels</a>
    </li>
    <li>
      <span class="oq-step">03 →</span>
      <strong>Validate</strong>
      <svg viewBox="0 0 160 48" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true" focusable="false"><path d="M8 8H62V20H8ZM8 28H94V40H8Z"/><path d="M82 8H118V20H82ZM114 28H150V40H114Z" class="oq-signal"/><path d="M70 4V24M102 24V44" stroke-dasharray="2 3"/></svg>
      <p>Separate training from testing. Control overlapping labels.</p>
      <a href="/openquant/modules/cross-validation/">Purged cross-validation</a>
    </li>
    <li>
      <span class="oq-step">04</span>
      <strong>Evaluate</strong>
      <svg viewBox="0 0 160 48" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true" focusable="false"><path d="M8 8H152M8 24H152M8 40H152"/><path d="M20 5V43M108 5V43"/><path d="M34 16H78M122 16H140M34 32H64M122 32H140" class="oq-signal" stroke-width="3"/></svg>
      <p>Read diagnostics with costs, uncertainty and limits in view.</p>
      <a href="/openquant/modules/evaluation/">Evaluation</a> · <a href="/openquant/modules/research/">Research</a>
    </li>
  </ol>
  <figcaption>Conceptual workflow, not a performance result. The sketches show each stage's role, not measured data; the linked modules document their contracts.</figcaption>
</figure>

## Install and maturity

**OpenQuant is pre-release.** These docs use a source checkout so the examples,
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

- **Prepare data and labels:** [load and inspect data](/modules/data/), [bars](/modules/data-structures/),
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
  documented output. Rust module examples and the complete Rust core workflow are
  compile-checked, not executed.
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
  [research journey](/workflows/research-journey/) for fitting and prediction, or the
  [Python workflow](/workflows/python-core-workflow/) for the convenience pipeline.
- **To work on the Rust core:** [Prerequisites](/setup/prerequisites/),
  [Local build](/setup/local-build/), [Rust workflow](/workflows/rust-core-workflow/).
- **To review methodology:**
  [Methodology and leakage controls](/project/governance/#methodology-and-leakage-controls),
  then the [research gallery](/runbooks/).

- **To automate a task:** [Agent guide](/agents/), with source paths and validation gates.

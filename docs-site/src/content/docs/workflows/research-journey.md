---
title: End-to-end research journey
description: Follow one synthetic dataset through bars, labels, training weights and walk-forward predictions.
status: authored
last_authored: '2026-11-07'
audience:
  - quant-dev
  - platform-engineering
sidebar:
  order: 3
---

**Start here to see how the parts connect.** This example follows one dataset all
the way from observations to out-of-sample predictions. It does not swap in an
unrelated dataset after ingestion, and it shows where fitting actually happens.

The [quickstart](/quickstart/) demonstrates the convenience research pipeline.
That pipeline **accepts model probabilities as an input**. This journey covers the
earlier work needed to produce and inspect predictions, using a tiny baseline you
can understand without installing a machine-learning framework.

:::caution[An integration example, not evidence of a strategy]
All observations are synthetic. Scores are uncalibrated frequency estimates,
labels ignore costs, and the output is a classification report, not P&L. Passing
this example does not establish profitability, independence or absence of every
form of leakage.
:::

## Choose the next step

- **New installation:** complete the [quickstart](/quickstart/), then return here to see
  how training differs from consuming supplied probabilities.
- **Ready to run this example:** follow [Run one command](#run-one-command), then
  [read the JSON report](#read-the-artifact-before-the-score).
- **Bringing your own data:** first use [data loading and quality reports](/modules/data/).
  Adapt the script explicitly; it generates its own synthetic observations and is not a
  CSV-import command. Preserve the [connected handoffs](#the-connected-path).
- **Looking for a saved research experiment:** use the [runbook gallery](/runbooks/)
  and [notebook execution guide](/workflows/notebook-research-workflow/). This page's
  script produces a JSON classification report, not an executed research notebook.

## Run one command

First complete [Python bindings setup](/setup/python-bindings/). From that same
repository root, with the extension built into its environment:

```bash
uv run --no-sync --python .venv/bin/python python examples/research_journey.py \
  --out /tmp/openquant-journey.json
```

Choose a **new output filename** for every run. The script refuses to overwrite an
existing report. On Windows, use a writable path such as `openquant-journey.json`
instead of `/tmp/openquant-journey.json`.

No API keys, downloads, market-data subscriptions or network access are needed
after setup. The default seed is 7 and the input has 1,024 observations. The
default run in the tested environment produced **489 complete dollar bars, 201
resolved events and 160 walk-forward predictions**. These counts describe a
fixture, not an acceptance threshold for another seed or dependency version.

Source: [examples/research_journey.py](https://github.com/Open-Quant/openquant/blob/main/examples/research_journey.py).
Regression tests: [test_research_journey.py](https://github.com/Open-Quant/openquant/blob/main/python/tests/test_research_journey.py).

## The connected path

| Stage | Call | Output handed to the next stage |
| --- | --- | --- |
| Normalize | `data.clean_ohlcv(..., return_report=True)` | One ordered OHLCV frame and its quality report |
| Sample activity | `bars.build_dollar_bars(...)` | The same observations grouped into dollar bars |
| Locate events | `filters.cusum_filter_timestamps(...)` | Event start times on that bar series |
| Define horizons | `labeling.add_vertical_barrier(...)` | A maximum end time for each eligible event |
| Resolve outcomes | `labeling.triple_barrier_events(...)`, then `get_bins(...)` | Start/end spans, gross returns and binary labels |
| Separate past from future | `cross_validation.walk_forward_split_with_diagnostics(...)` | Training/test **event indices**, with overlapping training spans removed |
| Weight training data | `sample_weights.get_weights_by_return(...)` | One weight per training event, recomputed in each fold |
| Fit and predict | `fit_bucket_probabilities(...)` in the example | A fixed, two-bucket frequency baseline fitted only on that fold's training events |
| Inspect outcomes | `evaluation.meta_label_metrics(...)` | Classification counts and metrics for tested events only |

The last two stages are separate on purpose. Splitters do not fit models, and the
pipeline cannot infer whether supplied probabilities were fitted on future data.

### 1. Keep one coherent data history

The generator creates a seeded Gaussian **log-return** random walk. Each input row
is one synthetic price and volume observation, with open/high/low/close equal.
It is not a realistic candle or a calibrated market model.

Dollar bars close at $200,000 of synthetic price × volume. The builder treats each
input row as a trade at its close and ignores the input high/low. That is suitable
for these trade-like observations; it would not recover the true intrabar range
from real OHLC candles. The final partial bar is excluded so the example does not
treat the end of a file as an activity-triggered close.

The quality report and hashes identify the cleaned input and complete bar frame.
Keep both when adapting the example. A clean schema does not prove vendor data
was available at the historical time being studied.

### 2. Label an explicit question

The primary hypothesis is **always long at an event**. Its meta-label is 1 if the
gross close-to-close return is positive at the first resolved barrier, otherwise 0.
There is no fitted primary trading model here.

- CUSUM threshold: 0.003 in log-return units.
- Horizontal target: 0.004 in simple-return units, with profit/stop multiples of 1.
- Maximum horizon: 12 elapsed minutes, resolved to the first bar at or after it.
- Missing vertical horizons at the dataset tail: excluded, not silently shortened.
- Barrier touches: evaluated on closes, not intrabar highs/lows or simulated fills.

These constants are chosen in advance for the example. Optimizing them against
the reported scores would make those scores part of model selection.

Each event retains `t0`, `t1`, `start_bar` and `end_bar`. The feature is the log
return ending at `t0`, so it is available at the event bar's close. The label uses
future prices through `t1`. Keeping those two availability times distinct is what
makes correct purging possible.

### 3. Fit only on resolved history

Five contiguous event folds produce four tests. The first fold is training-only.
For each later fold, training events must occur earlier **and have resolved before
the first test event starts**. Spans are closed intervals, so sharing an endpoint
counts as overlap. No model trains on a future fold.

Within each training fold, return-attribution weights use only its training events
to count concurrency. Even the price series passed to the weighting call stops at
the last training label's end. Computing weights once on every event would let
held-out event spans influence the training weights.

The baseline divides the causal feature into negative/nonnegative buckets. In
each bucket it estimates `(1 + weighted positive labels) / (2 + total weight)`.
The added pseudo-counts prevent an empty bucket from producing an undefined value.
This is intentionally simple example code, **not a new OpenQuant estimator API**.

A second baseline ignores the feature and estimates one frequency from that same
training fold. Comparing against it makes class imbalance visible. Neither
baseline's frequency scores are validated as calibrated probabilities.

Later folds may train on earlier test events once their labels resolve. This is an
expanding walk-forward evaluation, not a final untouched holdout. Test labels may
overlap each other, so their classification outcomes are not independent trials.

## Read the artifact before the score

The JSON report contains:

- `config` and `provenance`: fixed procedure, seed, example source hash, repository
  commit, Python, NumPy and Polars versions. Rebuild the extension after Rust edits;
  a repository commit alone does not prove the loaded extension matches it.
- `data`: cleaned observation/bar hashes, quality report and counts, including
  starts excluded because their full vertical horizon was unavailable.
- `events`: every feature, label, gross event return and exact start/end bar.
- `folds`: train/test/purged event indices, fold-local training weights and fitted
  bucket frequencies.
- `predictions`: one row for each tested event, with its fold and both baselines'
  scores. Initial training-only events are deliberately absent.
- `metrics` and `baseline_metrics`: counts, precision, recall, F1 and accuracy on
  those prediction rows only. `primary_*` describes taking every always-long bet.
- `limitations`: the interpretation limits that travel with every saved report.

Do not sum or compound `event_return`: these are overlapping label outcomes, not a
non-overlapping account return series. Accuracy also ignores the size of gains and
losses, transaction costs and capacity.

## Read back and check the handoffs

Run this from the repository root with the same Python environment as the example.
`check_report(path)` can inspect a saved JSON report without recomputing it. To make the
snippet runnable on its own, the final lines generate a fresh report at a unique temporary
path, read it back, then remove that temporary directory. Your earlier saved report is not
overwritten. Keep a durable report from [Run one command](#run-one-command) for comparisons.

```python
import json
import math
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory


def check_report(path):
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    assert report["schema_version"] == 1
    events, folds, predictions = report["events"], report["folds"], report["predictions"]
    assert len(events) == report["data"]["event_rows"]
    assert [e["sample_id"] for e in events] == list(range(len(events)))
    assert all(0 <= e["start_bar"] <= e["end_bar"] < report["data"]["bar_rows"]
               for e in events)
    by_fold = {fold["split_id"]: fold for fold in folds}
    assert len(by_fold) == len(folds)
    expected = set()
    for fold in folds:
        train, test = fold["train_indices"], fold["test_indices"]
        assert train and test and not set(train).intersection(test)
        assert all(0 <= i < len(events) for i in train + test)
        assert len(fold["training_weights"]) == len(train)
        assert all(math.isfinite(w) and w >= 0 for w in fold["training_weights"])
        assert max(events[i]["end_bar"] for i in train) < min(events[i]["start_bar"] for i in test)
        expected.update((fold["split_id"], i) for i in test)
    observed = [(p["split_id"], p["sample_id"]) for p in predictions]
    assert len(observed) == len(set(observed)) and set(observed) == expected
    assert len({p["sample_id"] for p in predictions}) == len(predictions)
    for prediction in predictions:
        event = events[prediction["sample_id"]]
        fold = by_fold[prediction["split_id"]]
        bucket = "nonnegative" if event["feature"] >= 0 else "negative"
        assert prediction["probability"] == fold["bucket_probabilities"][bucket]
        assert prediction["constant_probability"] == fold["constant_probability"]
        assert all(prediction[key] == event[key] for key in ("t0", "t1", "label"))
        assert math.isfinite(prediction["probability"]) and 0 <= prediction["probability"] <= 1
    threshold = report["config"]["decision_threshold"]
    correct = sum(p["label"] == (p["probability"] >= threshold) for p in predictions)
    assert predictions and math.isclose(report["metrics"]["accuracy"], correct / len(predictions))
    print(f"Checked {len(events)} events, {len(folds)} folds, {len(predictions)} predictions")
    return report


with TemporaryDirectory() as directory:
    output = Path(directory) / "journey.json"
    subprocess.run([sys.executable, "examples/research_journey.py", "--out", str(output)],
                   check=True, capture_output=True, text=True)
    report = check_report(output)
```

```text
Checked 201 events, 4 folds, 160 predictions
```

The counts above describe the default fixture. The assertions instead check relationships:
event IDs index the event list, training labels end before a test begins, each prediction
belongs to a recorded test fold, and reported accuracy agrees with those prediction rows.
They do not prove that the fitted model used only those rows or that features were available
historically. The script's [integration tests](https://github.com/Open-Quant/openquant/blob/main/python/tests/test_research_journey.py)
add independent weight arithmetic and perturbation checks; neither is a profitability test.

Before comparing two saved reports, compare `config`, the observation/bar hashes and
`provenance`, then inspect fold membership and predictions. A changed seed or source is a
different experiment. The JSON stores hashes rather than the complete observation/bar
frames, so it cannot reconstruct an external dataset by itself. Retain the actual inputs
and their historical availability alongside adapted reports. A commit and example-source
hash identify recorded source information, not a verified native binary or a clean checkout.

## Where the other modules fit

Not every module belongs in one mandatory pipeline. Choose the next task:

| Your next question | Modules and required handoff |
| --- | --- |
| Are the features useful? | [Feature diagnostics](/modules/feature-diagnostics/), [fracdiff](/modules/fracdiff/) and [microstructure](/modules/microstructural-features/). Fit transformations and select features within training folds; retain feature availability and label spans. |
| Which model or parameters should I choose? | [Hyperparameter tuning](/modules/hyperparameter-tuning/) inside a training-only search, then separate evaluation. Reusing the same test folds for choices consumes their independence. |
| How do probabilities become positions? | [Bet sizing](/modules/bet-sizing/) and [pipeline](/modules/pipeline/). Supply explicitly aligned, out-of-sample, suitably calibrated probabilities and primary sides; a frequency score from this demo is not enough. |
| What would executions have returned? | Define order timing, fills, fees, slippage and overlapping-signal handling before producing **one account return per time step**. The [Python workflow](/workflows/python-core-workflow/) shows the existing convenience pipeline and its limits. |
| How do several assets combine? | [Portfolio optimization](/modules/portfolio-optimization/), [HRP](/modules/hrp/) and [risk metrics](/modules/risk-metrics/). Align asset returns, fit allocations on training history and evaluate later realized portfolio returns. An in-sample allocator optimum is not strategy performance. |
| How sensitive is validation to the split? | [Cross-validation](/modules/cross-validation/) and [CPCV engine](/modules/backtesting-engine/). CPCV also trains on observations after a middle test fold; purging labels does not make that equivalent to forward-only forecasting. |
| How should repeated trials be reported? | [Evaluation](/modules/evaluation/) and its trial registry. Record every trial and the assumptions behind PSR/DSR; neither fixes an adaptively reused holdout. |
| How do I retain comparable runs? | [Research tools](/modules/research/), manifests and the [notebook research workflow](/workflows/notebook-research-workflow/). Keep inputs, configuration, software version and failed candidates, not just the winner. |

For exact parameter and return definitions while adapting a stage, use the
[Python API reference](/api/python/). If installation or imports fail, return to
[troubleshooting](/setup/troubleshooting/) before changing the example's logic.

## Instructions for an agent adapting this example

1. Read the script and public API signatures from the **same checkout**. Use the
   [agent guide](/agents/) for repository navigation and generated discovery data.
2. Run the unchanged example first and keep its JSON. Use a new output filename
   for every changed experiment; do not hide an overwritten baseline.
3. Change one handoff at a time. Preserve symbol ordering, event-to-bar mappings,
   full label spans and the time at which every input becomes available.
4. Keep fitting, weighting, normalization, feature selection and calibration inside
   the training boundary. Do not replace walk-forward with shuffled CV.
5. Run the example's integration tests, then the full Python suite:

```bash
uv run --python .venv/bin/python pytest python/tests/test_research_journey.py -q
uv run --python .venv/bin/python pytest python/tests -q
```

6. Report what ran, what failed and what remains untested. Do not interpret
   `has_forward_look_bias=False` from the convenience pipeline as a model audit;
   that legacy field is a constant, not a detector.
7. Before using real data or execution, specify the historical universe, corporate
   actions, time zones, data release times, cost model and final holdout. This
   example supplies none of those real-world guarantees.

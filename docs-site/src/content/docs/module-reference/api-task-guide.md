---
title: Find an API by task
description: Choose a Python wrapper, native binding, or Rust API for a concrete research task.
status: authored
last_authored: '2026-11-07'
audience:
  - quant-dev
  - platform-engineering
sidebar:
  order: 1
---

Start with the task below, then follow the linked signature and its input contract.
The [module index](/modules/) groups concepts; the [Python reference](/api/python/)
lists importable namespaces, and [rustdoc](/api/rust/openquant/) lists Rust paths.
A module guide's title is not necessarily its Python import name.
If a call rejects your inputs or returns an unsuitable result, use
[Diagnose input errors](/setup/input-errors/).

## Choose the operation

| Task | Python entry point | What to check before using it |
| --- | --- | --- |
| Preserve selected source values through cleaning | [`provenance.select_revisions`](/api/python/provenance/#select_revisions), [`rejoin_selected`](/api/python/provenance/#rejoin_selected) | Select before cleaning and rejoin by stable IDs. [Exact typed identity](/modules/provenance/) does not prove provider authenticity or history completeness. |
| Assemble complete declared feature rows | [`feature_assembly.assemble_features`](/api/python/feature_assembly/#assemble_features) | Specify feature order and required source fields. Retain [missing-input exclusions](/modules/feature-assembly/) and remap original row indices after filtering. |
| Audit training-time availability | [`availability.select_available_training`](/api/python/availability/#select_available_training) | Supply [explicit source and label clocks](/modules/availability/) and the splitter's original training indices; this filter does not replace purging or fold-local preprocessing. |
| Build meta-labels from a primary model's sides | [`labeling.get_events`](/api/python/labeling/#get_events), [`labeling.get_bins`](/api/python/labeling/#get_bins) | Preserve [primary-side provenance](/modules/labeling/#preserve-the-primary-sides-provenance). These functions consume supplied sides; they do not fit either model or enforce out-of-sample prediction boundaries. |
| Load and inspect local OHLCV data | [`data.load_ohlcv`](/api/python/data/#load_ohlcv), [`data.data_quality_report`](/api/python/data/#data_quality_report) | First resolve [point-in-time revision availability](/workflows/point-in-time-data/); loading and cleaning do not enforce it. The Python interface takes frames/files; the native [`_core.data`](/api/python/core-data/) exposes separate column and frame calls. Rust uses [`data_processing`](/api/rust/openquant/data_processing/). |
| Explain why rows left a validation fold | [`cross_validation.split_with_diagnostics`](/api/python/cross_validation/#split_with_diagnostics) | Supply ordered label spans in consistent units. Diagnostics describe purge/embargo exclusions, not whether feature construction leaked. |
| Bootstrap labels inside a training fold | [`sampling.seq_bootstrap`](/api/python/sampling/#seq_bootstrap); [worked handoff](/modules/sampling/#bootstrap-only-the-training-fold) | Draws index training-local label columns. Map through training indices to original rows, preserve multiplicity across fitting arrays, and keep test rows excluded. Repeated draws are not independent new observations. |
| Score predictions already produced out of sample | [`feature_importance.mda_from_probabilities`](/api/python/feature_importance/#mda_from_probabilities), [`sfi_from_probabilities`](/api/python/feature_importance/#sfi_from_probabilities) | These consume predictions you supply; they do not fit your model. Preserve original rows and feature columns; see [saved-prediction handoffs](/modules/feature-importance/#preserve-rows-and-feature-columns-when-supplying-probabilities). |
| Fit the built-in diagnostic models on features | [`feature_diagnostics.mda_importance`](/api/python/feature_diagnostics/#mda_importance), [`sfi_importance`](/api/python/feature_diagnostics/#sfi_importance) | These are different interfaces from the prediction consumers above. Read the [diagnostic model and preprocessing assumptions](/modules/feature-diagnostics/). |
| Track the trial count used for selection-adjusted reporting | [`evaluation.TrialRegistry`](/api/python/evaluation/#TrialRegistry) | Counts unique explicit configuration keys; repeating a key replaces its entry. Include all attempted choices in those keys; see the [trial-identity contract](/workflows/research-notebook-contract/#trial-identity-is-part-of-the-method). |
| Allocate using supplied returns or covariance | [`portfolio.allocate_from_inputs`](/api/python/portfolio/#allocate_from_inputs) | Follow [observations to named weights](/workflows/multi-asset-allocation/) for timestamp/asset alignment, then check covariance validity, bounds and units in [portfolio optimization](/modules/portfolio-optimization/). |
| Assemble predictions into CPCV paths | [`backtesting_engine.assemble_cpcv_paths`](/api/python/backtesting_engine/#assemble_cpcv_paths) | Restore [split and sample order](/modules/backtesting-engine/#restore-split-and-sample-order-before-assembly) before assembly. Paths reuse observations and are dependent; their spread is descriptive. |
| Convert an API result into a frame for analysis | [`adapters`](/api/python/adapters/) | Choose the adapter for the result type. Conversion does not establish timestamp alignment, calibration, or account-level P&L. |

For an integrated example, use the [Python workflow](/workflows/python-core-workflow/)
or [Rust workflow](/workflows/rust-core-workflow/). Both are API demonstrations;
training and held-out evaluation remain separate work. For a fitted baseline and
held-out predictions, follow the [end-to-end research journey](/workflows/research-journey/).
For parameter selection before one final chronological assessment, use the
[model-selection walkthrough](/workflows/model-selection-holdout/). It connects purged
inner splits, training-only preprocessing and explicit refitting; its final classification
score is separate from the search score and is not trading P&L.

## Python wrappers and native calls

Use `openquant.<namespace>` as the starting point. Seven namespaces have a
Python wrapper and a separately documented native surface:
`backtesting_engine`, `bars`, `cross_validation`, `data`, `feature_importance`,
`hyperparameter_tuning`, and `pipeline`. Their low-level references are named
`openquant._core.<namespace>`. A shared function name does not imply identical
accepted inputs or return types.

For example, the validation wrapper returns NumPy index arrays while its native
counterpart returns lists. This small example also checks that fold row positions
agree; the integers here are abstract, consistently ordered time units.

```python
import numpy as np
from openquant import cross_validation
from openquant._core import cross_validation as native_cv

t0 = [0, 2, 4, 6, 8, 10]
t1 = [1, 3, 5, 7, 9, 11]
wrapped = cross_validation.purged_kfold_splits(t0, t1, 3, 0.0)
native = native_cv.purged_kfold_splits(t0, t1, 3, 0.0)
assert isinstance(wrapped[0][0], np.ndarray)
assert isinstance(native[0][0], list)
assert [(train.tolist(), test.tolist()) for train, test in wrapped] == native
print(wrapped[0][0].tolist(), wrapped[0][1].tolist())
```

```text
[2, 3, 4, 5] [0, 1]
```

Other native namespaces such as `risk`, `hrp`, and `fracdiff` are re-exported
directly at the package root. Pure-Python namespaces such as `evaluation`,
`feature_diagnostics`, and `viz` have no same-name `_core` module.

## Translate a name between languages

These common aliases help when moving from a Python example to rustdoc. They
identify related modules, not interchangeable function signatures.

| Python namespace | Rust module |
| --- | --- |
| `bars` | [`data_structures`](/api/rust/openquant/data_structures/) |
| `data` | [`data_processing`](/api/rust/openquant/data_processing/) for OHLCV operations; Python caching/fetching is a separate layer |
| `risk` | [`risk_metrics`](/api/rust/openquant/risk_metrics/) |
| `portfolio` | [`portfolio_optimization`](/api/rust/openquant/portfolio_optimization/) |
| `backtest_stats` | [`backtest_statistics`](/api/rust/openquant/backtest_statistics/) |
| `synthetic_bt` | [`synthetic_backtesting`](/api/rust/openquant/synthetic_backtesting/) |
| `microstructural` | [`microstructural_features`](/api/rust/openquant/microstructural_features/) |
| `ensemble` | [`ensemble_methods`](/api/rust/openquant/ensemble_methods/) |
| `fast_ewma`, `volatility` | [`util::fast_ewma`](/api/rust/openquant/util/fast_ewma/), [`util::volatility`](/api/rust/openquant/util/volatility/) |

## Tasks that currently require Rust

These public Rust APIs have no Python binding. Similar names elsewhere do not
supply the same operation.

| Task | Rust entry point and worked guide |
| --- | --- |
| Build a rebalanced futures basket or roll series | [`etf_trick::EtfTrick`](/api/rust/openquant/etf_trick/struct.EtfTrick.html), [`get_futures_roll_series`](/api/rust/openquant/etf_trick/fn.get_futures_roll_series.html); [guide](/modules/etf-trick/) |
| Decompose a fitted predictor's feature effects | [`fingerprint::RegressionModelFingerprint`](/api/rust/openquant/fingerprint/struct.RegressionModelFingerprint.html); [guide](/modules/fingerprint/). Python feature importance measures are different diagnostics. |
| Dispatch a user-supplied computation over partitions | [`hpc_parallel::run_parallel`](/api/rust/openquant/hpc_parallel/fn.run_parallel.html); [guide](/modules/hpc-parallel/). Python `streaming_hpc` supplies specific streaming calculations, not this generic executor. |
| Supply a custom integer objective or solver adapter | [`combinatorial_optimization::IntegerObjective`](/api/rust/openquant/combinatorial_optimization/trait.IntegerObjective.html), [`SolverAdapter`](/api/rust/openquant/combinatorial_optimization/trait.SolverAdapter.html); [guide](/modules/combinatorial-optimization/). Python `dynamic_allocation` exposes a particular allocation problem, not these traits. |

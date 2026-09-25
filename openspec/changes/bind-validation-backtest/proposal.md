## Why

`cross_validation`, `backtesting_engine`, `feature_importance` and `hyperparameter_tuning` have
no Python bindings (brief RQ-010, issue #42), and every runbook needs them. The Rust functions
that fit models take a `SimpleClassifier` or an evaluator closure, so they cannot be bound as
they are without passing Python callables into Rust. Brief decision 4 (ADR 0002) settled that
runbooks model with scikit-learn from Python, so the Python surface returns index splits and
path assignments and scores results the caller computed.

## What Changes

- New `openquant.cross_validation`: `purged_kfold_splits`, `split_with_diagnostics`,
  `cpcv_splits`, `cpcv_paths`, `naive_kfold_splits` and `count_train_test_overlaps`, returning
  numpy index arrays. Label spans `t0`/`t1` are required arguments.
- New `openquant.backtesting_engine`: `cpcv_path_count`, `run_cpcv` over per-split
  out-of-sample returns the caller computed, and `assemble_cpcv_paths` (pure numpy).
- New `openquant.feature_importance`: `mean_decrease_impurity` from per-tree importances;
  `mda_from_probabilities` and `sfi_from_probabilities` score out-of-sample probabilities
  with the Rust MDA / SFI; `mean_decrease_accuracy` and `single_feature_importance` run the
  fit loop in Python for any estimator with `fit` and `predict_proba`. All CV variants take
  the label spans and build purged folds from them; there is no way to pass other folds.
- New `openquant.hyperparameter_tuning`: `expand_param_grid`, `sample_param_sets`,
  `classification_score` and `purged_search` (a Python fit loop over purged folds).
- Rust: `hyperparameter_tuning::sample_param_sets`, extracted from `randomized_search`
  unchanged, so Python can evaluate exactly the candidates the Rust search would.
- `numpy` becomes a runtime dependency of the Python package.

## Capabilities

### New Capabilities
- `python-research-bindings`: the four Python modules above.

### Modified Capabilities

## Impact

- Code: `crates/pyopenquant/src/{cross_validation,backtesting_engine,feature_importance,hyperparameter_tuning}.rs`,
  `python/openquant/` modules of the same names, `crates/openquant/src/hyperparameter_tuning.rs`.
- Tests: `python/tests/test_core_*.py` for the four modules; one Rust test for
  `sample_param_sets`.
- Docs: the four module pages gain a Python section and `api_surface: both`; API inventory.
- Dependencies: `numpy>=1.26,<3` (runtime). scikit-learn is not added; its tests skip without it.

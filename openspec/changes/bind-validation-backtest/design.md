## Context

The Rust `ml_cross_val_score`, `mean_decrease_accuracy`, `single_feature_importance`,
`grid_search`, `randomized_search` and `backtesting_engine::run_*` all call a model or an
evaluator that Rust owns. The splitters (`PurgedKFold::split`, `split_with_diagnostics`,
`cpcv_splits`, `cpcv_paths`, from #33) and the scorers do not.

## Goals / Non-Goals

**Goals:**
- Any scikit-learn-style model can be validated, backtested, explained and tuned on purged
  folds from Python.
- Every number Python reports is computed by the Rust code the Rust tests check.
- Cross-validated feature importance cannot run on unpurged folds.

**Non-Goals:**
- Passing Python callables into Rust.
- Changing split, purge, embargo or scoring behaviour. The two-sided embargo of `PurgedKFold`
  is #134's decision; the engine embargo and MDA/MDI standard errors are #132's; the MDA
  permutation is #127's.
- Walk-forward and purged-CV modes of `backtesting_engine` (only CPCV is bound).

## Decisions

- **Indices out, results in.** Splitters return numpy arrays; the caller fits. Results come
  back as arrays aligned to the splits and are scored in Rust.
- **Replay instead of callbacks.** To score MDA and SFI with the Rust functions, the binding
  passes a Rust `SimpleClassifier` whose `predict_proba` returns the caller's probabilities in
  the order the Rust function asks for them. `run_cpcv` gets a Rust closure that returns the
  caller's returns for each split. The alternative, making the aggregation helpers public in
  the core crate, changes more Rust for the same result. The replay checks that every block
  was consumed with the expected length, so a change in the Rust call order fails loudly.
- **Folds rebuilt from spans.** The `*_from_probabilities` functions take `t0`/`t1` and
  `n_splits`/`pct_embargo`, not splits, and rebuild the purged folds. Because purged k-fold
  test sets partition the samples, the caller's probabilities are one column per sample.
- **Timestamps as int64.** The Python layer converts `t0`/`t1` (datetime64, pandas/polars
  datetimes, `datetime`, ISO strings, or plain integers) to int64 nanoseconds; Rust maps
  them to `NaiveDateTime`. This keeps sub-second precision without the string wire format
  (#129) and lets bar positions serve as spans.
- **Module names follow the Rust modules and doc pages** (`backtesting_engine`, not
  `backtesting`), so the docs coverage gate maps each to its page.
- **`sample_param_sets` in Rust.** Its draws must match `randomized_search`, so the loop moves
  out of `randomized_search` unchanged rather than being re-implemented in Python.

## Risks / Trade-offs

- [Replay depends on the Rust call order] → checked at runtime and by the hand-worked
  parity tests; #127 keeps the order.
- [`run_cpcv` re-purges its own copy of the splits with a different rule] → only its test
  indices, which equal `cpcv_splits`' (tested), and the caller's returns are used; its
  training sets are reported, not trained on. Documented.
- [When #127 lands, Rust MDA takes a `seed`] → the binding passes any constant; the replay
  ignores the permutation. When #132 lands, MDA and MDI standard errors change; the Python
  tests assert the same brackets as the Rust tests.

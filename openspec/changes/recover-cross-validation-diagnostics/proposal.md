## Why

Commit `59ac6fa` (beads OQ-nbr.4) added purged cross-validation diagnostics and CPCV splits,
but it never reached `main` (issue #33, brief RQ-002). The Python bindings for validation
(`bind-validation-backtest`, #42) depend on it. `main` has since gained typed errors, no-panic
APIs and a purge-window fix, so the commit is re-ported by hand rather than rebased.

## What Changes

- `PurgedKFold::split_with_diagnostics` returns the folds of `split` and, for each, the test
  ranges, the purged and embargoed indices, and a post-purge overlap count.
- `PurgedKFold::cpcv_splits` returns the C(N, k) combinatorial purged splits (AFML §12.4).
  The original commit called this method `cpcv_paths`. The name now belongs to the next item.
- `PurgedKFold::cpcv_paths` returns the φ[N, k] = k/N · C(N, k) backtest paths: for each fold,
  which split's predictions a path uses.
- `naive_kfold_splits` (unpurged baseline) and `count_train_test_overlaps` (leakage count).
- `PurgedKFold::new` now rejects `pct_embargo` outside `[0, 1)` or non-finite, and
  information sets that end before they start. **BREAKING** for callers that passed such
  values. They were previously accepted and gave meaningless splits.
- `PurgedKFold::split` is re-implemented on top of the diagnostics. Its folds are unchanged,
  including the two-sided embargo counted from the fold edges (#134 decides that separately).

## Capabilities

### New Capabilities
- `purged-cross-validation`: purged k-fold diagnostics, CPCV splits and paths, the naive
  k-fold baseline and the train/test overlap count in `openquant::cross_validation`.

### Modified Capabilities

## Impact

- Code: `crates/openquant/src/cross_validation.rs`. `CrossValidationError` gains variants
  `InvalidEmbargo`, `InvalidInfoSet`, `InvalidTestSplits`, `TooManySplits` and
  `IndexOutOfRange`.
- Tests: `crates/openquant/tests/cross_validation.rs`.
- Docs: `docs-site/src/content/docs/modules/cross-validation.md`, API inventory.
- No Python binding in this change (#42).

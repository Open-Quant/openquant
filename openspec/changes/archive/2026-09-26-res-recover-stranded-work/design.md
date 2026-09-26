## Context

`59ac6fa` was written against a `cross_validation.rs` that returned `Result<_, String>`,
purged from the first test label's end, and had no pct_embargo validation. Since then `main`
gained typed errors (#85), no-panic APIs (#84) and the purge-window fix (#65). The commit
also rewrote `split` with a one-sided embargo, which is a behaviour change that #134 now owns.

## Goals / Non-Goals

**Goals:**
- The four public functions of `59ac6fa`, with its tests, on current `main`.
- Diagnostics that explain `split` exactly, so `split` can be built on them.
- CPCV counts that match AFML §12.4.

**Non-Goals:**
- Changing the embargo (two-sided, counted from fold edges): #134.
- Python bindings: #42.
- Replacing `backtesting_engine::run_cpcv`, which evaluates returns along paths.

## Decisions

- **One split builder.** `split`, `split_with_diagnostics` and `cpcv_splits` share one
  function that marks each sample as test, purged and/or embargoed; the rest train. The
  alternative, a separate diagnostics pass beside the old `split`, could drift from it.
- **Purged and embargoed may overlap.** As in `59ac6fa`, `embargo_indices` is every non-test
  sample in an embargo window, including purged ones. The original embargo test depends on
  it: with 5-bar labels and a 3-sample embargo counted from the fold edge, every embargoed
  sample is also purged. A disjoint "embargo only" list would be empty there. Callers get
  that list as `embargo_indices` minus `purged_indices`.
- **`cpcv_paths` returns paths, not splits.** In `59ac6fa`, `cpcv_paths` returned the C(N, k)
  splits. AFML calls those splits and reserves "paths" for the φ[N, k] recombinations. The
  splits method is `cpcv_splits`. `cpcv_paths` returns path assignments, using the same
  first-come order as `backtesting_engine::run_cpcv`, so both modules number paths alike. The
  original test is kept on `cpcv_splits` (10 splits for N = 5, k = 2) and also checks 4 paths.
- **Purge per block of adjacent test folds.** `cpcv_splits` merges adjacent test folds into
  one block and applies `split`'s window rule (first start to latest end) to each block, so
  k = 1 reproduces `split` exactly.
- **Validation in `new`.** The original commit validated `pct_embargo` and reversed
  information sets. Both are kept, as typed errors.
- **`count_train_test_overlaps` returns `Result`.** An out-of-range index is an error, not a
  panic.

## Risks / Trade-offs

- [Callers passing `pct_embargo` ≥ 1 or negative now get an error] → nothing in the
  workspace does, and such values gave empty or unembargoed training sets.
- [C(N, k) splits each hold index vectors, so memory grows fast with N] → the count is
  computed with overflow checks (`TooManySplits`), but memory is the caller's choice of N.

## ADDED Requirements

### Requirement: Purged splits as numpy index arrays
`openquant.cross_validation.purged_kfold_splits(t0, t1, n_splits, pct_embargo)` SHALL return
one `(train_idx, test_idx)` pair of one-dimensional integer numpy arrays per fold, equal to
`PurgedKFold::split` on the same spans. `split_with_diagnostics` and `cpcv_splits` SHALL return
the folds of the Rust functions of the same names as dicts whose index lists are numpy arrays,
and `cpcv_paths(n_splits, n_test_splits)` SHALL return the Rust path assignments as an
`(n_paths, n_splits)` integer array.

#### Scenario: Usable as a scikit-learn cv argument
- **WHEN** the result of `purged_kfold_splits` is passed as `cv=` to `cross_val_score`
- **THEN** each fold is fitted on its train indices and scored on its test indices

#### Scenario: Docs example
- **WHEN** 40 hourly labels lasting 3 hours are split into 5 folds with `pct_embargo = 0.15`
- **THEN** the third fold tests 16-23 and trains on 0-9 and 30-39

#### Scenario: CPCV paths
- **WHEN** `cpcv_paths(6, 2)` is called
- **THEN** it has 5 rows, the first `[0, 0, 1, 2, 3, 4]` and the last `[4, 8, 11, 13, 14, 14]`

### Requirement: Label spans are required
Every function that purges SHALL take the label spans `t0` and `t1` as required arguments.
`None` for either SHALL raise `ValueError`; omitting them SHALL raise `TypeError`. Spans MAY be
datetimes (numpy, pandas, polars, `datetime`, ISO strings) or integers, and both columns SHALL
be of the same kind and length.

#### Scenario: Missing spans
- **WHEN** `purged_kfold_splits`, `cpcv_splits`, `run_cpcv` or any feature-importance function is called with `t0=None`
- **THEN** it raises `ValueError` naming `t0`

#### Scenario: Formats agree
- **WHEN** the same spans are given as datetime64, `datetime`, ISO strings or bar positions
- **THEN** the splits are identical

### Requirement: No leakage and the embargo is honoured
No training index returned by `purged_kfold_splits` or `cpcv_splits` SHALL have a label span
that intersects the span of any test index of the same split. `ceil(pct_embargo * n)` samples on
each side of each test block SHALL be excluded from training (the current two-sided behaviour,
issue #134).

#### Scenario: Random labels
- **WHEN** random variable-length labels are split for random N, k and embargo
- **THEN** no train/test pair of spans intersects

#### Scenario: Point labels
- **WHEN** labels do not overlap and `pct_embargo` > 0
- **THEN** training is every sample outside `[start - e, stop + e)` of the test fold, with `e = ceil(pct_embargo * n)`

### Requirement: CPCV backtest from precomputed returns
`openquant.backtesting_engine.run_cpcv` SHALL score per-split out-of-sample returns with the
Rust `run_cpcv`. Its arguments are `t0, t1, split_returns` and the keywords `n_groups,
test_groups, pct_embargo, mode_provenance, trials_count, safeguards`, where `split_returns[s]` is aligned to the test indices of
split `s` of `cross_validation.cpcv_splits`. It SHALL raise `ValueError` when the number of
arrays or an array's length does not match, or a safeguard is missing or empty.
`assemble_cpcv_paths` SHALL stitch per-split values into an `(n_paths, n_samples)` array.

#### Scenario: Path assignment
- **WHEN** 12 point labels in 6 groups are tested 2 at a time and split `s` returns `1000 * s + index`
- **THEN** each of the 5 paths has 12 observations and the AFML §12.4 mean

#### Scenario: Same test samples
- **WHEN** `run_cpcv` and `cpcv_splits` are given the same spans and parameters
- **THEN** every split tests the same indices, and `path_assignments` equals `cpcv_paths`

#### Scenario: Assembled paths
- **WHEN** per-split returns are stitched with `assemble_cpcv_paths`
- **THEN** each row's mean, sample standard deviation and t-statistic equal the engine's path statistics

### Requirement: Feature importance on purged folds
`mean_decrease_impurity` SHALL return the Rust MDI of per-tree importances.
`mda_from_probabilities` and `sfi_from_probabilities` SHALL rebuild the purged folds from the
spans and score the caller's out-of-sample probabilities with the Rust MDA and SFI.
`mean_decrease_accuracy` and `single_feature_importance` SHALL fit a copy of the given
estimator per purged fold in Python and score the result the same way.

#### Scenario: MDA hand-worked
- **WHEN** the `SignOfFirstColumn` fixture of the Rust reference tests is scored
- **THEN** f0's accuracy MDA is 0.5, its neg-log-loss MDA is (1 − ln 0.9 / ln 0.1) / 2, and f1's is 0

#### Scenario: SFI hand-worked
- **WHEN** the same fixture is scored by SFI with accuracy
- **THEN** f0 is 0.75 with standard error 0.25/√2 and f1 is 0.5 with 0.5/√2

### Requirement: Hyperparameter search helpers
`expand_param_grid` and `sample_param_sets` SHALL return the candidates the Rust `grid_search`
and `randomized_search` evaluate, in the same order. `classification_score` SHALL return the
Rust sample-weighted score. `purged_search` SHALL fit a model per candidate and purged fold in
Python, score each test fold with `classification_score` and the test weights, and pick the
best mean score, the later candidate winning a tie.

#### Scenario: Grid fixture
- **WHEN** the Rust grid-search fixture is searched with a Python port of its classifier
- **THEN** 6 candidates are scored and the best threshold is 0.7

#### Scenario: Seeded draws
- **WHEN** `sample_param_sets` is called twice with the same space and seed
- **THEN** the draws are equal, and equal to the candidates of the Rust `randomized_search`

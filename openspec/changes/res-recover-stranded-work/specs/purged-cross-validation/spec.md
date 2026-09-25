## ADDED Requirements

### Requirement: Purged split diagnostics
`PurgedKFold::split_with_diagnostics(n_samples)` SHALL return one `PurgedSplit` per fold of
`PurgedKFold::split(n_samples)`, in the same order and with the same train and test indices.
Each split's diagnostics SHALL report its position (`split_id`), its test set as sorted
half-open ranges (`test_ranges`), the non-test samples whose information set overlaps a test
block's window (`purged_indices`), the non-test samples inside an embargo window whether or not
also purged (`embargo_indices`), and the number of training samples whose information set
intersects any test sample's (`overlap_count_after_purge`).

#### Scenario: Diagnostics agree with split
- **WHEN** `split` and `split_with_diagnostics` are called on the same `PurgedKFold`
- **THEN** each diagnosed split has the train and test indices of the corresponding `split` fold

#### Scenario: Every sample is accounted for
- **WHEN** a split is diagnosed
- **THEN** its train indices, test indices and the union of purged and embargoed indices are disjoint and together are `0..n_samples`

#### Scenario: Purging leaves no overlap
- **WHEN** a split is diagnosed, whatever the label lengths and embargo
- **THEN** `overlap_count_after_purge` is 0

#### Scenario: Worked example
- **WHEN** 40 hourly labels each lasting 3 hours are split into 5 folds with `pct_embargo = 0.15`
- **THEN** the third fold tests 16-23, purges 13-15 and 24-26, embargoes 10-15 and 24-29, and trains on 0-9 and 30-39

#### Scenario: Embargo inside the purged zone
- **WHEN** the same labels are split with `pct_embargo = 0.07`
- **THEN** the third fold's embargoed indices equal its purged indices and its training set is the one without an embargo

#### Scenario: No embargo
- **WHEN** `pct_embargo` is 0
- **THEN** every split's `embargo_indices` is empty

### Requirement: Existing split behaviour is preserved
`PurgedKFold::split` SHALL return the same folds as before this change. The embargo SHALL
still remove `ceil(pct_embargo * n_samples)` samples on both sides of each test fold, counted
from the fold's edges.

#### Scenario: Regression against the previous implementation
- **WHEN** random variable-length labels are split with embargoes of 0, 0.01, 0.05, 0.15 and 0.4
- **THEN** every fold equals the fold the previous implementation produced

### Requirement: Combinatorial purged splits
`PurgedKFold::cpcv_splits(n_samples, n_test_splits)` SHALL return one `CpcvSplit` for each
way of choosing `n_test_splits` of the `n_splits` contiguous folds as the test set: C(N, k)
splits, which equals C(N, N − k), in lexicographic order of `test_fold_ids` (AFML §12.4).
Each run of adjacent test folds SHALL be purged and embargoed as a `split` fold is.

#### Scenario: Split count
- **WHEN** N = 6 and k = 2
- **THEN** 15 splits are returned, the first testing folds 0 and 1 and the last folds 4 and 5

#### Scenario: Adjacent test folds form one block
- **WHEN** a split tests two adjacent folds
- **THEN** its `test_ranges` holds a single range covering both

#### Scenario: One test fold
- **WHEN** k = 1
- **THEN** the splits equal those of `split_with_diagnostics`

#### Scenario: No leakage
- **WHEN** random variable-length labels are split for any valid N, k and embargo
- **THEN** no training label's information set intersects any test label's

### Requirement: Combinatorial backtest paths
`PurgedKFold::cpcv_paths(n_test_splits)` SHALL return φ[N, k] = k / N · C(N, N − k) paths
(AFML §12.4). Path `j` SHALL take, for every fold `g`, the `j`-th split in `split_id` order
that tests `g` (`split_for_fold[g]`). Each (fold, split testing it) pair SHALL be used by
exactly one path.

#### Scenario: Path count
- **WHEN** N = 6 and k = 2
- **THEN** 5 paths are returned

#### Scenario: Path assignment
- **WHEN** N = 6 and k = 2
- **THEN** path 0 is `[0, 0, 1, 2, 3, 4]` and path 4 is `[4, 8, 11, 13, 14, 14]`

#### Scenario: Paths cover every prediction once
- **WHEN** 2 ≤ N ≤ 8 and 1 ≤ k < N
- **THEN** every path has one split per fold, that split tests the fold, and the paths together use each (fold, split) pair exactly once

### Requirement: Naive k-fold baseline
`naive_kfold_splits(n_samples, n_splits)` SHALL return `n_splits` contiguous test folds, the
first `n_samples % n_splits` one sample longer, each trained on every other sample with no
purging or embargo.

#### Scenario: Contiguous complements
- **WHEN** 10 samples are split into 3 folds
- **THEN** the test folds are 0-3, 4-6 and 7-9, and each fold's train set is its complement

#### Scenario: The baseline leaks and purging does not
- **WHEN** 180 labels lasting 30 bars are split into 6 folds
- **THEN** some naive fold has a positive `count_train_test_overlaps`, and every purged fold has 0

### Requirement: Train/test overlap count
`count_train_test_overlaps(info_sets, train_indices, test_indices)` SHALL return the number of
training indices whose information set intersects, as a closed interval, the information set
of at least one test index.

#### Scenario: Counting
- **WHEN** sample `i` spans minutes `[i, i + 2]`, the test set is `{5}` and every other sample trains
- **THEN** the count is 4 (samples 3, 4, 6 and 7)

#### Scenario: Each training sample counts once
- **WHEN** one training sample overlaps three test samples
- **THEN** the count is 1

### Requirement: Invalid input is an error
The functions in this capability SHALL return a `CrossValidationError` for invalid input and
SHALL NOT panic.

#### Scenario: Test fold count out of range
- **WHEN** `n_test_splits` is 0 or at least `n_splits`
- **THEN** `cpcv_splits` and `cpcv_paths` return `InvalidTestSplits`

#### Scenario: Dataset length mismatch
- **WHEN** `n_samples` differs from the number of information sets
- **THEN** `split_with_diagnostics` and `cpcv_splits` return `DatasetLengthMismatch`

#### Scenario: Embargo out of range
- **WHEN** `pct_embargo` is negative, at least 1, NaN or infinite
- **THEN** `PurgedKFold::new` returns `InvalidEmbargo`

#### Scenario: Reversed information set
- **WHEN** an information set ends before it starts
- **THEN** `PurgedKFold::new` returns `InvalidInfoSet` with its index

#### Scenario: Index out of range
- **WHEN** a train or test index is not a position in `info_sets`
- **THEN** `count_train_test_overlaps` returns `IndexOutOfRange`

#### Scenario: Impossible naive split
- **WHEN** `n_splits` is below 2 or above `n_samples`
- **THEN** `naive_kfold_splits` returns `InvalidSplits`

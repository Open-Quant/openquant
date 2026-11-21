from __future__ import annotations

from math import ceil
from random import Random

import pytest
from openquant.feature_diagnostics import _purged_kfold_splits


def _reference(
    ends: list[int], n_splits: int, pct_embargo: float
) -> list[tuple[list[int], list[int]]]:
    n = len(ends)
    k, extra = divmod(n, n_splits)
    width = ceil(n * pct_embargo)
    start = 0
    folds = []
    for fold in range(n_splits):
        stop = start + k + (fold < extra)
        test = list(range(start, stop))
        # Enumerate closed row spans independently of the implementation's interval math.
        used = {row for i in test for row in range(i, ends[i] + 1)}
        embargo = set(range(max(0, start - width), start))
        embargo.update(range(max(used) + 1, min(n, max(used) + 1 + width)))
        train = [
            i
            for i, end in enumerate(ends)
            if i not in embargo and used.isdisjoint(range(i, end + 1))
        ]
        if not train:
            raise ValueError("empty fold")
        folds.append((train, test))
        start = stop
    return folds


def test_purge_matches_closed_row_oracle():
    rng = Random(147)
    valid = empty = 0
    for _ in range(280):
        n = rng.randint(4, 45)
        n_splits = rng.randint(2, n)
        pct = rng.choice((0.0, 0.01, 0.1, 0.2))
        max_horizon = rng.choice((0, 1, 4, n))
        ends = [min(n - 1, i + rng.randrange(max_horizon + 1)) for i in range(n)]
        intervals = list(enumerate(ends))
        try:
            expected = _reference(ends, n_splits, pct)
        except ValueError:
            with pytest.raises(ValueError, match="empty train/test fold"):
                _purged_kfold_splits(intervals, n_splits, pct)
            empty += 1
        else:
            assert _purged_kfold_splits(intervals, n_splits, pct) == expected
            valid += 1
    assert valid > 100 and empty > 30


def test_purge_closed_touch_and_embargo_after_label_resolution():
    ends = [0, 2, 2, 5, 4, 5, 6, 7, 8, 9, 10, 11]
    # Middle fold starts at 3. Index 2 is excluded by pre-embargo. The last test
    # label ends at 5; post-embargo excludes 6, not the test row at 5.
    assert _purged_kfold_splits(list(enumerate(ends)), 4, 0.05)[1] == (
        [0, 1, 7, 8, 9, 10, 11],
        [3, 4, 5],
    )
    ends[1] = 3
    # Index 1's label touches the test interval at exactly 3.
    assert 1 not in _purged_kfold_splits(list(enumerate(ends)), 4, 0.0)[1][0]


def test_purge_comparison_work_is_linear_per_fold():
    class CountedIndex(int):
        comparisons = 0

        def __lt__(self, other: int) -> bool:
            CountedIndex.comparisons += 1
            return super().__lt__(other)

        def __gt__(self, other: int) -> bool:
            CountedIndex.comparisons += 1
            return super().__gt__(other)

        def __le__(self, other: int) -> bool:
            CountedIndex.comparisons += 1
            return super().__le__(other)

    n, folds = 1200, 4
    intervals = [(CountedIndex(i), CountedIndex(i)) for i in range(n)]
    assert len(_purged_kfold_splits(intervals, folds, 0.0)) == folds
    assert CountedIndex.comparisons <= 4 * n * folds

"""Cross-language CV contracts against an exact closed-interval reference."""

import numpy as np
import pytest
from openquant import cross_validation as cv


@pytest.mark.parametrize("origin", [-(2**63) + 1, -(10**18), -10000, 0, 10**18, 2**63 - 10000])
def test_native_overlap_and_purge_match_exact_boundary_reference(origin):
    rng = np.random.default_rng(991211)
    for _ in range(30):
        n = int(rng.integers(2, 50))
        starts = sorted(map(int, rng.integers(0, 4000, n)))
        ends = [start + int(rng.integers(0, 250)) for start in starts]
        t0 = np.array([origin + start for start in starts], dtype=np.int64)
        t1 = np.array([origin + end for end in ends], dtype=np.int64)
        # Duplicated train entries count separately; empty selections count zero.
        train = list(map(int, rng.integers(0, n, int(rng.integers(0, n)))))
        test = list(map(int, rng.integers(0, n, int(rng.integers(0, n)))))
        expected = sum(
            any(starts[i] <= ends[j] and starts[j] <= ends[i] for j in test) for i in train
        )
        dates = (t0.view("datetime64[ns]"), t1.view("datetime64[ns]"))
        assert cv.count_train_test_overlaps(t0, t1, train, test) == expected
        assert cv.count_train_test_overlaps(*dates, train, test) == expected
        ints = cv.purged_kfold_splits(t0, t1, min(n, 3), 0.07)
        times = cv.purged_kfold_splits(*dates, min(n, 3), 0.07)
        for (it, ie), (dt, de) in zip(ints, times, strict=True):
            np.testing.assert_array_equal(it, dt)
            np.testing.assert_array_equal(ie, de)
            assert cv.count_train_test_overlaps(t0, t1, it, ie) == 0

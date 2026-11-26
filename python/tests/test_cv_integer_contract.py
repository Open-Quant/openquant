"""Reject coercive timestamp/index conversions without losing signed-int64 bits."""

import numpy as np
import pytest
from openquant import cross_validation as cv


def test_mixed_supported_integers_preserve_label_spans():
    limit = np.iinfo(np.int64).max
    start = [np.int64(limit - 2), np.uint64(limit - 1)]
    end = [np.uint64(limit - 1), limit]
    assert cv._label_spans(start, end) == ([limit - 2, limit - 1], [limit - 1, limit])


def test_mixed_supported_integers_preserve_indices():
    limit = np.iinfo(np.int64).max
    assert cv._index_list([np.int64(0), np.uint64(limit)], "train_indices") == [0, limit]


@pytest.mark.parametrize("which", ["integer label", "index"])
def test_mixed_integer_overflow_is_rejected_as_overflow(which):
    values = [np.int64(1), np.uint64(2**63)]
    with pytest.raises(ValueError, match="int64 without overflow"):
        if which == "index":
            cv._index_list(values, "train_indices")
        else:
            cv._label_spans(values, [0, 0])


def test_mixed_float_indices_are_not_accepted_as_integers():
    with pytest.raises(ValueError, match="array of integers"):
        cv._index_list([np.uint64(1), 2.0], "train_indices")


def test_public_split_preserves_large_integer_label_geometry():
    # A float64 coercion would collapse consecutive differences at this scale.
    limit = np.iinfo(np.int64).max
    start = [np.uint64(limit - 6 + i) if i % 2 else limit - 6 + i for i in range(6)]
    end = [np.int64(limit - 5 + i) if i % 2 else np.uint64(limit - 5 + i) for i in range(6)]
    actual = cv.purged_kfold_splits(start, end, 3, 0.0)
    expected = cv.purged_kfold_splits(list(range(6)), list(range(1, 7)), 3, 0.0)
    for (a_train, a_test), (e_train, e_test) in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(a_train, e_train)
        np.testing.assert_array_equal(a_test, e_test)


def test_public_overlap_counter_accepts_mixed_integer_index_widths():
    assert cv.count_train_test_overlaps([0, 1, 2], [0, 2, 2], [np.int64(0), np.uint64(1)], [2]) == 1


@pytest.mark.parametrize("bad", [True, np.bool_(True), 2.9, 2.0, "2"])
@pytest.mark.parametrize(
    "parameter, call",
    [
        ("n_splits", lambda x: cv.purged_kfold_splits(range(4), range(4), x)),
        ("n_splits", lambda x: cv.split_with_diagnostics(range(4), range(4), x)),
        ("n_splits", lambda x: cv.cpcv_splits(range(4), range(4), x, 1)),
        ("n_test_splits", lambda x: cv.cpcv_splits(range(4), range(4), 4, x)),
        ("n_splits", lambda x: cv.cpcv_paths(x, 1)),
        ("n_test_splits", lambda x: cv.cpcv_paths(4, x)),
        ("n_splits", lambda x: cv.walk_forward_splits(range(4), range(4), x)),
        (
            "min_train_folds",
            lambda x: cv.walk_forward_splits(range(8), range(8), 4, min_train_folds=x),
        ),
        ("n_samples", lambda x: cv.naive_kfold_splits(x, 2)),
        ("n_splits", lambda x: cv.naive_kfold_splits(4, x)),
        (
            "n_null",
            lambda x: cv.null_score_distribution(
                lambda data, splits: 1.0, [None], lambda rng: None, n_null=x
            ),
        ),
    ],
)
def test_public_counts_reject_lossy_coercion(parameter, call, bad):
    with pytest.raises(ValueError, match=rf"{parameter} must be an integer"):
        call(bad)


def test_numpy_integer_counts_work_against_known_folds():
    splits = cv.purged_kfold_splits(range(4), range(4), np.uint64(2))
    assert [(train.tolist(), test.tolist()) for train, test in splits] == [
        ([2, 3], [0, 1]),
        ([0, 1], [2, 3]),
    ]
    assert cv.cpcv_paths(np.int32(4), np.uint64(2)).shape == (3, 4)


@pytest.mark.parametrize("masked_arg", ["t0", "t1", "train_indices", "test_indices"])
@pytest.mark.parametrize("hidden", [0, 10_000])
def test_public_overlap_rejects_masked_labels_and_indices(masked_arg, hidden):
    spans = [0, 1, 2, 3]
    args = {"t0": spans, "t1": spans, "train_indices": [0, 1], "test_indices": [2, 3]}
    values = list(args[masked_arg])
    values[-1] = hidden
    args[masked_arg] = np.ma.array(values, mask=[False] * (len(values) - 1) + [True])
    with pytest.raises(ValueError, match=rf"{masked_arg}.*masked"):
        cv.count_train_test_overlaps(**args)


def test_unmasked_wrapper_matches_materialized_input():
    values = np.ma.array([0, 1, 2, 3], mask=[False] * 4)
    actual = cv.purged_kfold_splits(values, values, 2)
    assert [(train.tolist(), test.tolist()) for train, test in actual] == [
        ([2, 3], [0, 1]),
        ([0, 1], [2, 3]),
    ]


def test_fold_count_rejects_masked_scalar_integer():
    with pytest.raises(ValueError, match="n_splits.*masked"):
        cv.purged_kfold_splits(range(4), range(4), np.ma.array(2, mask=True))

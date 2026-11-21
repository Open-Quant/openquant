"""Python-only boundary contracts for CV and out-of-sample path assembly."""

from copy import deepcopy
from itertools import combinations

import numpy as np
import pytest
from openquant import backtesting_engine as bt
from openquant import cross_validation as cv


@pytest.mark.parametrize("value", [2**63, 2**64 - 1])
def test_overlap_indices_reject_unsigned_overflow(value):
    with pytest.raises(ValueError, match="int64"):
        cv._index_list(np.array([value], dtype=np.uint64), "train_indices")


def test_overlap_indices_preserve_max_signed_integer():
    assert cv._index_list(np.array([2**63 - 1], dtype=np.uint64), "train_indices") == [2**63 - 1]


@pytest.mark.parametrize("values", [[1e308, 1e308], [-1e308, -1e308], [1e308, 1e308, -1e308]])
def test_bootstrap_recovers_finite_overflowing_means(values):
    from fractions import Fraction

    mean = sum(map(Fraction, values)) / len(values)
    expected = np.array([float(Fraction(value) - mean) for value in values])
    rng = np.random.default_rng(81)
    indices = rng.integers(0, len(values), len(values))
    with np.errstate(all="raise"):
        result = cv.bootstrap_returns(values)(np.random.default_rng(81))
    np.testing.assert_array_equal(result, expected[indices])


def test_bootstrap_overflow_fallback_is_columnwise():
    values = np.array([[1e308, 0.01], [1e308, 0.02], [-1e308, 0.03]])
    rng = np.random.default_rng(81)
    indices = rng.integers(0, 3, 3)
    with np.errstate(all="raise"):
        result = cv.bootstrap_returns(values)(np.random.default_rng(81))
    np.testing.assert_array_equal(result[:, 1], (values[:, 1] - values[:, 1].mean())[indices])
    assert np.isfinite(result).all()


def test_bootstrap_unrepresentable_centering_rejects_before_drawing():
    with pytest.raises(ValueError, match="representable"):
        cv.bootstrap_returns([1.79e308, -1.79e308, -1.79e308])


@pytest.mark.parametrize("demean", [False, True])
def test_bootstrap_captures_input_snapshot(demean):
    values = np.array([0.01, 0.02, 0.03])
    draw = cv.bootstrap_returns(values, demean=demean)
    expected = draw(np.random.default_rng(1))
    values[:] = 42.0
    np.testing.assert_array_equal(draw(np.random.default_rng(1)), expected)


def _small_cpcv():
    # N=3, k=2, two samples in each fold, independent of native splitting.
    folds = [[0, 1], [2, 3], [4, 5]]
    splits = []
    for s, groups in enumerate(combinations(range(3), 2)):
        splits.append(
            {
                "split_id": s,
                "test_indices": [i for g in groups for i in folds[g]],
                "test_fold_ids": groups,
            }
        )
    paths = [[0, 0, 1], [1, 2, 2]]
    values = [[100 * s + i for i in split["test_indices"]] for s, split in enumerate(splits)]
    return values, splits, paths


def test_cpcv_python_assembly_matches_explicit_reference():
    values, splits, paths = _small_cpcv()
    actual = bt.assemble_cpcv_paths(values, splits, paths)
    expected = [[100 * paths[p][i // 2] + i for i in range(6)] for p in range(2)]
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    "bad_paths",
    [[[0.9, 0, 1], [1, 2, 2]], [[True, 0, 1], [1, 2, 2]], [[0, 0, 1], [1, 2, 2**64 - 1]]],
)
def test_cpcv_rejects_lossy_path_indices(bad_paths):
    values, splits, _ = _small_cpcv()
    with pytest.raises(ValueError):
        bt.assemble_cpcv_paths(values, splits, bad_paths)


@pytest.mark.parametrize("bad_test", [[1, 0, 2, 3], [0, 0, 2, 3], [-1, 1, 2, 3], [0.5, 1, 2, 3]])
def test_cpcv_rejects_unsorted_duplicate_or_invalid_sample_indices(bad_test):
    values, splits, paths = _small_cpcv()
    splits[0]["test_indices"] = bad_test
    with pytest.raises(ValueError):
        bt.assemble_cpcv_paths(values, splits, paths)


def test_cpcv_rejects_overlapping_folds_instead_of_returning_uninitialized_cells():
    splits = [
        {"split_id": 0, "test_indices": [0, 1], "test_fold_ids": [0]},
        {"split_id": 1, "test_indices": [1, 2], "test_fold_ids": [1]},
    ]
    with pytest.raises(ValueError, match="partition"):
        bt.assemble_cpcv_paths([[10, 11], [20, 21]], splits, [[0, 1]])


def test_cpcv_rejects_empty_fold_intersection():
    values, splits, paths = _small_cpcv()
    splits[1]["test_indices"] = [2, 3, 4, 5]
    with pytest.raises(ValueError, match="fold"):
        bt.assemble_cpcv_paths(values, splits, paths)


def test_cpcv_does_not_mutate_input():
    values, splits, paths = _small_cpcv()
    snapshot = deepcopy((values, splits, paths))
    bt.assemble_cpcv_paths(values, splits, paths)
    assert (values, splits, paths) == snapshot


@pytest.mark.parametrize(
    "spans",
    [
        ["2500-01-01", "2020-01-01T00:00:00.000000001"],
        [np.datetime64("2500-01-01", "D"), np.datetime64("2020-01-01", "ns")],
    ],
)
def test_mixed_precision_dates_reject_overflow_before_inference(spans):
    with pytest.raises(ValueError, match="nanosecond"):
        cv._as_int64(spans, "t0")


@pytest.mark.parametrize(
    "spans",
    [
        ["2020-01-01T00:00:00.000000000001"],
        ["1969-12-31T23:59:59.999999999999"],
        ["1677-09-21T00:12:43.145224193"],
    ],
)
def test_string_nanoseconds_are_floored_without_intermediate_fine_unit_wrap(spans):
    actual, is_time = cv._as_int64(spans, "t0")
    assert is_time
    np.testing.assert_array_equal(actual, np.array(spans, dtype="datetime64[ns]").view("int64"))


@pytest.mark.parametrize("unit,divisor", [("ps", 1000), ("fs", 1000000), ("as", 1000000000)])
def test_fine_datetime_negative_extremes_use_exact_floor(unit, divisor):
    ticks = np.array([-(2**63) + 1, -1, 0, 2**63 - 1], dtype="int64")
    actual, _ = cv._as_int64(ticks.view(f"datetime64[{unit}]"), "t0")
    np.testing.assert_array_equal(actual, [int(tick) // divisor for tick in ticks])


@pytest.mark.parametrize(
    "unit,multiplier", [("2ns", 2), ("us", 1000), ("s", 10**9), ("D", 86400 * 10**9)]
)
def test_coarse_negative_near_boundary_uses_exact_multiplication(unit, multiplier):
    tick = -((2**63 - 1) // multiplier)
    actual, _ = cv._as_int64(np.array([tick], dtype="int64").view(f"datetime64[{unit}]"), "t0")
    assert actual.tolist() == [tick * multiplier]


@pytest.mark.parametrize(
    "unit,scale,divisor", [("2ps", 2, 1000), ("ps", 1, 1000), ("us", 1000, 1), ("ns", 1, 1)]
)
def test_datetime_counts_respect_byte_order(unit, scale, divisor):
    ticks = [-1001, -1, 0, 1, 1001]
    data = np.array(ticks, dtype=">i8").view(f">M8[{unit}]")
    actual, _ = cv._as_int64(data, "t0")
    np.testing.assert_array_equal(actual, [tick * scale // divisor for tick in ticks])


@pytest.mark.parametrize("values", [[[True, 1]], [True, 1], np.empty((2, 0), dtype=int)])
def test_overlap_indices_reject_boolean_or_multidimensional_input(values):
    with pytest.raises(ValueError, match="one-dimensional array of integers"):
        cv._index_list(values, "train_indices")

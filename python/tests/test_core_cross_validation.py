"""openquant.cross_validation: purged k-fold and CPCV splits as numpy index arrays.

Fixtures and expected values are those of crates/openquant/tests/cross_validation.rs; each
test names the Rust test it reproduces.
"""

from datetime import datetime, timedelta
from math import ceil

import numpy as np
import pytest

from openquant import cross_validation as cv


def make_spans(periods, step_minutes, horizon_minutes, start="2019-01-01T00:00:00"):
    """Rust `make_spans`: `periods` labels every `step_minutes`, each `horizon_minutes` long."""
    t0 = np.datetime64(start, "ns") + np.arange(periods) * np.timedelta64(step_minutes, "m")
    return t0, t0 + np.timedelta64(horizon_minutes, "m")


def docs_spans():
    """The docs-page example: 40 hourly labels from 2024-01-02 09:00, each lasting 3 hours."""
    t0 = np.datetime64("2024-01-02T09:00", "ns") + np.arange(40) * np.timedelta64(1, "h")
    return t0, t0 + np.timedelta64(3, "h")


def random_spans(rng, n):
    """Rust `random_spans`: increasing starts, label lengths 0-39 minutes."""
    starts = np.cumsum(rng.integers(1, 10, size=n))
    return starts, starts + rng.integers(0, 40, size=n)


def overlapping_pairs(t0, t1, train, test):
    t0, t1 = np.asarray(t0), np.asarray(t1)
    a0, a1 = t0[train][:, None], t1[train][:, None]
    b0, b1 = t0[test][None, :], t1[test][None, :]
    return int(((a0 <= b1) & (b0 <= a1)).sum())


def test_purged_kfold_splits_return_numpy_index_arrays():
    t0, t1 = docs_spans()
    splits = cv.purged_kfold_splits(t0, t1, n_splits=5, pct_embargo=0.0)
    assert len(splits) == 5
    for train, test in splits:
        assert isinstance(train, np.ndarray) and isinstance(test, np.ndarray)
        assert train.dtype == np.intp and test.dtype == np.intp
    assert np.concatenate([test for _, test in splits]).tolist() == list(range(40))


def test_docs_page_example_values():
    # Rust: test_docs_page_example_values
    t0, t1 = docs_spans()

    def train_of(pct):
        return cv.purged_kfold_splits(t0, t1, 5, pct)[2][0].tolist()

    purged_only = list(range(0, 13)) + list(range(27, 40))
    assert cv.purged_kfold_splits(t0, t1, 5, 0.0)[2][1].tolist() == list(range(16, 24))
    assert train_of(0.0) == purged_only
    assert train_of(0.07) == purged_only
    assert train_of(0.15) == list(range(0, 10)) + list(range(30, 40))


def test_split_with_diagnostics_docs_example():
    # Rust: test_split_with_diagnostics_docs_example
    t0, t1 = docs_spans()
    purged = [13, 14, 15, 24, 25, 26]

    no_embargo = cv.split_with_diagnostics(t0, t1, 5, 0.0)[2]
    assert no_embargo["split_id"] == 2
    assert no_embargo["test_ranges"] == [(16, 24)]
    assert no_embargo["purged_indices"].tolist() == purged
    assert no_embargo["embargo_indices"].tolist() == []

    narrow = cv.split_with_diagnostics(t0, t1, 5, 0.07)[2]
    assert narrow["embargo_indices"].tolist() == purged
    assert narrow["train_indices"].tolist() == no_embargo["train_indices"].tolist()

    wide = cv.split_with_diagnostics(t0, t1, 5, 0.15)[2]
    assert wide["purged_indices"].tolist() == purged
    assert wide["embargo_indices"].tolist() == list(range(10, 16)) + list(range(24, 30))
    assert wide["train_indices"].tolist() == list(range(0, 10)) + list(range(30, 40))
    assert wide["overlap_count_after_purge"] == 0

    assert len(cv.cpcv_splits(t0, t1, 5, 2, 0.15)) == 10
    assert cv.cpcv_paths(5, 2).shape == (4, 5)
    train, test = cv.naive_kfold_splits(40, 5)[2]
    assert cv.count_train_test_overlaps(t0, t1, train, test) == 6


def test_purged_kfold_purges_labels_overlapping_the_first_test_sample():
    # Rust: test_purged_kfold_purges_labels_overlapping_the_first_test_sample
    t0 = np.datetime64("2019-01-01", "ns") + np.arange(12) * np.timedelta64(1, "D")
    t1 = t0 + np.timedelta64(3, "D")
    expected_train = [[7, 8, 9, 10, 11], [0, 11], [0, 1, 2, 3, 4]]
    for fold, (train, test) in enumerate(cv.purged_kfold_splits(t0, t1, 3, 0.0)):
        assert test.tolist() == list(range(fold * 4, fold * 4 + 4))
        assert train.tolist() == expected_train[fold]


def test_cpcv_six_groups_two_test_groups():
    # Rust: test_cpcv_six_groups_two_test_groups
    t0, t1 = make_spans(60, 1, 0)
    splits = cv.cpcv_splits(t0, t1, 6, 2, 0.0)
    assert len(splits) == 15
    assert splits[0]["test_fold_ids"] == (0, 1)
    assert splits[14]["test_fold_ids"] == (4, 5)
    assert splits[0]["test_ranges"] == [(0, 20)]
    assert splits[1]["test_ranges"] == [(0, 10), (20, 30)]
    assert [s["split_id"] for s in splits] == list(range(15))

    paths = cv.cpcv_paths(6, 2)
    assert paths.shape == (5, 6)
    assert paths[0].tolist() == [0, 0, 1, 2, 3, 4]
    assert paths[4].tolist() == [4, 8, 11, 13, 14, 14]


def test_cpcv_split_and_path_counts_match_afml():
    # Rust: test_cpcv_split_and_path_counts_match_afml
    from math import comb

    t0, t1 = make_spans(48, 1, 3)
    for n in range(2, 9):
        for k in range(1, n):
            splits = cv.cpcv_splits(t0, t1, n, k, 0.0)
            paths = cv.cpcv_paths(n, k)
            assert len(splits) == comb(n, n - k)
            assert paths.shape == (comb(n - 1, k - 1), n)
            used = set()
            for row in paths:
                for fold, split_id in enumerate(row.tolist()):
                    assert fold in splits[split_id]["test_fold_ids"]
                    assert (fold, split_id) not in used
                    used.add((fold, split_id))
            assert len(used) == k * comb(n, k)


def test_cpcv_with_one_test_fold_is_purged_kfold():
    # Rust: test_cpcv_with_one_test_fold_is_purged_kfold
    t0, t1 = make_spans(37, 2, 7)
    cpcv = cv.cpcv_splits(t0, t1, 5, 1, 0.05)
    kfold = cv.split_with_diagnostics(t0, t1, 5, 0.05)
    assert len(cpcv) == len(kfold) == 5
    for c, k in zip(cpcv, kfold):
        assert c["test_fold_ids"] == (c["split_id"],)
        for key in ("train_indices", "test_indices", "purged_indices", "embargo_indices"):
            assert c[key].tolist() == k[key].tolist()
        assert c["test_ranges"] == k["test_ranges"]
    assert cv.cpcv_paths(5, 1).tolist() == [[0, 1, 2, 3, 4]]


def test_no_train_index_overlaps_a_test_label_span():
    # Rust: test_purged_kfold_no_train_label_overlaps_any_test_label and
    # test_cpcv_no_train_label_overlaps_any_test_label, with numpy's generator.
    rng = np.random.default_rng(7)
    for _ in range(120):
        n = int(rng.integers(8, 60))
        n_splits = int(rng.integers(3, min(n, 7) + 1))
        k = int(rng.integers(1, n_splits))
        pct = float(rng.choice([0.0, 0.02, 0.1]))
        t0, t1 = random_spans(rng, n)
        for train, test in cv.purged_kfold_splits(t0, t1, n_splits, pct):
            assert overlapping_pairs(t0, t1, train, test) == 0
            assert cv.count_train_test_overlaps(t0, t1, train, test) == 0
        for split in cv.cpcv_splits(t0, t1, n_splits, k, pct):
            train, test = split["train_indices"], split["test_indices"]
            assert overlapping_pairs(t0, t1, train, test) == 0
            removed = set(split["purged_indices"].tolist()) | set(split["embargo_indices"].tolist())
            assert sorted(set(train) | set(test) | removed) == list(range(n))


@pytest.mark.parametrize("pct_embargo", [0.01, 0.05, 0.1, 0.2])
def test_embargo_is_honoured(pct_embargo):
    # Point labels cannot overlap, so everything missing from training beyond the test fold is
    # the embargo: ceil(pct * n) samples on each side of the fold, counted from its edges.
    # Two-sided is the library's current behaviour (issue #134), and stricter than AFML.
    n = 100
    t = np.arange(n)
    width = ceil(pct_embargo * n)
    for train, test in cv.purged_kfold_splits(t, t, 4, pct_embargo):
        start, stop = int(test[0]), int(test[-1]) + 1
        blocked = set(range(max(0, start - width), min(n, stop + width)))
        assert set(train.tolist()) == set(range(n)) - blocked
    for split in cv.cpcv_splits(t, t, 5, 2, pct_embargo):
        train = set(split["train_indices"].tolist())
        for start, stop in split["test_ranges"]:
            assert not train & set(range(max(0, start - width), min(n, stop + width)))


def test_naive_kfold_leaks_but_purged_kfold_does_not():
    # Rust: test_naive_kfold_leaks_but_purged_kfold_does_not
    t0, t1 = make_spans(180, 1, 30)
    naive = cv.naive_kfold_splits(180, 6)
    assert any(cv.count_train_test_overlaps(t0, t1, tr, te) > 0 for tr, te in naive)
    for split in cv.split_with_diagnostics(t0, t1, 6, 0.02):
        tr, te = split["train_indices"], split["test_indices"]
        assert cv.count_train_test_overlaps(t0, t1, tr, te) == 0
        assert split["overlap_count_after_purge"] == 0


def test_count_train_test_overlaps_counts_training_samples():
    # Rust: test_count_train_test_overlaps_counts_training_samples
    t0, t1 = make_spans(10, 1, 2)
    train = [i for i in range(10) if i != 5]
    assert cv.count_train_test_overlaps(t0, t1, train, [5]) == 4
    assert cv.count_train_test_overlaps(t0, t1, [4], [3, 5, 6]) == 1
    assert cv.count_train_test_overlaps(t0, t1, [], [5]) == 0


def test_naive_kfold_splits_are_contiguous_complements():
    # Rust: test_naive_kfold_splits_are_contiguous_complements
    splits = cv.naive_kfold_splits(10, 3)
    assert [te.tolist() for _, te in splits] == [[0, 1, 2, 3], [4, 5, 6], [7, 8, 9]]
    for train, test in splits:
        assert sorted(train.tolist() + test.tolist()) == list(range(10))


def test_span_formats_give_identical_splits():
    t0, t1 = docs_spans()
    want = [(a.tolist(), b.tolist()) for a, b in cv.purged_kfold_splits(t0, t1, 5, 0.15)]
    py0 = [datetime(2024, 1, 2, 9) + timedelta(hours=i) for i in range(40)]
    py1 = [d + timedelta(hours=3) for d in py0]
    as_str = ([str(d) for d in py0], [str(d) for d in py1])
    positions = (np.arange(40), np.arange(40) + 3)
    for a, b in [(py0, py1), as_str, positions, (list(t0), list(t1))]:
        got = [(x.tolist(), y.tolist()) for x, y in cv.purged_kfold_splits(a, b, 5, 0.15)]
        assert got == want


def test_label_spans_are_required():
    t0, t1 = docs_spans()
    with pytest.raises(TypeError):
        cv.purged_kfold_splits(t0, n_splits=5)
    with pytest.raises(ValueError, match="t1 is required"):
        cv.purged_kfold_splits(t0, None, 5)
    with pytest.raises(ValueError, match="t0 is required"):
        cv.cpcv_splits(None, t1, 5, 2)


def test_invalid_input_raises_value_error():
    # Rust: test_new_split_apis_reject_invalid_input
    t0, t1 = make_spans(10, 1, 2)
    for k in (0, 4, 5):
        with pytest.raises(ValueError, match="n_test_splits"):
            cv.cpcv_splits(t0, t1, 4, k)
        with pytest.raises(ValueError, match="n_test_splits"):
            cv.cpcv_paths(4, k)
    for pct in (-0.01, 1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="pct_embargo"):
            cv.purged_kfold_splits(t0, t1, 4, pct)
    reversed_t0, reversed_t1 = t0.copy(), t1.copy()
    reversed_t0[3], reversed_t1[3] = t1[3], t0[3]
    with pytest.raises(ValueError, match=r"samples_info_sets\[3\] ends before it starts"):
        cv.purged_kfold_splits(reversed_t0, reversed_t1, 4)
    with pytest.raises(ValueError, match="length mismatch"):
        cv.purged_kfold_splits(t0, t1[:-1], 4)
    with pytest.raises(ValueError, match="both be datetimes or both be integers"):
        cv.purged_kfold_splits(t0, np.arange(10), 4)
    with pytest.raises(ValueError, match="NaT"):
        cv.purged_kfold_splits(t0, np.where(np.arange(10) == 2, np.datetime64("NaT", "ns"), t1), 4)
    with pytest.raises(ValueError, match="out of range"):
        cv.count_train_test_overlaps(t0, t1, [0, 10], [5])
    with pytest.raises(ValueError, match="non-negative"):
        cv.count_train_test_overlaps(t0, t1, [-1], [5])
    for n, k in ((0, 2), (5, 1), (5, 6)):
        with pytest.raises(ValueError, match="n_splits"):
            cv.naive_kfold_splits(n, k)
    with pytest.raises(ValueError, match="n_splits"):
        cv.purged_kfold_splits(t0, t1, 11)


def test_sklearn_cross_val_score_accepts_purged_splits():
    # scikit-learn is not a dependency of openquant; this runs where it is installed.
    sklearn_model_selection = pytest.importorskip("sklearn.model_selection")
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(0)
    n = 200
    X = rng.normal(size=(n, 3))
    y = (X[:, 0] + 0.5 * rng.normal(size=n) > 0).astype(int)
    t0 = np.arange(n)
    splits = cv.purged_kfold_splits(t0, t0 + 5, n_splits=5, pct_embargo=0.01)
    scores = sklearn_model_selection.cross_val_score(
        LogisticRegression(), X, y, cv=splits, scoring="accuracy"
    )
    assert scores.shape == (5,)
    assert scores.mean() > 0.7
    # GridSearchCV takes the same list.
    search = sklearn_model_selection.GridSearchCV(
        LogisticRegression(), {"C": [0.1, 1.0]}, cv=splits, scoring="neg_log_loss"
    ).fit(X, y)
    assert search.best_params_["C"] in (0.1, 1.0)

"""Purged k-fold, walk-forward and combinatorial purged cross-validation (AFML chapters 7, 12).

Every splitter returns indices, never fits a model: pass the splits to any estimator, for
example ``sklearn.model_selection.cross_val_score(model, X, y, cv=splits)``. The splitting,
purging and embargo are the Rust ``openquant::cross_validation`` code.

**Features with memory.** Purged k-fold and CPCV fit the model for a middle fold on samples
*after* it as well. Purging and the embargo remove overlapping labels, not that: a feature that
carries the price level lets the model learn where the path went after the test fold, and it
scores above chance on random walks (issue #217). Compare such features with
:func:`walk_forward_splits`, and against :func:`null_score_distribution`, a no-signal null run
through the same splits.

Label spans are required. ``t0[i]`` is when sample ``i``'s label starts (usually the event
time) and ``t1[i]`` when it is resolved (the triple-barrier touch). They may be numpy
``datetime64`` arrays, pandas or polars datetime columns, lists of ``datetime`` or ISO strings,
or plain integers such as bar positions; samples must be in time order.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from . import _core

_cv = _core.cross_validation

__all__ = [
    "purged_kfold_splits",
    "split_with_diagnostics",
    "cpcv_splits",
    "cpcv_paths",
    "walk_forward_splits",
    "walk_forward_split_with_diagnostics",
    "null_score_distribution",
    "null_p_value",
    "bootstrap_returns",
    "naive_kfold_splits",
    "count_train_test_overlaps",
]

_INDEX_KEYS = ("train_indices", "test_indices", "purged_indices", "embargo_indices")


def _as_int64(values: Any, name: str) -> tuple[np.ndarray, bool]:
    """One span column as int64, and whether it held datetimes.

    Datetimes become nanoseconds since the epoch; integers pass through unchanged.
    """
    if values is None:
        raise ValueError(
            f"{name} is required: purging needs every label's span (t0, t1). Without it, "
            "training labels that overlap the test fold cannot be found."
        )
    arr = np.asarray(values)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {arr.shape}")
    if arr.dtype.kind in "OUS":
        try:
            arr = np.asarray(values, dtype="datetime64[ns]")
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{name} must hold datetimes, ISO timestamp strings or integers: {exc}"
            ) from None
    if arr.dtype.kind == "M":
        arr = arr.astype("datetime64[ns]")
        if np.isnat(arr).any():
            raise ValueError(f"{name} contains NaT")
        return arr.view(np.int64), True
    if arr.dtype.kind in "iu" or arr.size == 0:
        return arr.astype(np.int64), False
    raise ValueError(f"{name} must hold datetimes or integers, got dtype {arr.dtype}")


def _label_spans(t0: Any, t1: Any) -> tuple[list[int], list[int]]:
    """Both span columns as int64 lists of equal length and the same kind."""
    s0, t0_is_time = _as_int64(t0, "t0")
    s1, t1_is_time = _as_int64(t1, "t1")
    if len(s0) and len(s1) and t0_is_time != t1_is_time:
        raise ValueError("t0 and t1 must both be datetimes or both be integers")
    if len(s0) != len(s1):
        raise ValueError(f"t0/t1 length mismatch: {len(s0)} vs {len(s1)}")
    return s0.tolist(), s1.tolist()


def _index_list(values: Sequence[int], name: str) -> list[int]:
    arr = np.asarray(values)
    if arr.size == 0:
        return []
    if arr.ndim != 1 or arr.dtype.kind not in "iu":
        raise ValueError(f"{name} must be a one-dimensional array of integers")
    if (arr < 0).any():
        raise ValueError(f"{name} must be non-negative")
    return arr.astype(np.int64).tolist()


def _indices(values: Sequence[int]) -> np.ndarray:
    return np.asarray(values, dtype=np.intp)


def _split_dict(d: dict) -> dict:
    out = dict(d)
    for key in _INDEX_KEYS:
        out[key] = _indices(out[key])
    out["test_ranges"] = [tuple(r) for r in out["test_ranges"]]
    if "test_fold_ids" in out:
        out["test_fold_ids"] = tuple(out["test_fold_ids"])
    return out


def purged_kfold_splits(
    t0: Any, t1: Any, n_splits: int, pct_embargo: float = 0.0
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Purged k-fold splits (AFML Snippet 7.3) as ``(train_idx, test_idx)`` numpy arrays.

    Folds are contiguous blocks in sample order. A training label is purged when its span
    intersects the test fold's window, from its first label's start to its latest label end.
    ``ceil(pct_embargo * n)`` further samples are embargoed after the fold only, starting where
    the purge ends: at the first sample whose label starts after the fold's latest label end.
    """
    s0, s1 = _label_spans(t0, t1)
    return [
        (_indices(train), _indices(test))
        for train, test in _cv.purged_kfold_splits(s0, s1, int(n_splits), float(pct_embargo))
    ]


def split_with_diagnostics(t0: Any, t1: Any, n_splits: int, pct_embargo: float = 0.0) -> list[dict]:
    """The folds of :func:`purged_kfold_splits`, each with why every excluded sample left.

    Each dict has ``split_id``, ``train_indices``, ``test_indices``, ``test_ranges``
    (half-open ``(start, stop)`` pairs), ``purged_indices``, ``embargo_indices`` (every
    non-test sample in an embargo window, purged or not) and ``overlap_count_after_purge``
    (always 0).
    """
    s0, s1 = _label_spans(t0, t1)
    return [
        _split_dict(d)
        for d in _cv.split_with_diagnostics(s0, s1, int(n_splits), float(pct_embargo))
    ]


def cpcv_splits(
    t0: Any, t1: Any, n_splits: int, n_test_splits: int, pct_embargo: float = 0.0
) -> list[dict]:
    """Combinatorial purged CV splits (AFML §12.4): one per choice of ``n_test_splits`` folds.

    There are C(n_splits, n_test_splits) splits, in lexicographic order of ``test_fold_ids``.
    Each dict has the keys of :func:`split_with_diagnostics` plus ``test_fold_ids``. Adjacent
    test folds are purged as one block, so ``n_test_splits = 1`` gives the k-fold splits.
    """
    s0, s1 = _label_spans(t0, t1)
    return [
        _split_dict(d)
        for d in _cv.cpcv_splits(s0, s1, int(n_splits), int(n_test_splits), float(pct_embargo))
    ]


def cpcv_paths(n_splits: int, n_test_splits: int) -> np.ndarray:
    """The φ = k/N · C(N, k) CPCV backtest paths as an ``(n_paths, n_splits)`` int array.

    ``paths[p, g]`` is the ``split_id`` of the split whose predictions path ``p`` uses for fold
    ``g``: the ``p``-th split, in ``split_id`` order, that tests fold ``g`` (AFML §12.4). The
    numbering matches :func:`openquant.backtesting_engine.run_cpcv`.
    """
    return np.asarray(_cv.cpcv_paths(int(n_splits), int(n_test_splits)), dtype=np.intp)


def walk_forward_splits(
    t0: Any, t1: Any, n_splits: int, pct_embargo: float = 0.0, min_train_folds: int = 1
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Walk-forward splits over the folds of :func:`purged_kfold_splits`, forward-only.

    Fold ``g >= min_train_folds`` is tested exactly as in :func:`purged_kfold_splits`, but it
    trains only on the samples *before* it whose label spans do not reach its window (purged).
    No model sees a sample after its test fold, so the embargo, which only drops samples after
    a test block, removes nothing. The first ``min_train_folds`` folds are never tested.

    Use it to compare features with memory (price levels, weakly differenced prices, long
    averages): purged k-fold and CPCV inflate them even on random walks (issue #217). Because
    the folds are the k-fold folds, a feature can be scored fold for fold under both schemes.
    """
    return [
        (d["train_indices"], d["test_indices"])
        for d in walk_forward_split_with_diagnostics(t0, t1, n_splits, pct_embargo, min_train_folds)
    ]


def walk_forward_split_with_diagnostics(
    t0: Any, t1: Any, n_splits: int, pct_embargo: float = 0.0, min_train_folds: int = 1
) -> list[dict]:
    """The splits of :func:`walk_forward_splits` as dicts, with the purged indices.

    Each dict has the keys of :func:`split_with_diagnostics` plus ``test_fold_id``, the fold
    tested. ``purged_indices`` lists the earlier samples purging removed; ``embargo_indices`` is
    always empty; samples after the test fold are left out without being listed.
    """
    s0, s1 = _label_spans(t0, t1)
    return [
        _split_dict(d)
        for d in _cv.walk_forward_splits(
            s0, s1, int(n_splits), float(pct_embargo), int(min_train_folds)
        )
    ]


def bootstrap_returns(
    returns: Any, demean: bool = True
) -> Callable[[np.random.Generator], np.ndarray]:
    """A no-signal generator for :func:`null_score_distribution`: returns drawn i.i.d.

    Each call draws ``len(returns)`` rows of ``returns`` with replacement (a 1-D array, or a
    2-D array whose rows are drawn whole, keeping the cross-section of each bar). With
    ``demean`` (the default) the mean return is subtracted first, per column, so the drawn
    path is a driftless random walk with the returns' distribution: nothing in it predicts the
    next return. Rebuild prices, features **and labels** from it.

    Two tempting alternatives are wrong for features with memory. Permuting the labels alone
    keeps every feature but cuts the link between a label and the path a level feature
    records, so it cannot show the bias of issue #217. Shuffling the returns *without*
    replacement fixes their sum, so every shuffled path ends where the real one does: a
    random-walk bridge, which is mean-reverting, and on which a level feature scores above
    chance even walk-forward.
    """
    arr = np.asarray(returns, dtype=float)
    if arr.ndim not in (1, 2) or arr.shape[0] < 2:
        raise ValueError("returns must be a 1-D or 2-D array with at least two rows")
    if not np.isfinite(arr).all():
        raise ValueError("returns must be finite")
    if demean:
        arr = arr - arr.mean(axis=0)

    def draw(rng: np.random.Generator) -> np.ndarray:
        return arr[rng.integers(0, arr.shape[0], size=arr.shape[0])]

    return draw


def null_score_distribution(
    evaluate: Callable[[Any, Sequence[Any]], float],
    splits: Sequence[Any],
    make_null: Callable[[np.random.Generator], Any],
    n_null: int = 100,
    seed: int | Sequence[int] | None = 0,
) -> np.ndarray:
    """Scores of a pipeline on ``n_null`` no-signal datasets, all run through the same splits.

    For each draw, ``make_null(rng)`` builds a dataset with no signal (for example
    :func:`bootstrap_returns`, or a simulation of the same length), and
    ``evaluate(data, splits)`` rebuilds features and labels from it, fits on each split's
    training indices, scores on its test indices and returns one number. The splits are fixed,
    so whatever the splitting scheme itself adds to the score (such as purged k-fold's bias
    toward features with memory, issue #217) is in the null too. Compare the real score with
    this distribution (:func:`null_p_value`), not with chance.

    ``make_null`` must return data of the length ``splits`` indexes. ``seed`` seeds one
    ``numpy.random.Generator`` passed to every call, so the distribution is reproducible.
    Returns the ``n_null`` scores in draw order; a ``NaN`` score is kept, not dropped.
    """
    n_null = int(n_null)
    if n_null < 1:
        raise ValueError(f"n_null must be at least 1, got {n_null}")
    splits = list(splits)
    if not splits:
        raise ValueError("splits cannot be empty")
    rng = np.random.default_rng(seed)
    out = np.empty(n_null)
    for i in range(n_null):
        out[i] = float(evaluate(make_null(rng), splits))
    return out


def null_p_value(observed: float, null: Any) -> float:
    """One-sided p-value of ``observed`` against a null distribution: P(null >= observed).

    With ``n`` finite null scores of which ``k`` are at least ``observed``, returns
    ``(k + 1) / (n + 1)``, so it is never 0 and a null of 99 draws resolves 0.01. ``NaN``
    null scores are ignored.
    """
    arr = np.asarray(null, dtype=float).ravel()
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        raise ValueError("null has no finite scores")
    if not math.isfinite(observed):
        raise ValueError(f"observed must be finite, got {observed}")
    return float((np.count_nonzero(arr >= observed) + 1) / (arr.size + 1))


def naive_kfold_splits(n_samples: int, n_splits: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Unpurged contiguous k-fold: the leaky baseline AFML §7.3 warns against.

    For measuring leakage with :func:`count_train_test_overlaps`, not for validating models.
    """
    return [
        (_indices(train), _indices(test))
        for train, test in _cv.naive_kfold_splits(int(n_samples), int(n_splits))
    ]


def count_train_test_overlaps(
    t0: Any, t1: Any, train_indices: Sequence[int], test_indices: Sequence[int]
) -> int:
    """How many training samples have a label span intersecting some test sample's span."""
    s0, s1 = _label_spans(t0, t1)
    return _cv.count_train_test_overlaps(
        s0,
        s1,
        _index_list(train_indices, "train_indices"),
        _index_list(test_indices, "test_indices"),
    )

"""Purged k-fold and combinatorial purged cross-validation (AFML chapters 7 and 12).

Every function returns indices, never fits a model: pass the splits to any estimator, for
example ``sklearn.model_selection.cross_val_score(model, X, y, cv=splits)``. The splitting,
purging and embargo are the Rust ``openquant::cross_validation`` code.

Label spans are required. ``t0[i]`` is when sample ``i``'s label starts (usually the event
time) and ``t1[i]`` when it is resolved (the triple-barrier touch). They may be numpy
``datetime64`` arrays, pandas or polars datetime columns, lists of ``datetime`` or ISO strings,
or plain integers such as bar positions; samples must be in time order.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from . import _core

_cv = _core.cross_validation

__all__ = [
    "purged_kfold_splits",
    "split_with_diagnostics",
    "cpcv_splits",
    "cpcv_paths",
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
    ``ceil(pct_embargo * n)`` further samples are embargoed on *both* sides of the fold, counted
    from the fold's edges; the book embargoes only after the fold (issue #134 tracks this).
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

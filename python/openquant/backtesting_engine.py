"""CPCV backtest paths from out-of-sample results you computed (AFML chapters 11 and 12).

The workflow keeps the model in Python and the bookkeeping in Rust:

1. ``splits = openquant.cross_validation.cpcv_splits(t0, t1, n_groups, test_groups, pct_embargo)``
2. for each split, fit on ``train_indices`` and compute the strategy's out-of-sample return
   for every sample in ``test_indices`` (in that order);
3. ``run_cpcv(t0, t1, split_returns, ...)`` scores every split and every one of the
   φ[N, k] backtest paths, with the Rust ``openquant::backtesting_engine::run_cpcv``.

:func:`assemble_cpcv_paths` stitches any per-split out-of-sample values (predictions,
positions, returns) into full paths, for statistics the engine does not compute.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from numbers import Integral
from typing import Any

import numpy as np

from . import _core
from .cross_validation import _label_spans

_bt = _core.backtesting_engine

__all__ = ["cpcv_path_count", "run_cpcv", "assemble_cpcv_paths"]


def cpcv_path_count(n_groups: int, test_groups: int) -> int:
    """φ[N, k] = k/N · C(N, k), the number of backtest paths CPCV produces (AFML §12.4)."""
    return _bt.cpcv_path_count(int(n_groups), int(test_groups))


def run_cpcv(
    t0: Any,
    t1: Any,
    split_returns: Sequence[Sequence[float]],
    *,
    n_groups: int,
    test_groups: int,
    pct_embargo: float,
    mode_provenance: str,
    trials_count: int,
    safeguards: Mapping[str, str],
) -> dict:
    """Score precomputed CPCV out-of-sample returns per split and per backtest path.

    ``split_returns[s]`` holds split ``s``'s out-of-sample returns, one per sample of that
    split's ``test_indices`` from :func:`openquant.cross_validation.cpcv_splits` with the same
    ``n_groups``, ``test_groups`` and spans; the two modules test the same samples in the same
    split order. ``mode_provenance``, ``trials_count`` (how many configurations were tried,
    for deflating the Sharpe ratio later) and the five ``safeguards`` statements
    (``survivorship_bias_control``, ``look_ahead_control``, ``data_mining_control``,
    ``cost_assumption``, ``multiple_testing_control``) are required and recorded in the
    diagnostics, as AFML chapter 11 asks.

    Returns a dict with ``folds`` (per-split ``sharpe``, ``mean_return``, ``std_return``,
    ``observations``), ``splits``, ``path_count``, ``path_assignments`` (one list per path,
    ``split_for_group``), ``path_distribution`` (per-path statistics) and ``diagnostics``.
    Each ``sharpe`` is the t-statistic ``mean / std * sqrt(n)``, not annualised.

    The engine purges and embargoes its own copy of the splits, reported in ``splits``. Its
    purge compares each training label with each test label and its embargo is the one
    ``cross_validation`` uses, so with increasing label starts its training sets equal
    ``cpcv_splits``'s. Those training sets are reported, not used: only the test indices,
    identical in both modules, and your returns are.
    """
    s0, s1 = _label_spans(t0, t1)
    returns = [np.asarray(r, dtype=np.float64).tolist() for r in split_returns]
    return _bt.run_cpcv(
        s0,
        s1,
        returns,
        n_groups=int(n_groups),
        test_groups=int(test_groups),
        pct_embargo=float(pct_embargo),
        mode_provenance=str(mode_provenance),
        trials_count=int(trials_count),
        safeguards={str(k): str(v) for k, v in dict(safeguards).items()},
    )


def _index_array(values: Any, name: str, ndim: int) -> np.ndarray:
    """Validate indices before narrowing; float/bool coercion can change identity."""
    limit = np.iinfo(np.intp).max
    if isinstance(values, np.ndarray) and values.dtype.kind in "iu":
        if values.ndim != ndim:
            raise ValueError(f"{name} must have {ndim} dimensions")
        if (values < 0).any() or (values > limit).any():
            raise ValueError(f"{name} indices must be non-negative and fit in np.intp")
        return values.astype(np.intp, copy=False)
    # Preserve element types in ordinary lists: np.asarray([True, 1]) would
    # turn the boolean into an integer before it can be rejected.
    raw = np.asarray(values, dtype=object)
    if raw.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions")
    for value in raw.flat:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
            raise ValueError(f"{name} must contain integer indices, not floats or booleans")
        if not 0 <= int(value) <= limit:
            raise ValueError(f"{name} indices must be non-negative and fit in np.intp")
    return np.asarray(raw, dtype=np.intp)


def assemble_cpcv_paths(
    split_values: Sequence[Sequence[float]],
    splits: Sequence[Mapping[str, Any]],
    paths: Any,
) -> np.ndarray:
    """Stitch per-split out-of-sample values into full CPCV paths.

    ``splits`` is the output of :func:`openquant.cross_validation.cpcv_splits`, ``paths`` that
    of :func:`openquant.cross_validation.cpcv_paths`, and ``split_values[s]`` holds one value
    per sample of ``splits[s]["test_indices"]``. Returns an ``(n_paths, n_samples)`` float array
    whose row ``p`` takes, for every fold ``g``, the values split ``paths[p, g]`` produced for
    that fold's samples: one out-of-sample value per sample, per path.
    Indices must be exact non-negative integers. Per-split test indices must be
    sorted and unique; inferred folds must partition contiguous sample indices
    starting at zero. Invalid indices or inconsistent fold partitions raise
    ValueError rather than yielding partially initialized paths. Values may contain NaN/inf;
    this function aligns them but does not score or validate returns.
    """
    paths = _index_array(paths, "paths", 2)
    if paths.shape[0] == 0 or paths.shape[1] == 0:
        raise ValueError("paths must be the (n_paths, n_groups) array cpcv_paths returns")
    if len(split_values) != len(splits):
        raise ValueError(
            f"split_values has {len(split_values)} entries but there are {len(splits)} splits"
        )
    tests = [
        _index_array(s["test_indices"], f"splits[{i}].test_indices", 1)
        for i, s in enumerate(splits)
    ]
    groups = [
        _index_array(s["test_fold_ids"], f"splits[{i}].test_fold_ids", 1)
        for i, s in enumerate(splits)
    ]
    for i, (test, group) in enumerate(zip(tests, groups)):
        if test.size == 0 or (test[1:] <= test[:-1]).any():
            raise ValueError(f"split {i} test_indices must be nonempty, sorted and unique")
        if group.size == 0 or len(np.unique(group)) != len(group):
            raise ValueError(f"split {i} must test distinct, nonempty fold IDs")
    values = []
    for s, test in enumerate(tests):
        v = np.asarray(split_values[s], dtype=np.float64)
        if v.shape != test.shape:
            raise ValueError(
                f"split_values[{s}] has {v.size} values but split {s} tests {test.size} samples"
            )
        values.append(v)

    # Fold g's samples are the ones every split testing g tests: with k < N, any other fold is
    # left out of at least one of those splits.
    n_groups = paths.shape[1]
    folds: list[np.ndarray | None] = [None] * n_groups
    for i, (group, test) in enumerate(zip(groups, tests)):
        for g in group:
            if not 0 <= g < n_groups:
                raise ValueError(f"split {i} tests fold {g}, outside 0..{n_groups}")
            folds[g] = test if folds[g] is None else np.intersect1d(folds[g], test)
    fold_samples = [f for f in folds if f is not None]
    if len(fold_samples) != n_groups:
        raise ValueError("every fold must be tested by at least one split")
    if any(f.size == 0 for f in fold_samples):
        raise ValueError("every fold must have a nonempty sample intersection")
    n_samples = sum(len(f) for f in fold_samples)
    if not np.array_equal(np.sort(np.concatenate(fold_samples)), np.arange(n_samples)):
        raise ValueError("fold samples must partition contiguous indices starting at zero")
    for i, group in enumerate(groups):
        expected = np.sort(np.concatenate([fold_samples[g] for g in group]))
        if not np.array_equal(expected, tests[i]):
            raise ValueError(f"split {i} test_indices do not match its inferred folds")

    out = np.empty((paths.shape[0], n_samples))
    for p, row in enumerate(paths):
        for g, s in enumerate(row):
            if not 0 <= s < len(splits) or g not in groups[s]:
                raise ValueError(f"path {p} takes fold {g} from split {s}, which does not test it")
            out[p, fold_samples[g]] = values[s][np.searchsorted(tests[s], fold_samples[g])]
    return out

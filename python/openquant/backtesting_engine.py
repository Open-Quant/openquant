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

    The engine purges and embargoes its own copy of the splits, reported in ``splits``: its
    purge compares each training label with each test label, and its embargo differs from
    ``cross_validation`` (see the backtesting-engine page). Those training sets are reported,
    not used: only the test indices, identical in both modules, and your returns are.
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
    """
    paths = np.asarray(paths, dtype=np.intp)
    if paths.ndim != 2 or paths.shape[0] == 0:
        raise ValueError("paths must be the (n_paths, n_groups) array cpcv_paths returns")
    if len(split_values) != len(splits):
        raise ValueError(
            f"split_values has {len(split_values)} entries but there are {len(splits)} splits"
        )
    tests = [np.asarray(s["test_indices"], dtype=np.intp) for s in splits]
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
    for split, test in zip(splits, tests):
        for g in split["test_fold_ids"]:
            if not 0 <= g < n_groups:
                raise ValueError(f"split {split['split_id']} tests fold {g}, outside 0..{n_groups}")
            folds[g] = test if folds[g] is None else np.intersect1d(folds[g], test)
    fold_samples = [f for f in folds if f is not None]
    if len(fold_samples) != n_groups:
        raise ValueError("every fold must be tested by at least one split")
    n_samples = sum(len(f) for f in fold_samples)

    out = np.empty((paths.shape[0], n_samples))
    for p, row in enumerate(paths):
        for g, s in enumerate(row):
            if not 0 <= s < len(splits) or g not in tuple(splits[s]["test_fold_ids"]):
                raise ValueError(f"path {p} takes fold {g} from split {s}, which does not test it")
            out[p, fold_samples[g]] = values[s][np.searchsorted(tests[s], fold_samples[g])]
    return out

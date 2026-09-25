"""Hyperparameter search on purged k-fold splits (AFML chapter 9), for any Python model.

With scikit-learn, pass purged splits as ``cv``; the leakage control is in the splits::

    from sklearn.model_selection import GridSearchCV
    from openquant.cross_validation import purged_kfold_splits

    splits = purged_kfold_splits(t0, t1, n_splits=5, pct_embargo=0.01)
    search = GridSearchCV(model, param_grid, cv=splits, scoring="neg_log_loss")
    search.fit(X, y, sample_weight=w)

scikit-learn fits with ``sample_weight`` but scores unweighted (AFML §9.2 and Snippet 7.4).
:func:`purged_search` scores each fold with :func:`classification_score`, which applies the
test fold's weights, and evaluates candidates from :func:`expand_param_grid` or
:func:`sample_param_sets` in the same order as the Rust ``grid_search`` and
``randomized_search``.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

import numpy as np

from . import _core
from .cross_validation import purged_kfold_splits

_ht = _core.hyperparameter_tuning

__all__ = ["expand_param_grid", "sample_param_sets", "classification_score", "purged_search"]


def _scalar(value: Any) -> Any:
    """numpy scalars as the Python int / float / bool the Rust side accepts."""
    return value.item() if isinstance(value, np.generic) else value


def expand_param_grid(param_grid: Mapping[str, Sequence[Any]]) -> list[dict[str, Any]]:
    """Every combination of a grid of int, float or bool values, keys in sorted order.

    The same candidates, in the same order, as the Rust ``grid_search`` evaluates.
    """
    grid = {str(k): [_scalar(v) for v in values] for k, values in param_grid.items()}
    return _ht.expand_param_grid(grid)


def sample_param_sets(
    param_space: Mapping[str, tuple], n_iter: int, seed: int
) -> list[dict[str, Any]]:
    """``n_iter`` random draws from ``param_space``, seeded, as the Rust ``randomized_search``.

    Each value of ``param_space`` is one of ``("choice", [values])``, ``("uniform", low,
    high)``, ``("log_uniform", low, high)`` (AFML §9.3: sample scale parameters such as ``C``
    uniformly in their logarithm) or ``("int", low, high)`` with ``high`` included. The same
    ``seed`` always gives the same draws.
    """
    space = {}
    for key, spec in param_space.items():
        spec = tuple(spec)
        if spec and spec[0] == "choice" and len(spec) == 2:
            spec = ("choice", [_scalar(v) for v in spec[1]])
        else:
            spec = tuple(_scalar(v) for v in spec)
        space[str(key)] = spec
    return _ht.sample_param_sets(space, int(n_iter), int(seed))


def classification_score(
    y_true: Any,
    probabilities: Any,
    sample_weight: Any = None,
    scoring: str = "neg_log_loss",
) -> float:
    """Score probabilities of ``y == 1`` against 0/1 labels, weighting samples.

    ``scoring`` is ``"neg_log_loss"`` (higher is better), ``"accuracy"`` or
    ``"balanced_accuracy"`` (mean recall over the classes present).
    """
    return _ht.classification_score(
        np.asarray(y_true, dtype=np.float64).tolist(),
        np.asarray(probabilities, dtype=np.float64).tolist(),
        None if sample_weight is None else np.asarray(sample_weight, dtype=np.float64).tolist(),
        scoring,
    )


def _positive_proba(model: Any, x: np.ndarray) -> np.ndarray:
    p = np.asarray(model.predict_proba(x), dtype=np.float64)
    if p.ndim == 2:
        classes = list(getattr(model, "classes_", [0, 1]))
        if 1 not in classes:
            return np.zeros(len(x))
        p = p[:, classes.index(1)]
    return p


def purged_search(
    make_estimator: Callable[[dict[str, Any]], Any],
    param_sets: Sequence[Mapping[str, Any]],
    X: Any,
    y: Any,
    t0: Any,
    t1: Any,
    *,
    n_splits: int = 5,
    pct_embargo: float = 0.01,
    scoring: str = "neg_log_loss",
    sample_weight: Any = None,
) -> dict[str, Any]:
    """Cross-validate each parameter set on purged folds and pick the best mean score.

    ``make_estimator(params)`` returns an unfitted model with ``fit(X, y, sample_weight=...)``
    and ``predict_proba(X)``. Each fold fits a fresh model on the training samples with their
    weights and scores the test fold with :func:`classification_score` and the test weights,
    as the Rust ``grid_search`` does. Returns ``best_params``, ``best_score`` and ``trials``
    (``params``, ``fold_scores``, ``mean_score`` per candidate). On a tie the later candidate
    wins, as in Rust.
    """
    x = np.asarray(X, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"X must have shape (n_samples, n_features), got {x.shape}")
    yv = np.asarray(y, dtype=np.float64)
    if yv.shape != (len(x),):
        raise ValueError(f"y must have {len(x)} values, got shape {yv.shape}")
    w = None if sample_weight is None else np.asarray(sample_weight, dtype=np.float64)
    if w is not None and w.shape != (len(x),):
        raise ValueError(f"sample_weight must have {len(x)} values, got shape {w.shape}")
    if not param_sets:
        raise ValueError("param_sets cannot be empty")

    splits = purged_kfold_splits(t0, t1, n_splits, pct_embargo)
    n_spans = sum(len(test) for _, test in splits)
    if n_spans != len(x):
        raise ValueError(f"t0/t1 describe {n_spans} samples but X has {len(x)} rows")

    trials = []
    for params in param_sets:
        params = dict(params)
        fold_scores = []
        for train, test in splits:
            if len(train) == 0 or len(test) == 0:
                raise ValueError("a purged fold has an empty training or test set")
            model = make_estimator(params)
            if w is None:
                model.fit(x[train], yv[train])
            else:
                model.fit(x[train], yv[train], sample_weight=w[train])
            proba = _positive_proba(model, x[test])
            fold_scores.append(
                classification_score(yv[test], proba, None if w is None else w[test], scoring)
            )
        trials.append(
            {
                "params": params,
                "fold_scores": fold_scores,
                "mean_score": sum(fold_scores) / len(fold_scores),
            }
        )

    best = trials[0]
    for trial in trials[1:]:
        if trial["mean_score"] >= best["mean_score"]:
            best = trial
    return {"best_params": best["params"], "best_score": best["mean_score"], "trials": trials}

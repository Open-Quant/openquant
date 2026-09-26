"""MDI, MDA and SFI feature importance (AFML chapter 8) for any scikit-learn-style model.

MDA and SFI are cross-validated, and always on purged k-fold splits built from the label
spans ``t0``/``t1``, which are required (AFML §8.3, and issue #27). There are two ways in:

* :func:`mean_decrease_accuracy` and :func:`single_feature_importance` take an estimator with
  ``fit(X, y, sample_weight=...)`` and ``predict_proba(X)``, run the fit/predict loop here in
  Python, and hand the out-of-sample probabilities to Rust for scoring.
* :func:`mda_from_probabilities` and :func:`sfi_from_probabilities` take those probabilities
  directly, for a model trained elsewhere on the folds of
  :func:`openquant.cross_validation.purged_kfold_splits` with the same arguments.

Either way the scoring, the MDA normalisation and the aggregation are the Rust
``openquant::feature_importance`` code. Results map each feature name to
``{"mean": ..., "std": ...}``, where ``std`` is the standard error across folds (MDA, SFI) or
trees (MDI), in the order of ``feature_names``.
"""

from __future__ import annotations

import copy
import warnings
from collections.abc import Sequence
from typing import Any

import numpy as np

from . import _core
from .cross_validation import _label_spans, purged_kfold_splits

_fi = _core.feature_importance

__all__ = [
    "mean_decrease_impurity",
    "mean_decrease_accuracy",
    "single_feature_importance",
    "mda_from_probabilities",
    "sfi_from_probabilities",
]


def _names(n_features: int, feature_names: Sequence[str] | None) -> list[str]:
    if feature_names is None:
        return [f"f{i}" for i in range(n_features)]
    names = [str(n) for n in feature_names]
    if len(names) != n_features:
        raise ValueError(f"feature_names has {len(names)} names for {n_features} features")
    if len(set(names)) != len(names):
        raise ValueError("feature_names must be unique")
    return names


def _ordered(result: dict, names: list[str]) -> dict[str, dict[str, float]]:
    return {n: {"mean": result[n][0], "std": result[n][1]} for n in names}


def _vector(values: Any, name: str) -> list[float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    return arr.tolist()


def _weights(sample_weight: Any) -> list[float] | None:
    return None if sample_weight is None else _vector(sample_weight, "sample_weight")


def _columns(values: Any, n_features: int, name: str) -> list[list[float]]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != n_features:
        raise ValueError(f"{name} must have shape (n_samples, {n_features}), got {arr.shape}")
    return arr.T.tolist()


def mean_decrease_impurity(
    per_tree_importances: Any, feature_names: Sequence[str] | None = None
) -> dict[str, dict[str, float]]:
    """MDI (AFML Snippet 8.2) from one row of impurity importances per tree.

    With scikit-learn: ``[t.feature_importances_ for t in forest.estimators_]``. A zero means
    the tree never split on the feature and is left out of that feature's mean. Means are
    normalised to sum to 1. AFML fits the forest with ``max_features=1`` so that every feature
    gets a chance to be split on.
    """
    rows = np.asarray(per_tree_importances, dtype=np.float64)
    if rows.ndim != 2:
        raise ValueError("per_tree_importances must be two-dimensional (trees x features)")
    names = _names(rows.shape[1], feature_names)
    return _ordered(_fi.mean_decrease_impurity(rows.tolist(), names), names)


def _seed(seed: int) -> int:
    value = int(seed)
    if value < 0:
        raise ValueError(f"seed must be a non-negative integer, got {seed}")
    return value


def mda_from_probabilities(
    y: Any,
    t0: Any,
    t1: Any,
    base_proba: Any,
    permuted_proba: Any,
    *,
    n_splits: int,
    pct_embargo: float = 0.0,
    scoring: str = "neg_log_loss",
    sample_weight: Any = None,
    feature_names: Sequence[str] | None = None,
    seed: int | None = None,
) -> dict[str, dict[str, float]]:
    """MDA (AFML Snippet 8.3) scored from out-of-sample probabilities of ``y == 1``.

    ``base_proba[i]`` is the probability for sample ``i`` from the model trained on the fold
    that tests ``i``; ``permuted_proba[i, j]`` is the same model's probability after feature
    ``j`` was shuffled within that test fold. The folds are
    ``purged_kfold_splits(t0, t1, n_splits, pct_embargo)``; every sample is tested once.
    Per fold, a feature's importance is ``(base - permuted) / (1 - permuted)`` for accuracy
    and F1, and ``(base - permuted) / -permuted`` for negative log loss. ``sample_weight``
    weights the scores.

    ``seed`` is deprecated and passing it emits a ``DeprecationWarning``: the shuffling
    happened when ``permuted_proba`` was computed, so no seed here can change the result.
    """
    if seed is not None:
        warnings.warn(
            "mda_from_probabilities: seed is deprecated and has no effect; the shuffling "
            "happened when permuted_proba was computed",
            DeprecationWarning,
            stacklevel=2,
        )
    y_list = _vector(y, "y")
    s0, s1 = _label_spans(t0, t1)
    permuted = np.asarray(permuted_proba, dtype=np.float64)
    if permuted.ndim != 2:
        raise ValueError("permuted_proba must have shape (n_samples, n_features)")
    names = _names(permuted.shape[1], feature_names)
    result = _fi.mda_from_probabilities(
        y_list,
        s0,
        s1,
        _vector(base_proba, "base_proba"),
        _columns(permuted, len(names), "permuted_proba"),
        names,
        n_splits=int(n_splits),
        pct_embargo=float(pct_embargo),
        scoring=scoring,
        sample_weight=_weights(sample_weight),
    )
    return _ordered(result, names)


def sfi_from_probabilities(
    y: Any,
    t0: Any,
    t1: Any,
    proba: Any,
    *,
    n_splits: int,
    pct_embargo: float = 0.0,
    scoring: str = "neg_log_loss",
    sample_weight: Any = None,
    feature_names: Sequence[str] | None = None,
) -> dict[str, dict[str, float]]:
    """SFI (AFML Snippet 8.4) scored from out-of-sample probabilities of ``y == 1``.

    ``proba[i, j]`` is the probability for sample ``i`` from a model trained on feature ``j``
    alone, on the fold of ``purged_kfold_splits(t0, t1, n_splits, pct_embargo)`` that tests
    ``i``. ``sample_weight`` weights each fold's score, as in :func:`mda_from_probabilities`
    and AFML's ``cvScore``; weight the fit on your side as well.
    """
    y_list = _vector(y, "y")
    s0, s1 = _label_spans(t0, t1)
    arr = np.asarray(proba, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("proba must have shape (n_samples, n_features)")
    names = _names(arr.shape[1], feature_names)
    result = _fi.sfi_from_probabilities(
        y_list,
        s0,
        s1,
        _columns(arr, len(names), "proba"),
        names,
        n_splits=int(n_splits),
        pct_embargo=float(pct_embargo),
        scoring=scoring,
        sample_weight=_weights(sample_weight),
    )
    return _ordered(result, names)


def _clone(estimator: Any) -> Any:
    try:
        from sklearn.base import clone
    except ImportError:
        return copy.deepcopy(estimator)
    try:
        return clone(estimator)
    except TypeError:
        return copy.deepcopy(estimator)


def _fit(estimator: Any, x: np.ndarray, y: np.ndarray, weight: np.ndarray | None) -> Any:
    model = _clone(estimator)
    if weight is None:
        model.fit(x, y)
    else:
        model.fit(x, y, sample_weight=weight)
    return model


def _positive_proba(model: Any, x: np.ndarray) -> np.ndarray:
    p = np.asarray(model.predict_proba(x), dtype=np.float64)
    if p.ndim == 2:
        classes = list(getattr(model, "classes_", [0, 1]))
        if 1 not in classes and 1.0 not in classes:
            # A fold whose training labels are all 0 never predicts class 1.
            return np.zeros(len(x))
        p = p[:, classes.index(1)]
    return p


def _prepare(
    X: Any, y: Any, sample_weight: Any
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    x = np.asarray(X, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"X must have shape (n_samples, n_features), got {x.shape}")
    yv = np.asarray(y, dtype=np.float64)
    if yv.shape != (len(x),):
        raise ValueError(f"y must have {len(x)} values, got shape {yv.shape}")
    w = None if sample_weight is None else np.asarray(sample_weight, dtype=np.float64)
    if w is not None and w.shape != (len(x),):
        raise ValueError(f"sample_weight must have {len(x)} values, got shape {w.shape}")
    return x, yv, w


def mean_decrease_accuracy(
    estimator: Any,
    X: Any,
    y: Any,
    t0: Any,
    t1: Any,
    *,
    n_splits: int = 5,
    pct_embargo: float = 0.01,
    scoring: str = "neg_log_loss",
    sample_weight: Any = None,
    feature_names: Sequence[str] | None = None,
    seed: int = 42,
) -> dict[str, dict[str, float]]:
    """MDA (AFML Snippet 8.3) for a binary classifier with ``fit`` and ``predict_proba``.

    For each purged fold a copy of ``estimator`` (``sklearn.base.clone`` when scikit-learn is
    installed) is fitted on the training samples with their ``sample_weight``, then each
    feature column of the test fold is shuffled in turn with ``numpy.random.default_rng(seed)``.
    ``seed`` defaults to 42, as in :func:`openquant.feature_diagnostics.mda_importance`. Labels
    must be 0/1. See
    :func:`mda_from_probabilities` for the scoring.
    """
    x, yv, w = _prepare(X, y, sample_weight)
    names = _names(x.shape[1], feature_names)
    rng = np.random.default_rng(_seed(seed))
    base = np.empty(len(x))
    permuted = np.empty(x.shape)
    for train, test in purged_kfold_splits(t0, t1, n_splits, pct_embargo):
        model = _fit(estimator, x[train], yv[train], None if w is None else w[train])
        x_test = x[test]
        base[test] = _positive_proba(model, x_test)
        for j in range(x.shape[1]):
            shuffled = x_test.copy()
            shuffled[:, j] = rng.permutation(shuffled[:, j])
            permuted[test, j] = _positive_proba(model, shuffled)
    return mda_from_probabilities(
        yv,
        t0,
        t1,
        base,
        permuted,
        n_splits=n_splits,
        pct_embargo=pct_embargo,
        scoring=scoring,
        sample_weight=w,
        feature_names=names,
    )


def single_feature_importance(
    estimator: Any,
    X: Any,
    y: Any,
    t0: Any,
    t1: Any,
    *,
    n_splits: int = 5,
    pct_embargo: float = 0.01,
    scoring: str = "neg_log_loss",
    sample_weight: Any = None,
    feature_names: Sequence[str] | None = None,
) -> dict[str, dict[str, float]]:
    """SFI (AFML Snippet 8.4): each feature's out-of-sample score when the model sees only it.

    A copy of ``estimator`` is fitted per feature and purged fold on that one column, with the
    training ``sample_weight``. Labels must be 0/1. See :func:`sfi_from_probabilities`.
    """
    x, yv, w = _prepare(X, y, sample_weight)
    names = _names(x.shape[1], feature_names)
    proba = np.empty(x.shape)
    splits = purged_kfold_splits(t0, t1, n_splits, pct_embargo)
    for j in range(x.shape[1]):
        column = x[:, [j]]
        for train, test in splits:
            model = _fit(estimator, column[train], yv[train], None if w is None else w[train])
            proba[test, j] = _positive_proba(model, column[test])
    return sfi_from_probabilities(
        yv,
        t0,
        t1,
        proba,
        n_splits=n_splits,
        pct_embargo=pct_embargo,
        scoring=scoring,
        sample_weight=w,
        feature_names=names,
    )

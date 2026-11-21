"""Internal validation shared by estimator-driven out-of-sample diagnostics."""

from decimal import Decimal
from fractions import Fraction
from typing import Any

import numpy as np


def _positive_proba(model: Any, x: np.ndarray) -> np.ndarray:
    """One positive-class probability per requested row, without broadcasting."""
    raw = np.asarray(model.predict_proba(x))
    if raw.ndim not in (1, 2) or raw.shape[0] != len(x):
        raise ValueError("predict_proba must return one probability row per input row")
    if np.iscomplexobj(raw):
        raise ValueError("predict_proba must contain real probabilities")
    if raw.dtype.kind == "O":
        # Preserve exact limits for Decimal, Fraction and extended NumPy real
        # objects rather than allowing the binary64 conversion to hide them.
        for value in raw.flat:
            if isinstance(value, (complex, np.complexfloating)):
                raise ValueError("predict_proba must contain real probabilities")
            if (isinstance(value, Decimal) and not value.is_finite()) or (
                isinstance(value, (Decimal, Fraction, int, float, np.integer, np.floating))
                and not 0 <= value <= 1
            ):
                raise ValueError("predict_proba must contain finite probabilities in [0, 1]")
    # Check real floating arrays at their original precision: a long double
    # just outside [0, 1] can round into the interval when narrowed to float64.
    p = raw if raw.dtype.kind == "f" else raw.astype(np.float64, copy=False)
    if not np.all(np.isfinite(p)) or np.any((p < 0.0) | (p > 1.0)):
        raise ValueError("predict_proba must contain finite probabilities in [0, 1]")
    p = p.astype(np.float64, copy=False)
    if p.ndim == 1:
        # The vector API is already P(y == 1), independent of classes_ order.
        return p
    # Matrix columns follow classes_; absent metadata retains the binary default.
    classes = np.asarray(getattr(model, "classes_", [0, 1]))
    if (
        classes.ndim != 1
        or len(classes) != p.shape[1]
        or len(classes) not in (1, 2)
        or not np.all((classes == 0) | (classes == 1))
        or len(np.unique(classes)) != len(classes)
    ):
        raise ValueError("classes_ must identify unique binary labels for predict_proba columns")
    if 1 not in classes:
        # A valid one-class model trained only on class 0 has no positive column.
        return np.zeros(len(x))
    return p[:, int(np.flatnonzero(classes == 1)[0])]

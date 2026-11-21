"""Reject malformed held-out predictions rather than broadcasting or discarding them."""

import numpy as np
import pytest
from openquant import feature_importance as fi
from openquant import hyperparameter_tuning as ht


class Predictor:
    def __init__(self, kind):
        self.kind = kind

    def fit(self, x, y, sample_weight=None):
        self.classes_ = [0, 1]
        if self.kind == "no-positive":
            self.classes_ = [0]
        if self.kind == "wrong-classes":
            self.classes_ = [2, 3]
        if self.kind in ("zero-class-nan", "zero-class-negative"):
            self.classes_ = [0]
        if self.kind == "duplicate-classes":
            self.classes_ = [1, 1]
        return self

    def predict_proba(self, x):
        if self.kind == "zero-class-nan":
            return np.full((len(x), 1), np.nan)
        if self.kind == "zero-class-negative":
            return np.full((len(x), 1), -0.2)
        if self.kind == "scalar":
            return 0.7
        if self.kind == "short-vector":
            return np.array([0.7])
        if self.kind in ("short-matrix", "no-positive"):
            return np.array([[0.3, 0.7]])
        if self.kind == "extra-column":
            return np.tile([0.2, 0.3, 0.5], (len(x), 1))
        return np.tile([0.3, 0.7], (len(x), 1))


def run(method, model):
    x = np.arange(8.0).reshape(-1, 1)
    y = np.array([0.0, 1.0] * 4)
    t = np.arange(8)
    if method == "search":
        return ht.purged_search(lambda _: model, [{}], x, y, t, t, n_splits=2, pct_embargo=0)
    return getattr(fi, method)(model, x, y, t, t, n_splits=2, pct_embargo=0)


@pytest.mark.parametrize(
    "method", ["mean_decrease_accuracy", "single_feature_importance", "search"]
)
@pytest.mark.parametrize(
    "kind",
    [
        "scalar",
        "short-vector",
        "short-matrix",
        "no-positive",
        "extra-column",
        "wrong-classes",
        "duplicate-classes",
        "zero-class-nan",
        "zero-class-negative",
    ],
)
def test_malformed_predictions_are_not_silently_scored(method, kind):
    with pytest.raises(ValueError, match="predict_proba|classes_"):
        run(method, Predictor(kind))


class ValidPredictor:
    def __init__(self, kind):
        self.kind = kind

    def fit(self, x, y, sample_weight=None):
        if self.kind == "reversed":
            self.classes_ = np.array([1, 0])
        elif self.kind in ("only-zero", "only-one"):
            self.classes_ = [0 if self.kind == "only-zero" else 1]
        return self

    def predict_proba(self, x):
        if self.kind == "vector":
            return np.full(len(x), 0.7)
        if self.kind == "reversed":
            return np.tile([0.7, 0.3], (len(x), 1))
        if self.kind in ("only-zero", "only-one"):
            return np.ones((len(x), 1))
        return np.tile([0.3, 0.7], (len(x), 1))


@pytest.mark.parametrize(
    "method", ["mean_decrease_accuracy", "single_feature_importance", "search"]
)
@pytest.mark.parametrize("kind", ["vector", "default-matrix", "reversed", "only-zero", "only-one"])
def test_valid_binary_estimators_keep_expected_scores(method, kind):
    import math

    result = run(method, ValidPredictor(kind))
    if method == "mean_decrease_accuracy":
        # A constant predictor's score is unchanged by permutation.
        assert result["f0"]["mean"] == pytest.approx(0.0, abs=1e-15)
    else:
        # Each held-out fold contains equal counts of 0/1. Native scoring clips
        # boundary probabilities at 1e-15 and 1 - 1e-15.
        probability = 0.0 if kind == "only-zero" else 1.0 if kind == "only-one" else 0.7
        probability = max(1e-15, min(1 - 1e-15, probability))
        expected = (math.log(probability) + math.log1p(-probability)) / 2
        actual = result["best_score"] if method == "search" else result["f0"]["mean"]
        assert actual == pytest.approx(expected, rel=1e-12)

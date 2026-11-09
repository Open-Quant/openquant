"""Search validation must run before user estimators receive invalid training data."""

import numpy as np
import pytest
from openquant import hyperparameter_tuning as ht
from openquant.cross_validation import purged_kfold_splits


@pytest.mark.parametrize("field", ["labels", "weights"])
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_search_rejects_nonfinite_inputs_before_building_estimator(field, bad):
    x = np.zeros((12, 1))
    y = np.zeros(12)
    w = np.ones(12)
    (y if field == "labels" else w)[-1] = bad
    t0 = np.datetime64("2024-01-01", "ns") + np.arange(12) * np.timedelta64(1, "D")

    def unexpected_estimator(params):
        pytest.fail("invalid data reached the estimator factory")

    with pytest.raises(ValueError, match="finite"):
        ht.purged_search(
            unexpected_estimator,
            [{}],
            x,
            y,
            t0,
            t0,
            n_splits=3,
            pct_embargo=0.0,
            sample_weight=w,
        )


@pytest.mark.parametrize("field, bad", [("labels", 2.0), ("weights", -1.0)])
def test_search_rejects_invalid_finite_inputs_before_building_estimator(field, bad):
    x = np.zeros((12, 1))
    y = np.zeros(12)
    w = np.ones(12)
    (y if field == "labels" else w)[-1] = bad
    t0 = np.datetime64("2024-01-01", "ns") + np.arange(12) * np.timedelta64(1, "D")

    def unexpected_estimator(params):
        pytest.fail("invalid data reached the estimator factory")

    with pytest.raises(ValueError):
        ht.purged_search(
            unexpected_estimator,
            [{}],
            x,
            y,
            t0,
            t0,
            n_splits=3,
            pct_embargo=0.0,
            sample_weight=w,
        )


class ConstantEstimator:
    def fit(self, x, y, sample_weight=None):
        return self

    def predict_proba(self, x):
        return np.full(len(x), 0.5)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("bad_first", [True, False])
def test_search_rejects_nonfinite_fold_score_in_any_trial_order(monkeypatch, bad, bad_first):
    # Isolate selection from the scorer: invalid scoring output must never become a winner.
    scores = iter(([bad] * 3 + [0.5] * 3) if bad_first else ([0.5] * 3 + [bad] * 3))
    monkeypatch.setattr(ht, "classification_score", lambda *args: next(scores))
    x = np.zeros((12, 1))
    t0 = np.datetime64("2024-01-01", "ns") + np.arange(12) * np.timedelta64(1, "D")
    with pytest.raises(ValueError, match="finite"):
        ht.purged_search(
            lambda params: ConstantEstimator(),
            [{"id": 0}, {"id": 1}],
            x,
            np.zeros(12),
            t0,
            t0,
            n_splits=3,
            pct_embargo=0.0,
        )


def test_search_rejects_nonfinite_fold_mean(monkeypatch):
    monkeypatch.setattr(ht, "classification_score", lambda *args: 1e308)
    x = np.zeros((12, 1))
    t0 = np.datetime64("2024-01-01", "ns") + np.arange(12) * np.timedelta64(1, "D")
    with pytest.raises(ValueError, match="finite"):
        ht.purged_search(
            lambda params: ConstantEstimator(),
            [{}],
            x,
            np.zeros(12),
            t0,
            t0,
            n_splits=3,
            pct_embargo=0.0,
        )


def test_search_preserves_last_candidate_tie_break(monkeypatch):
    monkeypatch.setattr(ht, "classification_score", lambda *args: 0.5)
    x = np.zeros((12, 1))
    t0 = np.datetime64("2024-01-01", "ns") + np.arange(12) * np.timedelta64(1, "D")
    result = ht.purged_search(
        lambda params: ConstantEstimator(),
        [{"id": 0}, {"id": 1}],
        x,
        np.zeros(12),
        t0,
        t0,
        n_splits=3,
        pct_embargo=0.0,
    )
    assert result["best_params"] == {"id": 1}
    assert result["best_score"] == 0.5
    assert [trial["fold_scores"] for trial in result["trials"]] == [[0.5] * 3] * 2


def test_search_rejects_overflowing_weight_total_before_building_estimator():
    x = np.zeros((12, 1))
    t0 = np.datetime64("2024-01-01", "ns") + np.arange(12) * np.timedelta64(1, "D")

    def unexpected_estimator(params):
        pytest.fail("overflowing weights reached the estimator factory")

    with pytest.raises(ValueError, match="finite"):
        ht.purged_search(
            unexpected_estimator,
            [{}],
            x,
            np.zeros(12),
            t0,
            t0,
            n_splits=3,
            pct_embargo=0.0,
            sample_weight=np.full(12, 1e308),
        )


class _TwoColumnModel:
    """predict_proba with one column per entry of ``classes_``; column of 1.0 is ``x[:, 0]``."""

    def __init__(self, classes):
        self.classes_ = classes

    def fit(self, X, y, sample_weight=None):
        return self

    def predict_proba(self, X):
        p = np.asarray(X, dtype=float)[:, 0]
        columns = {1.0: p, 0.0: 1.0 - p}
        return np.column_stack([columns[float(c)] for c in self.classes_])


@pytest.mark.parametrize("classes", [[0.0, 1.0], [1.0, 0.0], [0, 1]])
def test_search_scores_the_class_one_column_whatever_its_position(classes):
    x = np.linspace(0.05, 0.95, 12).reshape(-1, 1)
    y = (x[:, 0] > 0.5).astype(float)
    t0 = np.datetime64("2024-01-01", "ns") + np.arange(12) * np.timedelta64(1, "D")
    result = ht.purged_search(
        lambda params: _TwoColumnModel(classes), [{}], x, y, t0, t0, n_splits=3, pct_embargo=0.0
    )
    expected = [
        ht.classification_score(y[test], x[test, 0])
        for _, test in purged_kfold_splits(t0, t0, 3, 0.0)
    ]
    assert result["trials"][0]["fold_scores"] == pytest.approx(expected)


def test_search_predicts_zero_when_the_model_never_saw_class_one():
    # A fold trained on 0 labels only has no class-1 column: its probabilities are all 0.
    x = np.linspace(0.05, 0.95, 12).reshape(-1, 1)
    y = np.array([0.0, 1.0] * 6)
    t0 = np.datetime64("2024-01-01", "ns") + np.arange(12) * np.timedelta64(1, "D")
    result = ht.purged_search(
        lambda params: _TwoColumnModel([0.0]),
        [{}],
        x,
        y,
        t0,
        t0,
        n_splits=3,
        pct_embargo=0.0,
        scoring="accuracy",
    )
    assert result["trials"][0]["fold_scores"] == [0.5, 0.5, 0.5]


@pytest.mark.parametrize(
    "X, y, w, message",
    [
        (np.zeros(12), np.zeros(12), None, "X must have shape"),
        (np.zeros((12, 1)), np.zeros(11), None, "y must have 12 values"),
        (np.zeros((12, 1)), np.zeros(12), np.ones(11), "sample_weight must have 12 values"),
    ],
)
def test_search_rejects_misshapen_inputs(X, y, w, message):
    t0 = np.datetime64("2024-01-01", "ns") + np.arange(12) * np.timedelta64(1, "D")
    with pytest.raises(ValueError, match=message):
        ht.purged_search(
            lambda params: pytest.fail("misshapen data reached the estimator factory"),
            [{}],
            X,
            y,
            t0,
            t0,
            n_splits=3,
            pct_embargo=0.0,
            sample_weight=w,
        )

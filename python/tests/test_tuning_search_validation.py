"""Search validation must run before user estimators receive invalid training data."""

import numpy as np
import pytest
from openquant import hyperparameter_tuning as ht


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

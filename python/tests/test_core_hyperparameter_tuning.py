"""openquant.hyperparameter_tuning: candidate generation, weighted scoring, purged search.

Fixtures and expected values are those of crates/openquant/tests/hyperparameter_tuning.rs;
`ThresholdClassifier` is a Python port of the Rust test classifier.
"""

import math

import numpy as np
import pytest

from openquant import hyperparameter_tuning as ht


def minute_spans(n):
    """Rust `make_series`: one label a minute, each lasting 3 minutes."""
    t0 = np.datetime64("2019-01-01T00:00", "ns") + np.arange(n) * np.timedelta64(1, "m")
    return t0, t0 + np.timedelta64(3, "m")


class ThresholdClassifier:
    def __init__(self, params):
        self.threshold = float(params.get("threshold", 0.5))
        self.sharpness = float(params.get("sharpness", 6.0))
        self.trained_prior = 0.5

    def fit(self, X, y, sample_weight=None):
        w = np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight)
        total = w.sum()
        self.trained_prior = w[np.asarray(y) == 1.0].sum() / total if total > 0 else 0.5
        return self

    def predict_proba(self, X):
        z = (np.asarray(X)[:, 0] - self.threshold) * self.sharpness
        logistic = 1.0 / (1.0 + np.exp(-z))
        return np.clip(0.85 * logistic + 0.15 * self.trained_prior, 0.0, 1.0)


GRID = {"threshold": [0.5, 0.7, 0.9], "sharpness": [4.0, 8.0]}


def test_expand_param_grid_in_rust_order():
    sets = ht.expand_param_grid(GRID)
    # Keys in sorted order, the first key varying slowest.
    assert sets == [
        {"sharpness": 4.0, "threshold": 0.5},
        {"sharpness": 4.0, "threshold": 0.7},
        {"sharpness": 4.0, "threshold": 0.9},
        {"sharpness": 8.0, "threshold": 0.5},
        {"sharpness": 8.0, "threshold": 0.7},
        {"sharpness": 8.0, "threshold": 0.9},
    ]
    typed = ht.expand_param_grid({"depth": [np.int64(3)], "flag": [True], "c": [0.5]})
    assert typed == [{"c": 0.5, "depth": 3, "flag": True}]
    assert type(typed[0]["depth"]) is int and type(typed[0]["flag"]) is bool


def test_grid_search_with_purged_kfold_and_embargo():
    # Rust: test_grid_search_with_purged_kfold_and_embargo
    n = 120
    x = (np.arange(n) / (n - 1.0)).reshape(-1, 1)
    y = (x[:, 0] >= 0.7).astype(float)
    w = np.where(y == 1.0, 4.0, 1.0)
    t0, t1 = minute_spans(n)
    result = ht.purged_search(
        ThresholdClassifier, ht.expand_param_grid(GRID), x, y, t0, t1,
        n_splits=4, pct_embargo=0.02, scoring="neg_log_loss", sample_weight=w,
    )
    assert len(result["trials"]) == 6
    assert math.isfinite(result["best_score"])
    assert result["best_params"]["threshold"] == pytest.approx(0.7, abs=1e-9)
    for trial in result["trials"]:
        assert len(trial["fold_scores"]) == 4
        assert trial["mean_score"] == pytest.approx(sum(trial["fold_scores"]) / 4)


# `sample_param_sets(SPACE, 12, 42)`: the first two draws, pinned in the Rust test
# `test_sample_param_sets_are_the_randomized_search_candidates`.
SPACE = {"threshold": ("uniform", 0.45, 0.85), "sharpness": ("log_uniform", 1e-1, 2e1)}
RUST_FIRST_DRAW = {"sharpness": 1.6278875156462431, "threshold": 0.6670900839612576}
RUST_SECOND_DRAW = {"sharpness": 2.914239794850492, "threshold": 0.6123607032923106}


def test_randomized_search_seeded_deterministic_and_log_uniform():
    # Rust: test_randomized_search_seeded_deterministic_and_log_uniform
    n = 90
    x = (np.arange(n) / (n - 1.0)).reshape(-1, 1)
    y = (x[:, 0] >= 0.65).astype(float)
    t0, t1 = minute_spans(n)
    draws = ht.sample_param_sets(SPACE, 12, 42)
    assert draws == ht.sample_param_sets(SPACE, 12, 42)
    assert draws != ht.sample_param_sets(SPACE, 12, 43)
    assert len(draws) == 12
    for d in draws:
        assert 0.45 <= d["threshold"] < 0.85
        assert 1e-1 <= d["sharpness"] <= 2e1

    def run():
        return ht.purged_search(
            ThresholdClassifier, draws, x, y, t0, t1,
            n_splits=3, pct_embargo=0.01, scoring="balanced_accuracy",
        )

    first, second = run(), run()
    assert first == second
    assert len(first["trials"]) == 12


def test_sample_param_sets_match_the_rust_pinned_draws():
    draws = ht.sample_param_sets(SPACE, 12, 42)
    assert draws[0] == RUST_FIRST_DRAW
    assert draws[1] == RUST_SECOND_DRAW


def test_sample_param_sets_distributions():
    draws = ht.sample_param_sets(
        {"depth": ("int", 2, 4), "kernel": ("choice", [1, 2, 3]), "flag": ("choice", [True, False])},
        200,
        7,
    )
    assert {d["depth"] for d in draws} == {2, 3, 4}
    assert {d["kernel"] for d in draws} == {1, 2, 3}
    assert {d["flag"] for d in draws} == {True, False}


def test_scoring_layer_handles_imbalance_weighted_neg_log_loss_and_metrics():
    # Rust: test_scoring_layer_handles_imbalance_weighted_neg_log_loss_and_metrics
    y = np.array([0.0] * 95 + [1.0] * 5)
    probs = np.full(100, 0.1)
    accuracy = ht.classification_score(y, probs, scoring="accuracy")
    balanced = ht.classification_score(y, probs, scoring="balanced_accuracy")
    unweighted = ht.classification_score(y, probs, scoring="neg_log_loss")
    weighted = ht.classification_score(y, probs, np.where(y == 1.0, 20.0, 1.0), "neg_log_loss")
    assert accuracy == pytest.approx(0.95)
    assert balanced == pytest.approx(0.5)
    assert balanced < accuracy
    assert weighted < unweighted
    assert unweighted == pytest.approx(0.95 * math.log(0.9) + 0.05 * math.log(0.1))


def test_invalid_input_raises_value_error():
    with pytest.raises(ValueError, match="cannot be empty"):
        ht.expand_param_grid({"a": []})
    with pytest.raises(ValueError, match="int, float or bool"):
        ht.expand_param_grid({"a": ["x"]})
    with pytest.raises(ValueError, match="expected"):
        ht.sample_param_sets({"a": ("normal", 0.0, 1.0)}, 3, 0)
    with pytest.raises(ValueError, match="log-uniform bounds"):
        ht.sample_param_sets({"a": ("log_uniform", 0.0, 1.0)}, 3, 0)
    with pytest.raises(ValueError, match="n_iter"):
        ht.sample_param_sets(SPACE, 0, 0)
    with pytest.raises(ValueError, match="scoring must be one of"):
        ht.classification_score([1.0], [0.5], scoring="f1")
    with pytest.raises(ValueError, match="binary labels"):
        ht.classification_score([2.0], [0.5])
    with pytest.raises(ValueError, match="t0 is required"):
        ht.purged_search(ThresholdClassifier, [{}], np.zeros((4, 1)), np.zeros(4), None, None)
    t0, t1 = minute_spans(10)
    with pytest.raises(ValueError, match="describe 10 samples"):
        ht.purged_search(ThresholdClassifier, [{}], np.zeros((12, 1)), np.zeros(12), t0, t1, n_splits=2)

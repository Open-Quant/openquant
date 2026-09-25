"""Regression tests for the model every feature_diagnostics report is built on (#99).

The built-in model used to fit ridge least squares of the 0/1 label and then pass the
fitted value -- already on the probability scale -- through a sigmoid. Predictions were
squeezed into roughly 0.4-0.8, ``p >= 0.5`` called class 1 for ~93% of rows on balanced
data, and an informative feature scored worse than a coin flip under SFI.
"""

import math
import random

import openquant
import pytest
from openquant.feature_diagnostics import _fit_linear_probability_model, _predict_proba

COIN_FLIP_LOG_LOSS = -math.log(2.0)


def _docs_example(n: int = 1200):
    """The synthetic data on the feature_diagnostics docs page (AFML 8.6 in miniature)."""
    rng = random.Random(8)
    a = [rng.gauss(0, 1) for _ in range(n)]
    b = [rng.gauss(0, 1) for _ in range(n)]
    x = [[a[i], a[i] + rng.gauss(0, 0.2), b[i], rng.gauss(0, 1), rng.gauss(0, 1)] for i in range(n)]
    y = [1.0 if a[i] + 0.6 * b[i] + rng.gauss(0, 0.7) > 0 else 0.0 for i in range(n)]
    ends = [min(i + 5, n - 1) for i in range(n)]
    return x, y, ["a", "a_copy", "b", "noise_1", "noise_2"], ends


def test_predicted_positive_share_tracks_the_base_rate():
    x, y, _, _ = _docs_example()
    prob = _predict_proba(_fit_linear_probability_model(x, y, None), x)

    base_rate = sum(y) / len(y)
    predicted_positive = sum(1 for p in prob if p >= 0.5) / len(prob)
    accuracy = sum(1 for p, t in zip(prob, y) if (p >= 0.5) == (t > 0.5)) / len(y)

    assert abs(predicted_positive - base_rate) < 0.05  # was 0.934 vs 0.503
    assert accuracy > 0.8  # was ~0.57
    assert min(prob) < 0.05 and max(prob) > 0.95  # was 0.377 .. 0.833


def test_recovers_the_coefficients_of_a_logistic_model():
    rng = random.Random(11)
    true_intercept, true_coeffs = -0.4, [1.5, -0.8]
    x, y = [], []
    for _ in range(4000):
        row = [rng.gauss(0, 1), rng.gauss(0, 1)]
        z = true_intercept + sum(c * v for c, v in zip(true_coeffs, row))
        x.append(row)
        y.append(1.0 if rng.random() < 1.0 / (1.0 + math.exp(-z)) else 0.0)

    model = _fit_linear_probability_model(x, y, None)

    assert model.intercept == pytest.approx(true_intercept, abs=0.15)
    assert model.coeffs == pytest.approx(true_coeffs, abs=0.15)


def test_separable_data_gives_confident_finite_predictions():
    x = [[v / 10.0] for v in range(-50, 50) if v != 0]
    y = [1.0 if row[0] > 0 else 0.0 for row in x]

    model = _fit_linear_probability_model(x, y, None)
    prob = _predict_proba(model, x)

    assert all(math.isfinite(c) for c in [model.intercept, *model.coeffs])
    clipped = [min(max(p, 1e-15), 1 - 1e-15) for p in prob]
    log_loss = -sum(t * math.log(p) + (1 - t) * math.log(1 - p) for t, p in zip(y, clipped)) / len(
        y
    )
    assert log_loss < 0.05  # was 0.56 on the old model
    assert all((p >= 0.5) == (t > 0.5) for p, t in zip(prob, y))


def test_integer_sample_weights_equal_duplicated_rows():
    rng = random.Random(3)
    x = [[rng.gauss(0, 1), rng.gauss(0, 1)] for _ in range(200)]
    y = [1.0 if r[0] - 0.5 * r[1] + rng.gauss(0, 1) > 0 else 0.0 for r in x]
    w = [float(1 + i % 3) for i in range(len(x))]

    weighted = _fit_linear_probability_model(x, y, w)
    dup_x = [row for row, k in zip(x, w) for _ in range(int(k))]
    dup_y = [t for t, k in zip(y, w) for _ in range(int(k))]
    duplicated = _fit_linear_probability_model(dup_x, dup_y, None)

    assert weighted.intercept == pytest.approx(duplicated.intercept, abs=1e-6)
    assert weighted.coeffs == pytest.approx(duplicated.coeffs, abs=1e-6)


def test_sfi_scores_informative_features_above_a_coin_flip():
    x, y, names, ends = _docs_example()
    out = openquant.feature_diagnostics.sfi_importance(
        x, y, feature_names=names, event_end_indices=ends
    )
    sfi = {r["feature"]: r["mean"] for r in out["records"]}

    # `b` scored -0.698 on the old model: informative, yet worse than a coin flip.
    for informative in ["a", "a_copy", "b"]:
        assert sfi[informative] > COIN_FLIP_LOG_LOSS
    # Noise cannot beat a coin flip by much out of sample.
    for noise in ["noise_1", "noise_2"]:
        assert sfi[noise] < COIN_FLIP_LOG_LOSS + 0.01
        assert sfi[noise] < sfi["b"]


def test_accuracy_scoring_measures_more_than_class_balance():
    x, y, names, ends = _docs_example()
    out = openquant.feature_diagnostics.mda_importance(
        x, y, feature_names=names, event_end_indices=ends, scoring="accuracy"
    )
    assert out["cv"]["mean_base_score"] > 0.8  # was ~0.57 with a 50.3% base rate

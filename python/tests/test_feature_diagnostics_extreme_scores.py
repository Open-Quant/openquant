import math
from fractions import Fraction

import pytest
from openquant.feature_diagnostics import _score


def reference(y, prob, weights, metric):
    w = list(map(Fraction, weights))
    if metric == "accuracy":
        return float(
            sum(
                (wi for yi, pi, wi in zip(y, prob, w) if abs(yi - float(pi >= 0.5)) < 1e-12),
                Fraction(),
            )
            / sum(w)
        )
    if metric == "neg_log_loss":
        losses = []
        for yi, pi in zip(y, prob):
            pi = min(max(pi, 1e-15), 1.0 - 1e-15)
            losses.append(Fraction(-(yi * math.log(pi) + (1.0 - yi) * math.log(1.0 - pi))))
        return -float(sum((wi * li for wi, li in zip(w, losses)), Fraction()) / sum(w))
    tp = sum((wi for yi, pi, wi in zip(y, prob, w) if yi > 0.5 and pi >= 0.5), Fraction())
    fp = sum((wi for yi, pi, wi in zip(y, prob, w) if yi <= 0.5 and pi >= 0.5), Fraction())
    fn = sum((wi for yi, pi, wi in zip(y, prob, w) if yi > 0.5 and pi < 0.5), Fraction())
    return float(2 * tp / (2 * tp + fp + fn)) if 2 * tp + fp + fn else 0.0


@pytest.mark.parametrize("metric", ["accuracy", "neg_log_loss", "f1"])
@pytest.mark.parametrize(
    ("y", "prob", "weights"),
    [
        ([1.0, 0.0], [0.9, 0.9], [1e308, 1e308]),
        ([1.0, 1.0, 0.0], [0.9, 0.9, 0.9], [1e308, 1e308, 1e308]),
        ([1.0, 0.0], [0.9, 0.9], [math.ulp(0.0), math.ulp(0.0)]),
        ([1.0, 0.0, 1.0], [0.9, 0.9, 0.1], [1e-308, 1.0, 1.0]),
        ([0.0, 1.0], [0.1, 0.9], [1e308, math.ulp(0.0)]),
    ],
)
def test_extreme_scores_match_exact_weighted_reference(y, prob, weights, metric):
    expected = reference(y, prob, weights, metric)
    actual = _score(y, prob, metric, weights)
    assert math.isfinite(actual)
    assert actual == pytest.approx(expected, rel=2e-15, abs=math.ulp(0.0))


def test_subnormal_logloss_retains_nonzero_mean():
    assert _score([1.0, 0.0], [0.9, 0.1], "neg_log_loss", [math.ulp(0.0)] * 2) == reference(
        [1.0, 0.0], [0.9, 0.1], [math.ulp(0.0)] * 2, "neg_log_loss"
    )

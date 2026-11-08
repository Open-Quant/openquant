"""Every tuning entry point requires exact binary labels before fitting/scoring."""

import numpy as np
import pytest
from openquant import hyperparameter_tuning as ht

NEAR_LABELS = [
    np.nextafter(0.0, 1.0),
    -np.nextafter(0.0, 1.0),
    1e-12,
    -1e-12,
    np.nextafter(1.0, 0.0),
    np.nextafter(1.0, 2.0),
    1.0 - 5e-13,
    1.0 + 5e-13,
]


@pytest.mark.parametrize("label", NEAR_LABELS)
@pytest.mark.parametrize("weight", [0.0, 1.0])
@pytest.mark.parametrize("scoring", ["accuracy", "balanced_accuracy", "neg_log_loss"])
def test_scorer_rejects_near_binary_labels(label, weight, scoring):
    with pytest.raises(ValueError, match="binary"):
        ht.classification_score([label, 1.0], [0.1, 0.9], [weight, 1.0], scoring)


@pytest.mark.parametrize("label", NEAR_LABELS)
@pytest.mark.parametrize("weight", [0.0, 1.0])
def test_search_rejects_near_binary_labels_before_factory(label, weight):
    x = np.zeros((12, 1))
    y = np.zeros(12)
    y[-1] = label
    weights = np.ones(12)
    weights[-1] = weight
    t0 = np.datetime64("2024-01-01", "ns") + np.arange(12) * np.timedelta64(1, "D")

    def factory(params):
        pytest.fail("near-binary labels reached estimator factory")

    with pytest.raises(ValueError, match="binary"):
        ht.purged_search(factory, [{}], x, y, t0, t0, n_splits=3, sample_weight=weights)


@pytest.mark.parametrize("zero", [0.0, -0.0])
@pytest.mark.parametrize("scoring", ["accuracy", "balanced_accuracy", "neg_log_loss"])
def test_exact_binary_and_signed_zero_controls(zero, scoring):
    actual = ht.classification_score([zero, 1.0], [0.1, 0.9], scoring=scoring)
    expected = np.log(0.9) if scoring == "neg_log_loss" else 1.0
    assert actual == pytest.approx(expected, abs=1e-15)

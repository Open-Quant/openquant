"""Keep probability and binary-class validation ahead of narrowing casts."""

from decimal import Decimal
from fractions import Fraction

import numpy as np
import pytest
from openquant import evaluation
from openquant._estimator import _positive_proba


@pytest.mark.parametrize(
    "value",
    [
        Decimal("1.0000000000000000000000000000001"),
        Decimal("-1e-400"),
        Fraction(2**80 + 1, 2**80),
        Fraction(-1, 10**400),
        np.complex64(0.5 + 1j),
    ],
)
def test_meta_predictions_reject_invalid_original_probability(value):
    with pytest.raises(ValueError, match="meta_predictions"):
        evaluation.meta_label_metrics([1], [value])


@pytest.mark.parametrize(
    "value",
    [
        Decimal("1.0000000000000000000000000000001"),
        Fraction(2**80 + 1, 2**80),
        np.complex64(0.5 + 1j),
    ],
)
def test_meta_threshold_rejects_invalid_original_probability(value):
    with pytest.raises(ValueError, match="threshold"):
        evaluation.meta_label_metrics([1], [0.5], threshold=value)


@pytest.mark.parametrize("value", [0j, 1 + 0j, np.complex64(1 + 0j)])
def test_meta_labels_reject_complex_binary_lookalikes(value):
    with pytest.raises(ValueError, match="meta_labels"):
        evaluation.meta_label_metrics([value], [0.5])


@pytest.mark.parametrize(
    "classes",
    [
        np.array([0, 1], dtype=np.complex64),
        np.array([0, 1 + 0j], dtype=object),
        np.array([np.complex64(0), 1], dtype=object),
        np.array([Decimal("sNaN"), 1], dtype=object),
    ],
)
def test_estimator_classes_require_real_binary_labels(classes):
    class Estimator:
        def __init__(self, labels):
            self.classes_ = labels

        def predict_proba(self, features):
            return np.tile([0.85, 0.15], (len(features), 1))

    with pytest.raises(ValueError, match="classes_"):
        _positive_proba(Estimator(classes), np.zeros((2, 1)))


def test_meta_accepts_valid_exact_and_ordinary_inputs():
    assert evaluation.meta_label_metrics(
        [0, np.int64(1), False, 1.0], [Decimal("0.25"), Fraction(3, 4), 0, np.float32(1)]
    ) == evaluation.meta_label_metrics([0, 1, 0, 1], [0.25, 0.75, 0, 1])

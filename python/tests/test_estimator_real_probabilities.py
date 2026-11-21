"""Probability checks must precede lossy conversion into binary64."""

from decimal import Decimal
from fractions import Fraction

import numpy as np
import pytest
from openquant._estimator import _positive_proba


class Model:
    def __init__(self, predictions):
        self.predictions = predictions

    def predict_proba(self, x):
        return self.predictions


@pytest.mark.parametrize(
    "predictions",
    [
        np.array([0.2 + 1j, 0.3, 0.5]),
        np.array([complex(0.2, np.nan), 0.3, 0.5]),
        np.array([[0.8, 0.2], [0.7, 0.3j], [0.5, 0.5]]),
        np.array([0.2, 0.3, 0.5], dtype=complex),
        np.array([np.complex64(0.2 + 1j), 0.3, 0.5], dtype=object),
    ],
)
def test_reject_complex_probabilities_before_cast(predictions):
    with pytest.raises(ValueError, match="predict_proba.*real"):
        _positive_proba(Model(predictions), np.arange(3).reshape(-1, 1))


def test_finite_real_probabilities_still_convert():
    p = _positive_proba(Model(np.array([0.2, 0.3, 0.5], dtype=np.float32)), np.zeros((3, 1)))
    assert p.dtype == np.float64
    assert p == pytest.approx([0.2, 0.3, 0.5])


@pytest.mark.skipif(
    bool(np.finfo(np.longdouble).eps >= np.finfo(np.float64).eps),
    reason="long double has no extra precision on this platform",
)
@pytest.mark.parametrize("edge", ["above-one", "below-zero"])
def test_reject_out_of_range_before_extended_precision_rounding(edge):
    value = (
        np.nextafter(np.longdouble(1.0), np.longdouble(2.0))
        if edge == "above-one"
        else np.nextafter(np.longdouble(0.0), np.longdouble(-1.0))
    )
    with pytest.raises(ValueError, match="predict_proba.*probabilities"):
        _positive_proba(Model(np.array([value], dtype=np.longdouble)), np.zeros((1, 1)))


@pytest.mark.parametrize(
    "value",
    [
        Decimal("1.0000000000000000000000000000001"),
        Decimal("-1e-400"),
        Decimal("NaN"),
        Decimal("sNaN"),
        Fraction(2**80 + 1, 2**80),
        Fraction(-1, 10**400),
        10**400,
    ],
)
def test_object_numeric_probability_range_checked_before_rounding(value):
    predictions = np.array([0.2, value, 0.7], dtype=object)
    with pytest.raises(ValueError, match="predict_proba.*probabilities"):
        _positive_proba(Model(predictions), np.zeros((3, 1)))


def test_exact_numeric_probability_objects_within_range_convert():
    predictions = np.array([Decimal("0.2"), Fraction(1, 3), np.longdouble("0.7")], dtype=object)
    result = _positive_proba(Model(predictions), np.zeros((3, 1)))
    assert result.dtype == np.float64
    assert result == pytest.approx([0.2, 1 / 3, 0.7])


def test_object_longdouble_probability_range_checked_before_rounding():
    above_one = np.nextafter(np.longdouble(1), np.longdouble(2))
    predictions = np.array([above_one], dtype=object)
    with pytest.raises(ValueError, match="predict_proba.*probabilities"):
        _positive_proba(Model(predictions), np.zeros((1, 1)))

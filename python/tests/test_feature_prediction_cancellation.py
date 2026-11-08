from fractions import Fraction
from math import exp

import pytest
from openquant import feature_diagnostics as diagnostics


@pytest.mark.parametrize(
    ("intercept", "coefficients", "row"),
    [
        (0.0, [1e16 + 2.0, -1e16], [1.0 - 2**-52, 1.0]),
        (0.0, [1.1, -1.0], [1e16, 1.1e16]),
        (0.0, [1.1, -1.0], [1e6, 1.1e6]),
        (64.0, [1.1, -1.0], [2e18, 2.2000000000000003e18]),
        (-64.0, [-1.1, 1.0], [2e18, 2.2000000000000003e18]),
    ],
)
def test_finite_product_rounding_does_not_hide_prediction_signal(intercept, coefficients, row):
    # Reference the stored binary64 model exactly. Summing rounded products
    # with fsum cannot restore information already lost in multiplication.
    exact_logit = Fraction(intercept) + sum(
        (Fraction(c) * Fraction(x) for c, x in zip(coefficients, row)), Fraction()
    )
    expected = 1.0 / (1.0 + exp(-float(exact_logit)))
    model = diagnostics._LinearModel(coeffs=coefficients, intercept=intercept)
    assert diagnostics._predict_proba(model, [row]) == pytest.approx([expected], abs=1e-15)

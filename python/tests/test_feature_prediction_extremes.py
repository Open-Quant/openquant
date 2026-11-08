import math

import pytest
from openquant import feature_diagnostics as diagnostics


@pytest.mark.parametrize("intercept", [0.0, 1.0, -1.0])
def test_cancelling_overflowed_products_preserve_finite_logit(intercept):
    model = diagnostics._LinearModel(coeffs=[1e308, 1e308], intercept=intercept)
    # Exact products cancel; only the representable intercept remains.
    expected = 1.0 / (1.0 + math.exp(-intercept))
    assert diagnostics._predict_proba(model, [[2.0, -2.0]]) == pytest.approx([expected])


@pytest.mark.parametrize("sign", [-1.0, 1.0])
def test_true_out_of_range_logits_keep_saturated_probability(sign):
    model = diagnostics._LinearModel(coeffs=[sign * 1e308], intercept=0.0)
    assert diagnostics._predict_proba(model, [[2.0]]) == [float(sign > 0)]


def test_nonfinite_model_inputs_keep_legacy_nan_result():
    model = diagnostics._LinearModel(coeffs=[math.inf, -math.inf], intercept=0.0)
    assert math.isnan(diagnostics._predict_proba(model, [[1.0, 1.0]])[0])

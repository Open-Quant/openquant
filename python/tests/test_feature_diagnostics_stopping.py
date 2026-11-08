from decimal import Decimal, localcontext

import pytest
from openquant import feature_diagnostics as fd


def test_negligible_weight_outliers_cannot_mask_other_logit_steps():
    # Independent symmetric cells: the first coefficient is about ln(3)*1e150;
    # the second is separable and should use all 50 Newton iterations.
    x = [[-1e-150, 0.0]] * 4 + [[1e-150, 0.0]] * 4 + [[0.0, -1.0], [0.0, 1.0]]
    y = [0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0]
    weights = [1e308] * len(x)
    control = fd._fit_linear_probability_model(x, y, weights)
    # These correctly classified observations add negligible loss but enormous
    # logits; a global maximum logit must not relax tolerance on other rows.
    x += [[-1.0, 0.0], [1.0, 0.0]]
    y += [0.0, 1.0]
    weights += [1e-308, 1e-308]
    fitted = fd._fit_linear_probability_model(x, y, weights)
    with localcontext() as ctx:
        ctx.prec = 100
        beta = Decimal(0)
        penalty = Decimal.from_float(1e-3) / (2 * Decimal.from_float(1e308))
        for _ in range(50):
            q = (-beta).exp()
            p = q / (1 + q)
            beta -= (-p + penalty * beta) / (p * (1 - p) + penalty)
        expected = float(beta)
    assert control.coeffs[1] == pytest.approx(expected, abs=1e-8)
    assert fitted.coeffs[1] == pytest.approx(expected, abs=1e-8)
    assert fitted.coeffs[0] == pytest.approx(control.coeffs[0], rel=1e-12)

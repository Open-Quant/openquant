"""Analytic sensitivities must survive changes of payout units and frequency scale."""

import math
from decimal import Decimal, localcontext

import pytest
from openquant import evaluation as ev


def reference(p, n, up, down):
    with localcontext() as ctx:
        ctx.prec = 100
        p, n, up, down = map(Decimal.from_float, (p, n, up, down))
        d = up - down
        v = p * (1 - p)
        theta = (n / v).sqrt()
        sr = theta * (p + down / d)
        return {
            "d_sharpe_d_precision": n.sqrt() * (p * up - (1 - p) * down) / (2 * d * v * v.sqrt()),
            "d_sharpe_d_frequency": sr / (2 * n),
            "d_sharpe_d_pi_plus": -theta * down / d**2,
            "d_sharpe_d_pi_minus": theta * up / d**2,
        }


@pytest.mark.parametrize(
    "p,n,up,down",
    [
        (0.6, 100.0, 1e-200, -1e-200),
        (0.6, 100.0, 5e-324, -5e-324),
        (5e-324, 1.0, 2.0, -1.0),
        (0.6, 100.0, 1e200, -1e200),
        (0.6, 100.0, 1e308, -1e308),
        (0.6, 1e308, 2.0, -1.0),
        (1e-200, 1e-300, 2.0, -1.0),
        (1e-250, 1e-300, 2.0, -1.0),
        (0.6, 100.0, 2.0, -1.0),
        (math.nextafter(math.nextafter(1 / 3, 1.0), 1.0), 100.0, 2.0, 1.0),
        (math.nextafter(math.nextafter(1 / 3, 1.0), 1.0), 100.0, 2.0, -1.0),
    ],
)
def test_sensitivities_match_high_precision_reference(p, n, up, down):
    result = ev.strategy_risk_sensitivity(p, n, up, down)
    for key, expected in reference(p, n, up, down).items():
        assert result[key] == pytest.approx(float(expected), rel=2e-14, abs=0.0), key


def test_extreme_sensitivity_ignores_callers_decimal_context():
    expected = reference(0.6, 100.0, 1e-200, -1e-200)
    with localcontext() as ctx:
        ctx.prec = 3
        result = ev.strategy_risk_sensitivity(0.6, 100.0, 1e-200, -1e-200)
    for key, value in expected.items():
        assert result[key] == pytest.approx(float(value), rel=2e-14, abs=0.0)


@pytest.mark.parametrize("scale", [1e-200, 1e200])
@pytest.mark.parametrize("p", [0.01, 0.4, 0.99])
@pytest.mark.parametrize("up,down", [(2.0, -1.0), (2.0, 1.0), (-1.0, -2.0)])
def test_payout_units_preserve_dimensionless_sensitivities(scale, p, up, down):
    baseline = ev.strategy_risk_sensitivity(p, 100.0, up, down)
    scaled = ev.strategy_risk_sensitivity(p, 100.0, up * scale, down * scale)
    for key in ("sharpe", "d_sharpe_d_precision", "d_sharpe_d_frequency"):
        assert scaled[key] == pytest.approx(baseline[key], rel=3e-14, abs=0.0)
    for key in ("d_sharpe_d_pi_plus", "d_sharpe_d_pi_minus"):
        assert scaled[key] * scale == pytest.approx(baseline[key], rel=3e-14, abs=0.0)


@pytest.mark.parametrize("exponent", [240, 400, 700])
def test_sensitivities_preserve_residuals_beyond_decimal_precision(exponent):
    p = 2.0**-exponent
    # For n=up=1 and down=-p, the frequency derivative is
    # p**(3/2) / (2*(1+p)*sqrt(1-p)). The correction to the power
    # of two is far below a binary64 ULP at all these exponents.
    negative = ev.strategy_risk_sensitivity(p, 1.0, 1.0, -p)
    assert negative["d_sharpe_d_frequency"] == 2.0 ** (-3 * exponent // 2 - 1)


@pytest.mark.parametrize("exponent", [240, 400, 700])
def test_precision_sensitivity_preserves_tiny_stationary_point_residual(exponent):
    p = 2.0**-exponent
    # For down=+p, the precision derivative instead simplifies to
    # sqrt(p) / (2*(1-p)**(5/2)), also rounding to a power of two.
    positive = ev.strategy_risk_sensitivity(p, 1.0, 1.0, p)
    assert positive["d_sharpe_d_precision"] == 2.0 ** (-exponent // 2 - 1)

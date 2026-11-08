"""Public bindings recover ill-conditioned signed trade impact regressions."""

import math

import openquant
import pytest


@pytest.mark.parametrize(
    "name", ["get_trades_based_kyle_lambda", "get_trades_based_hasbrouck_lambda"]
)
def test_normal_accumulations_can_still_require_cancellation_recovery(name):
    calculate = getattr(openquant.microstructural, name)
    # The exact signed products are 1e16, 3, -1e16, whose sum is three.
    assert calculate([1e16, -3.0, -1e16], [1.0] * 3, [1.0, -1.0, 1.0]) == 1.0


@pytest.mark.parametrize("scale", [1e200, 1e-200])
def test_kyle_product_extremes_retain_finite_slope(scale):
    assert (
        openquant.microstructural.get_trades_based_kyle_lambda(
            [scale, -scale / 2], [scale] * 2, [1.0] * 2
        )
        == 0.25
    )


def test_hasbrouck_partial_product_underflow():
    assert (
        openquant.microstructural.get_trades_based_hasbrouck_lambda(
            [math.ldexp(1.0, -537), -math.ldexp(1.0, -538)],
            [math.ulp(0.0)] * 2,
            [1.0] * 2,
        )
        == 0.25
    )


def test_kyle_recovery_preserves_negative_zero():
    value = openquant.microstructural.get_trades_based_kyle_lambda(
        [-0.0] * 2, [1e-200] * 2, [1.0] * 2
    )
    assert value == 0.0
    assert math.copysign(1.0, value) == -1.0

"""Positive finite prices must not lose log returns to ratio range limits."""

import math
from decimal import Decimal, localcontext

import pytest
from openquant import sample_weights

DATES = [f"2024-01-0{i} 00:00:00" for i in range(1, 4)]
EVENTS = [(t, t, 1.0) for t in DATES[1:]]


@pytest.mark.parametrize(
    "prices",
    [
        [1e-300, 1e300, 1e-300],
        [1e300, 1e-300, 1e300],
        [1.5, math.ulp(0.0), 2.0],
        [math.ulp(0.0), 1.5, 2.0],
    ],
)
def test_extreme_finite_price_ratios_match_decimal_return_attribution(prices):
    with localcontext() as ctx:
        ctx.prec = 100
        log_prices = [Decimal.from_float(x).ln() for x in prices]
        raw = [abs(b - a) for a, b in zip(log_prices, log_prices[1:])]
        expected = [float(w * 2 / sum(raw)) for w in raw]
    actual = [w for _, w in sample_weights.get_weights_by_return(EVENTS, DATES, prices)]
    assert all(math.isfinite(w) for w in actual)
    assert actual == pytest.approx(expected, rel=2e-14, abs=1e-15)


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_price_behavior_is_not_silently_reinterpreted(bad):
    actual = sample_weights.get_weights_by_return(EVENTS, DATES, [1.0, bad, 2.0])
    assert any(not math.isfinite(w) for _, w in actual)

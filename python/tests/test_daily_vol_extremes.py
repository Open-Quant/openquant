"""Extreme finite returns must expose overflow rather than claim zero risk."""

import math
import sys
from datetime import datetime, timedelta

import pytest
from openquant import volatility


def _estimate(returns, span):
    anchor = datetime(2024, 1, 1)
    timestamps = [anchor] + [anchor + timedelta(days=2, seconds=i) for i in range(len(returns))]
    stamps = [stamp.strftime("%Y-%m-%d %H:%M:%S") for stamp in timestamps]
    # Every later bar uses the same eligible unit-price anchor.
    out = volatility.get_daily_vol(stamps, [1.0] + [ret + 1.0 for ret in returns], span)
    assert [stamp for stamp, _ in out] == stamps[1:]
    return [value for _, value in out]


@pytest.mark.parametrize(
    "level",
    [1e153, math.sqrt(sys.float_info.max), math.nextafter(math.sqrt(sys.float_info.max), math.inf)],
)
def test_initial_square_overflow_preserves_finite_variance(level):
    # Independent closed-form weighted sums; no overflowing square is needed.
    expected = [math.nan, 0.0, level * math.sqrt(3 / 7), level * math.sqrt(9 / 35)]
    assert _estimate([level, level, 0.0, 0.0], 3) == pytest.approx(
        expected, nan_ok=True, rel=1e-14, abs=1e-12
    )


@pytest.mark.parametrize("level", [1e155, 1e200, 1e308])
def test_variance_overflow_is_infinity_not_zero(level):
    # pandas 2.2.3 golden values: this f64 recurrence cannot represent variance.
    assert _estimate([level, level, 0.0, 0.0], 3) == pytest.approx(
        [math.nan, 0.0, math.inf, math.inf], nan_ok=True
    )


@pytest.mark.parametrize(("level", "span"), [(1e150, 4), (1e308, 3), (sys.float_info.max, 100)])
def test_identical_extreme_returns_have_exactly_zero_variance(level, span):
    actual = _estimate([level] * 8, span)
    assert math.isnan(actual[0])
    assert actual[1:] == [0.0] * 7


def test_undefined_overflow_arithmetic_stays_nan():
    returns = [8e307, 9e307, 1.25e308, 1.1111111111111112e308, 0.0, 0.0]
    assert _estimate(returns, 3) == pytest.approx(
        [math.nan, math.inf, math.inf, math.nan, math.nan, math.nan], nan_ok=True
    )


def test_first_finite_return_after_missing_values_is_initialized_safely():
    assert _estimate([math.nan, math.nan, 1e155, 1e155, 0.0, 0.0], 3) == pytest.approx(
        [math.nan, math.nan, math.nan, 0.0, math.inf, math.inf], nan_ok=True
    )


def test_weight_underflow_does_not_reset_overflowed_variance():
    actual = _estimate([1e200, 1e200, 0.0] + [math.nan] * 1100 + [0.0, 1.0], 3)
    # Resetting when a computed weight underflows would falsely report 0.707;
    # the high-precision weighted std is still about 1.439e34. pandas stays NaN.
    assert all(math.isnan(value) for value in actual[-2:])

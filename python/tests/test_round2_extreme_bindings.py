"""Exercise numerical recoveries through the installed native Python module."""

from fractions import Fraction

import openquant
import pytest


def test_symmetric_inverse_preserves_large_finite_target():
    assert openquant.strategy_risk.implied_precision_symmetric(1e154, 1e308) == pytest.approx(
        0.8535533905932737, abs=2e-15
    )


def test_asymmetric_inverse_preserves_large_payouts():
    assert openquant.strategy_risk.implied_frequency_asymmetric(
        0.75, 2.0, 1e308, -1e308
    ) == pytest.approx(12.0, rel=2e-14)


@pytest.mark.parametrize(
    ("returns", "volumes"),
    [
        ([1e200, 2e200], [1e200, 2e200]),
        ([1e-200, 2e-200], [1e-200, 2e-200]),
        ([7.228216483043474e52], [4.0208288850127243e-256]),
        ([1.0, 1.0], [1e308, 1e308]),
    ],
)
def test_amihud_extremes_match_exact_binary_input_reference(returns, volumes):
    numerator = sum(Fraction(v) * abs(Fraction(r)) for r, v in zip(returns, volumes, strict=True))
    denominator = sum(Fraction(v) ** 2 for v in volumes)
    expected = float(numerator / denominator)
    actual = openquant.microstructural.get_trades_based_amihud_lambda(returns, volumes)
    assert actual == expected

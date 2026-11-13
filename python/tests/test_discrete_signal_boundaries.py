"""Regression coverage for finite discretisation quotient overflow."""

import math
from fractions import Fraction

import openquant
import pytest


def _exact_discrete_signal(signal, step):
    quotient = Fraction(signal) / Fraction(step)
    magnitude = abs(quotient)
    rounded, remainder = divmod(magnitude.numerator, magnitude.denominator)
    rounded += 2 * remainder >= magnitude.denominator
    if quotient < 0:
        rounded = -rounded
    return float(max(Fraction(-1), min(Fraction(1), rounded * Fraction(step))))


@pytest.mark.parametrize("step", [1e-320, math.ulp(0.0), 1e-309, 0.25])
def test_discrete_signal_matches_exact_round_away_from_zero(step):
    signals = [0.125, -0.125, 0.5, -0.5, 2.0, -2.0]
    expected = [_exact_discrete_signal(signal, step) for signal in signals]
    assert openquant.bet_sizing.discrete_signal(signals, step) == expected


@pytest.mark.parametrize("step", [0.0, -0.0, -0.1, -math.inf])
def test_discrete_signal_nonpositive_step_preserves_input(step):
    signals = [-0.0, 0.5, -0.5, 2.0, -2.0, math.inf, -math.inf, math.nan]
    actual = openquant.bet_sizing.discrete_signal(signals, step)
    assert [x.hex() for x in actual] == [x.hex() for x in signals]


@pytest.mark.parametrize("step", [math.inf, math.nan])
def test_discrete_signal_nonfinite_positive_step_propagates_nan(step):
    actual = openquant.bet_sizing.discrete_signal([0.0, 0.5, math.inf, math.nan], step)
    assert all(math.isnan(x) for x in actual)


def test_discrete_signal_nonfinite_signal_preserves_behavior():
    actual = openquant.bet_sizing.discrete_signal([math.inf, -math.inf, math.nan], 1e-320)
    assert actual[:2] == [1.0, -1.0]
    assert math.isnan(actual[2])

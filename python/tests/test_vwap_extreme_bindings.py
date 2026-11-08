"""Finite VWAP ratios survive overflowing sums through the public binding."""

import math
import sys
from fractions import Fraction

import openquant
import pytest


@pytest.mark.parametrize(
    ("dollar_volume", "volume"),
    [
        ([1e308, 1e308], [1e308, 1e308]),
        ([1e308, 1e308], [1.0, 1.0]),
        ([1.0, 1.0], [1e308, 1e308]),
        ([sys.float_info.max] * 2, [1.0, 1.0]),
    ],
)
def test_vwap_matches_exact_ratio_when_individual_sums_overflow(dollar_volume, volume):
    expected = float(sum(map(Fraction, dollar_volume)) / sum(map(Fraction, volume)))
    assert openquant.microstructural.vwap(dollar_volume, volume) == expected


def test_vwap_recovers_range_boundary_even_when_both_rounded_sums_are_normal():
    dollar_volume = [sys.float_info.max, math.ldexp(1.0, 969)]
    volume = [0.5, math.nextafter(0.5, 0.0)]
    exact = sum(map(Fraction, dollar_volume)) / sum(map(Fraction, volume))
    with pytest.raises(OverflowError):
        float(exact)
    assert openquant.microstructural.vwap(dollar_volume, volume) == math.inf


def test_vwap_preserves_negative_zero_when_denominator_overflows():
    value = openquant.microstructural.vwap([-0.0, -0.0], [sys.float_info.max] * 2)
    assert value == 0.0
    assert math.copysign(1.0, value) == -1.0

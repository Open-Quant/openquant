"""Resolved label returns recover arithmetic independently of touch selection."""

import math
import sys
from fractions import Fraction

import openquant
import pytest

STAMPS = ["2024-01-01 00:00:00", "2024-01-02 00:00:00"]


@pytest.mark.parametrize(
    ("start", "end", "side"),
    [
        (1e-308, 3.0, 1e-308),
        (1e-308, 3.0, -1e-308),
        (1e-308, 3.0, 0.0),
        (1e-308, 3.0, -0.0),
        (math.nextafter(sys.float_info.max, 0.0), sys.float_info.max, None),
        (math.nextafter(sys.float_info.max, 0.0), sys.float_info.max, 1e308),
    ],
)
def test_resolved_returns_match_exact_input_reference(start, end, side):
    exact = (Fraction(end) / Fraction(start) - 1) * Fraction(1.0 if side is None else side)
    expected = float(exact)
    if expected == 0.0 and side is not None:
        expected = math.copysign(0.0, math.copysign(1.0, end - start) * math.copysign(1.0, side))
    event = (STAMPS[0], STAMPS[1], 0.1, side, 0.0, 0.0)
    rows = openquant.labeling.get_bins([event], STAMPS, [start, end])
    assert len(rows) == 1
    assert rows[0][1].hex() == expected.hex()
    expected_label = int(expected > 0.0) if side is not None else (expected > 0) - (expected < 0)
    assert rows[0][3] == expected_label

"""CUSUM filter bindings: the scalar threshold and the per-bar threshold (issue #187)."""

import math
import random

import numpy as np
import pytest
from openquant import filters

# Log returns of CLOSE, bar by bar (bar 0 has none):
#   r1 = ln(101/100)    = +0.009950
#   r2 = ln(102/101)    = +0.009852
#   r3 = ln(101.5/102)  = -0.004914
#   r4 = ln(100/101.5)  = -0.014889
#   r5 = ln(99/100)     = -0.010050
CLOSE = [100.0, 101.0, 102.0, 101.5, 100.0, 99.0]
STAMPS = [f"2024-01-02 09:3{i}:00" for i in range(len(CLOSE))]


def _reference(close, thresholds):
    """AFML Snippet 2.4 on log returns, one threshold per bar."""
    events, s_pos, s_neg = [], 0.0, 0.0
    for t in range(1, len(close)):
        r = math.log(close[t] / close[t - 1])
        s_pos, s_neg = max(0.0, s_pos + r), min(0.0, s_neg + r)
        if s_neg < -thresholds[t]:
            s_neg = 0.0
            events.append(t)
        elif s_pos > thresholds[t]:
            s_pos = 0.0
            events.append(t)
    return events


def test_per_bar_threshold_hand_worked():
    # Element 0 is never read, so a volatility estimate's leading NaN is fine there.
    h = [math.nan, 0.02, 0.005, 0.02, 0.01, 0.02]
    # bar 1: S+ = 0.00995 <= 0.02
    # bar 2: S+ = 0.01980 >  0.005 -> event, S+ reset
    # bar 3: S- = -0.00491, not below -0.02
    # bar 4: S- = -0.01980 < -0.01 -> event, S- reset
    # bar 5: S- = -0.01005, not below -0.02
    assert filters.cusum_filter_indices(CLOSE, h) == [2, 4]
    assert filters.cusum_filter_timestamps(CLOSE, STAMPS, h) == [STAMPS[2], STAMPS[4]]
    # Neither scalar reproduces it: the per-bar values are what decide.
    assert filters.cusum_filter_indices(CLOSE, 0.02) == [5]
    assert filters.cusum_filter_indices(CLOSE, 0.005) == [1, 2, 4, 5]
    assert _reference(CLOSE, h) == [2, 4]


def test_constant_per_bar_threshold_matches_scalar():
    rng = random.Random(5)
    close, price = [], 100.0
    for i in range(2_000):
        price *= math.exp(rng.gauss(0, 0.004 if 800 <= i < 1_200 else 0.001))
        close.append(price)
    stamps = [
        f"2024-01-02 {9 + i // 3600:02d}:{i // 60 % 60:02d}:{i % 60:02d}" for i in range(2_000)
    ]
    for h in (0.0, 0.002, 0.01, 0.03):
        scalar = filters.cusum_filter_indices(close, h)
        assert filters.cusum_filter_indices(close, [h] * len(close)) == scalar
        assert filters.cusum_filter_indices(close, np.full(len(close), h)) == scalar
        assert filters.cusum_filter_timestamps(close, stamps, [h] * len(close)) == (
            filters.cusum_filter_timestamps(close, stamps, h)
        )
    assert filters.cusum_filter_indices(close, 0.01)  # the comparison is not vacuous


def test_volatility_scaled_threshold_matches_reference():
    rng = random.Random(9)
    close, price = [], 100.0
    for i in range(1_000):
        price *= math.exp(rng.gauss(0, 0.004 if 400 <= i < 600 else 0.001))
        close.append(price)
    # A trailing 20-bar standard deviation of log returns, doubled: NaN until it is defined.
    rets = np.diff(np.log(close), prepend=np.nan)
    vol = np.array([np.nan if t < 21 else rets[t - 19 : t + 1].std(ddof=1) for t in range(1_000)])
    h = 2.0 * np.where(np.isnan(vol), 0.002, vol)
    h[0] = np.nan
    events = filters.cusum_filter_indices(close, h)
    assert events == _reference(close, h.tolist())
    assert events != filters.cusum_filter_indices(close, 0.004)


def test_scalar_like_inputs_still_read_as_scalars():
    assert filters.cusum_filter_indices(CLOSE, 0.005) == [1, 2, 4, 5]
    assert filters.cusum_filter_indices(CLOSE, np.float64(0.005)) == [1, 2, 4, 5]
    assert filters.cusum_filter_indices(CLOSE, np.array(0.005)) == [1, 2, 4, 5]
    assert filters.cusum_filter_indices(CLOSE, 1) == []


@pytest.mark.parametrize(
    ("threshold", "error", "match"),
    [
        ([0.01] * 5, ValueError, "5 values but close has 6"),
        ([0.01] * 7, ValueError, "7 values but close has 6"),
        ([0.01, 0.01, math.nan, 0.01, 0.01, 0.01], ValueError, r"threshold\[2\] = NaN"),
        ([0.01, 0.01, 0.01, math.inf, 0.01, 0.01], ValueError, r"threshold\[3\]"),
        ([0.01, 0.01, 0.01, 0.01, -0.01, 0.01], ValueError, r"threshold\[4\]"),
        ([0.01, "a", 0.01, 0.01, 0.01, 0.01], TypeError, "must all be numbers"),
        (math.nan, ValueError, "finite and non-negative"),
        (-0.01, ValueError, "finite and non-negative"),
        ("0.01", TypeError, "not a string"),
        (None, TypeError, "number or a sequence"),
    ],
)
def test_threshold_is_validated(threshold, error, match):
    with pytest.raises(error, match=match):
        filters.cusum_filter_indices(CLOSE, threshold)
    with pytest.raises(error, match=match):
        filters.cusum_filter_timestamps(CLOSE, STAMPS, threshold)

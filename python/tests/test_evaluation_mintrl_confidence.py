"""The high-level MinTRL helper supports confidence strictly above one half."""

import math
from statistics import NormalDist

import pytest
from openquant import evaluation as ev

ORDINARY = [0.01, 0.02, 0.03]
CANCELLED = [x / 256 for x in [1.3962847939999437] + [0.39628479399994365] * 4]


@pytest.mark.parametrize("alpha", [0.5, math.nextafter(0.5, math.inf), 0.9, 1.0])
@pytest.mark.parametrize("case", ["ordinary", "underperforming", "cancelled"])
def test_mintrl_rejects_confidence_at_or_below_one_half(alpha, case):
    values = CANCELLED if case == "cancelled" else ORDINARY
    benchmark = ev.return_moments(values).sharpe if case == "cancelled" else 0.0
    if case == "underperforming":
        benchmark = 10.0
    with pytest.raises(ValueError, match="alpha"):
        ev.minimum_track_record_length(iter(values), benchmark, alpha)


@pytest.mark.parametrize("alpha", [-math.inf, -0.1, 0.0, math.nan, math.inf])
def test_mintrl_rejects_invalid_significance(alpha):
    with pytest.raises(ValueError, match="alpha"):
        ev.minimum_track_record_length(ORDINARY, alpha=alpha)


@pytest.mark.parametrize("alpha", [0.01, 0.05, 0.1, 0.25, 0.49])
def test_mintrl_solves_the_requested_upper_confidence_target(alpha):
    # Independent moment values for [0.01, 0.02, 0.03]: SR=2, skew=0, raw kurtosis=1.5.
    length = ev.minimum_track_record_length(ORDINARY, alpha=alpha)
    z = 2.0 * math.sqrt(length - 1.0) / math.sqrt(1.5)
    assert NormalDist().cdf(z) == pytest.approx(1.0 - alpha, abs=2e-14)


def test_mintrl_preserves_supported_boundary_and_underperformance_behavior():
    for alpha in [math.ulp(0.0), 1e-20, math.nextafter(0.5, 0.0)]:
        assert math.isfinite(ev.minimum_track_record_length(ORDINARY, alpha=alpha))
        assert ev.minimum_track_record_length(ORDINARY, benchmark_sr=10.0, alpha=alpha) == math.inf


def test_mintrl_rejects_the_mirrored_tail_instead_of_reporting_the_same_length():
    assert ev.minimum_track_record_length(ORDINARY, alpha=0.1) == pytest.approx(
        1.6158904056811818, abs=2e-14
    )
    with pytest.raises(ValueError, match="alpha"):
        ev.minimum_track_record_length(ORDINARY, alpha=0.9)


def test_native_algebraic_endpoint_contract_is_unchanged():
    from openquant import backtest_stats

    for alpha in [0.0, 1.0]:
        assert backtest_stats.minimum_track_record_length(0.1, 0.0, 0.0, 3.0, alpha) == math.inf
    assert backtest_stats.minimum_track_record_length(0.1, 0.0, 0.0, 3.0, 0.5) == 1.0

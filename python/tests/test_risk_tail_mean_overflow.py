"""Finite tail/class means recover overflow without changing tail membership."""

import math
from fractions import Fraction

import pytest
from openquant import risk, strategy_risk


@pytest.mark.parametrize(
    "tail",
    [
        [-1e308, -1e308],
        [-1e308, -9e307],
        [-1e308, -1e308, 5e307],
        [1e308, 9e307],
        [-1.7e308, -1.7e308, 1.6e308, 1.6e308, 1e-308],
        [8e307] * 3 + [-8e307] * 3 + [1e-308],
        [8e307] * 3 + [-8e307] * 3 + [math.ulp(0.0), 3 * math.ulp(0.0)],
        [8e307] * 3 + [-8e307] * 3 + [-math.ulp(0.0), -3 * math.ulp(0.0)],
    ],
)
def test_expected_shortfall_overflow_matches_exact_binary_input_mean(tail):
    maximum = float.fromhex("0x1.fffffffffffffp+1023")
    expected = float(sum(map(Fraction.from_float, tail), Fraction()) / len(tail))
    assert risk.calculate_expected_shortfall([*tail, maximum], 1.0).hex() == expected.hex()


@pytest.mark.parametrize("curve", [[1e308, 0.0, 0.0], [1e308, 0.0, 1e307]])
def test_cdar_overflow_averages_the_existing_drawdown_values(curve):
    drawdowns = [max(curve[: i + 1]) - x for i, x in enumerate(curve)]
    threshold = sorted(drawdowns)[1]
    tail = [x for x in drawdowns if x >= threshold]
    expected = float(sum(map(Fraction.from_float, tail), Fraction()) / len(tail))
    assert risk.calculate_conditional_drawdown_risk(curve, 0.5) == expected


def test_strategy_payout_overflow_preserves_scale_invariant_bootstrap():
    kwargs = dict(
        years_elapsed=1.0,
        target_sharpe=1.0,
        investor_horizon_years=1.0,
        bootstrap_iterations=20,
        seed=42,
    )
    baseline = strategy_risk.estimate_strategy_failure_probability([1.0, 0.9, -1.0, -0.9], **kwargs)
    actual = strategy_risk.estimate_strategy_failure_probability(
        [1e308, 9e307, -1e308, -9e307], **kwargs
    )
    expected = float((Fraction.from_float(1e308) + Fraction.from_float(9e307)) / 2)
    assert actual["pi_plus"] == expected
    assert actual["pi_minus"] == -expected
    assert actual["implied_precision_threshold"] == pytest.approx(
        baseline["implied_precision_threshold"], abs=1e-15
    )
    assert actual["bootstrap_precision_samples"] == baseline["bootstrap_precision_samples"]


def test_empty_tail_and_nonfinite_inputs_keep_existing_behavior():
    assert math.isnan(risk.calculate_expected_shortfall([1.0, 1.0], 0.5))
    assert math.isnan(risk.calculate_expected_shortfall([-2.0, -1.0], 0.0))
    assert risk.calculate_expected_shortfall([-math.inf, 0.0], 1.0) == -math.inf
    # Individual drawdown overflow remains distinct from finite-tail sum overflow.
    assert risk.calculate_conditional_drawdown_risk([1e308, -1e308], 1.0) == math.inf

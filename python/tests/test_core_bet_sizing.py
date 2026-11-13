"""Limit prices of dynamic bet sizing (AFML Snippet 10.4, issue #163).

Mirrors the hand-worked cases in crates/openquant/tests/bet_sizing.rs. With the power curve
and w = 1 the inverse price is f - m, so the limit price is f minus the mean of the positions
passed through, divided by the maximum position.
"""

import math

import pytest
from _core_fixtures import load_json
from openquant import bet_sizing


@pytest.mark.parametrize(
    ("current", "target", "expected"),
    [
        (0.0, 4.0, 99.75),  # increasing a long: 1, 2, 3, 4
        (10.0, 5.0, 99.3),  # reducing a long: 9, 8, 7, 6, 5
        (3.0, -2.0, 100.0),  # crossing zero: 2, 1, 0, -1, -2
        (-2.0, 3.0, 99.9),  # crossing zero upwards: -1, 0, 1, 2, 3
        (0.0, -4.0, 100.25),  # going short: -1, -2, -3, -4
        (-6.0, -2.0, 100.35),  # covering a short: -5, -4, -3, -2
    ],
)
def test_limit_price_power_follows_the_path(current, target, expected):
    got = bet_sizing.limit_price_power(target, current, 100.0, 1.0, 10.0)
    assert got == pytest.approx(expected, abs=1e-12)
    assert bet_sizing.limit_price(target, current, 100.0, 1.0, 10.0, "power") == got


@pytest.mark.parametrize(
    ("current", "target", "expected"),
    [
        (2.0, 4.0, 98.75),  # 3, 4 -> f - 0.9, f - 1.6
        (5.0, 3.0, 98.75),  # 4, 3
        (-2.0, -4.0, 101.25),  # -3, -4 -> f + 0.9, f + 1.6
        (4.0, -4.0, 100.2),  # 3..=-3 cancel, leaving f + 1.6 over 8 terms
    ],
)
def test_limit_price_sigmoid_follows_the_path(current, target, expected):
    # w = 1.44: sizes 0.6 and 0.8 (positions 3 and 4 of 5) map to f - 0.9 and f - 1.6.
    got = bet_sizing.limit_price_sigmoid(target, current, 100.0, 1.44, 5.0)
    assert got == pytest.approx(expected, abs=1e-12)


def test_limit_price_is_nan_without_a_trade():
    assert math.isnan(bet_sizing.limit_price_sigmoid(3.9, 3.2, 100.0, 1.44, 5.0))
    assert math.isnan(bet_sizing.limit_price_power(-2.0, -2.0, 100.0, 1.0, 10.0))


def test_bet_size_dynamic_limit_prices_match_reference():
    fixture = load_json("bet_sizing/prob_dynamic_budget.json")["dynamic"]
    rows = bet_sizing.bet_size_dynamic(
        [25.0, 35.0, 45.0, 40.0, 30.0],
        [55.0],
        [75.5, 76.9, 74.1, 67.75, 62.0],
        [80.0, 75.0, 72.5, 65.0, 70.8],
    )
    got = [limit for _, _, limit in rows]
    assert got == pytest.approx(fixture["l_p_path"], abs=1e-9)
    # Rows 0 and 4 grow a long position, where Snippet 10.4 as written agrees.
    for i in (0, 4):
        assert got[i] == pytest.approx(fixture["l_p"][i], abs=1e-9)


NAN = float("nan")
FIT = [0.0, 1.0, 1.0, 2.0, 0.5]
STARTS = ["2024-01-01 00:00:00", "2024-01-02 00:00:00"]
ENDS = ["2024-01-03 00:00:00", "2024-01-04 00:00:00"]
SIDES = [1.0, -1.0]


@pytest.mark.parametrize(
    ("call", "name"),
    [
        (lambda: bet_sizing.cdf_mixture(NAN, 1.0, 1.0, 2.0, 0.5, 0.0), "mu1"),
        (lambda: bet_sizing.cdf_mixture(0.0, NAN, 1.0, 2.0, 0.5, 0.0), "mu2"),
        (lambda: bet_sizing.cdf_mixture(0.0, 1.0, NAN, 2.0, 0.5, 0.0), "sigma1"),
        (lambda: bet_sizing.cdf_mixture(0.0, 1.0, 1.0, 2.0, 0.5, NAN), "x"),
        (lambda: bet_sizing.single_bet_size_mixed(NAN, FIT), "c"),
        (lambda: bet_sizing.single_bet_size_mixed(0.5, [NAN, *FIT[1:]]), "mu1"),
        (lambda: bet_sizing.bet_size_reserve(STARTS, ENDS, SIDES, [0.0, NAN, *FIT[2:]]), "mu2"),
        (lambda: bet_sizing.bet_size_reserve_with_fit(STARTS, ENDS, SIDES, [*FIT[:4], NAN]), "p1"),
    ],
)
def test_mixture_functions_raise_value_error_on_nan(call, name):
    # Issue #194: a NaN mean used to raise pyo3_runtime.PanicException, a BaseException that
    # `except ValueError` and `except Exception` both miss.
    with pytest.raises(ValueError, match=rf"'{name}' must be finite, got NaN"):
        call()


@pytest.mark.parametrize("sign", [-1.0, 1.0])
def test_sigmoid_finite_inputs_survive_intermediate_overflow(sign):
    # AFML 10.4: these sizes round to +/-1, so a 50-lot book must not go flat.
    assert bet_sizing.bet_size_sigmoid(1.0, sign * 1e200) == sign
    assert bet_sizing.get_target_pos_sigmoid(1.0, sign * 1e200, 0.0, 50.0) == sign * 50.0
    assert bet_sizing.bet_size_sigmoid(1e308, sign * 1e154) == pytest.approx(
        sign / math.sqrt(2.0), rel=1e-15
    )


@pytest.mark.parametrize("sign", [-1.0, 1.0])
def test_inverse_sigmoid_finite_result_survives_intermediate_overflow(sign):
    # m=3/5 gives m/sqrt(1-m²)=3/4, allowing binary64 rounding of m.
    expected = -sign * 0.75 * math.sqrt(1.7e308)
    assert bet_sizing.inv_price_sigmoid(0.0, 1.7e308, sign * 0.6) == pytest.approx(
        expected, rel=1e-15
    )


@pytest.mark.parametrize("sign", [-1.0, 1.0])
def test_sigmoid_calibration_recovers_extreme_finite_width(sign):
    # x=2^520, m=1-2^-53 implies w=2^988*(1-2^-54)/(1-2^-53)^2.
    got = bet_sizing.get_w_sigmoid(sign * 2.0**520, sign * math.nextafter(1.0, 0.0))
    assert got == pytest.approx(2.0**988, rel=1e-15)
    assert bet_sizing.get_w_sigmoid(sign * 1e-200, sign * 1e-200) == 1.0
    assert bet_sizing.get_w_sigmoid(sign * 1e-200, sign * 1e-100) == pytest.approx(
        1e-200, rel=1e-15, abs=0.0
    )


@pytest.mark.parametrize("divergence", [1e-200, -1e-200, math.ulp(0.0), -math.ulp(0.0)])
def test_zero_width_sigmoid_keeps_tiny_divergence_direction(divergence):
    assert bet_sizing.bet_size_sigmoid(0.0, divergence) == math.copysign(1.0, divergence)
    assert math.isnan(bet_sizing.bet_size_sigmoid(0.0, 0.0))


@pytest.mark.parametrize("sign", [-1.0, 1.0])
def test_sigmoid_subnormal_squared_norm_keeps_relative_accuracy(sign):
    assert bet_sizing.bet_size_sigmoid(0.0, sign * 1.6e-162) == sign
    width = math.ulp(0.0)
    assert bet_sizing.bet_size_sigmoid(width, sign * math.sqrt(width)) == pytest.approx(
        sign / math.sqrt(2.0), rel=1e-15
    )
    # x=(3/4)*sqrt(w) gives (3/4)/sqrt(1+9/16)=3/5 exactly.
    assert bet_sizing.bet_size_sigmoid(width, sign * 0.75 * math.sqrt(width)) == pytest.approx(
        sign * 0.6, rel=1e-15
    )

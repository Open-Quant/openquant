"""Independent dollar-cost fixtures and boundary contracts for a synthetic lab."""

from __future__ import annotations

import hashlib
import importlib.util
import itertools
import json
import math
import subprocess
import sys
from decimal import Decimal, localcontext
from pathlib import Path

import numpy as np
import pytest

MODULE = Path(__file__).resolve().parents[2] / "experiments" / "capacity_scaling_lab.py"


def load_lab():
    assert MODULE.is_file(), "the fixed-schedule capacity lab is not implemented"
    spec = importlib.util.spec_from_file_location("capacity_scaling_lab", MODULE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixture(**updates):
    values = dict(
        scale=100.0,
        trade_fractions=[0.25],
        spread=[0.01],
        impact=[0.5],
        volatility=[0.2],
        dollar_volume=[100.0],
        gross_return=0.1,
        participation=0.1,
    )
    values.update(updates)
    return values


def test_hand_derived_dollar_cost_and_distinct_capacity_definitions():
    # q=25; spread=.25, impact=.1*125/10=1.25; L=.0025, I=.00125.
    row = load_lab().evaluate_schedule(**fixture())
    assert row["linear_coefficient"] == pytest.approx(0.0025)
    assert row["impact_coefficient"] == pytest.approx(0.00125)
    assert row["linear_cost_dollars"] == pytest.approx(0.25)
    assert row["impact_cost_dollars"] == pytest.approx(1.25)
    assert row["total_cost_dollars"] == pytest.approx(1.5)
    assert row["normalized_cost"] == pytest.approx(0.015)
    assert row["net_expected_pnl_per_scale"] == pytest.approx(0.085)
    assert row["net_expected_pnl_dollars"] == pytest.approx(8.5)
    assert row["positive_net"] is True
    assert row["break_even"]["status"] == "finite"
    assert row["break_even"]["scale"] == pytest.approx(6084)
    assert row["participation_cap"]["scale"] == pytest.approx(40)
    assert row["participation_cap"]["status"] == "finite"
    assert row["within_participation_cap"] is False
    assert row["representation_limits"] == {}


def direct_dollar_oracle(inputs):
    """Independent 110-digit sum from dollar trades, not coefficient decomposition."""
    with localcontext() as ctx:
        ctx.prec = 110
        a = Decimal.from_float(float(inputs["scale"]))
        linear, impact = Decimal(0), Decimal(0)
        for d, s, eta, sigma, volume in zip(
            *(
                np.array(inputs[key]).flatten()
                for key in ("trade_fractions", "spread", "impact", "volatility", "dollar_volume")
            ),
            strict=True,
        ):
            q = abs(a * Decimal.from_float(float(d)))
            linear += Decimal.from_float(float(s)) * q
            impact += (
                Decimal.from_float(float(eta))
                * Decimal.from_float(float(sigma))
                * (q * q * q / Decimal.from_float(float(volume))).sqrt()
            )
        net = a * Decimal.from_float(float(inputs["gross_return"])) - linear - impact
        return tuple(float(x) for x in (linear, impact, net, net / a))


@pytest.mark.parametrize("scale", [0.01, 1, 137, 1e7])
def test_multiple_execution_buckets_match_independent_direct_dollar_oracle(scale):
    values = fixture(
        scale=scale,
        trade_fractions=[[0.1, -0.3], [0, 0.04]],
        spread=[[0.0001, 0.0004], [0.2, 0.0008]],
        impact=[[0.1, 0.4], [1, 0]],
        volatility=[[0.01, 0.03], [0.5, 0.1]],
        dollar_volume=[[1e6, 4e6], [1, 2300]],
        participation=[[0.1, 0.2], [0, 0.05]],
    )
    row = load_lab().evaluate_schedule(**values)
    expected = direct_dollar_oracle(values)
    for key, want in zip(
        (
            "linear_cost_dollars",
            "impact_cost_dollars",
            "net_expected_pnl_dollars",
            "net_expected_pnl_per_scale",
        ),
        expected,
        strict=True,
    ):
        assert row[key] == pytest.approx(want, rel=2e-14, abs=1e-16)


def test_doubling_scale_has_linear_and_three_halves_cost_exponents():
    lab = load_lab()
    a = lab.evaluate_schedule(**fixture())
    b = lab.evaluate_schedule(**fixture(scale=200))
    assert b["linear_cost_dollars"] / a["linear_cost_dollars"] == pytest.approx(2)
    assert b["impact_cost_dollars"] / a["impact_cost_dollars"] == pytest.approx(2**1.5)
    assert b["break_even"] == a["break_even"]


def test_scaling_capital_and_dollar_volume_preserves_normalized_impact():
    lab = load_lab()
    a = lab.evaluate_schedule(**fixture())
    b = lab.evaluate_schedule(**fixture(scale=700, dollar_volume=[700]))
    assert b["impact_cost_dollars"] / 700 == pytest.approx(a["impact_cost_dollars"] / 100)
    assert b["normalized_cost"] == pytest.approx(a["normalized_cost"])
    assert b["participation_cap"]["scale"] == pytest.approx(7 * a["participation_cap"]["scale"])


def test_signed_trades_have_same_cost_without_cancellation():
    lab = load_lab()
    a = lab.evaluate_schedule(**fixture(trade_fractions=[0.25]))
    b = lab.evaluate_schedule(**fixture(trade_fractions=[-0.25]))
    assert a == b
    both = lab.evaluate_schedule(
        **fixture(
            trade_fractions=[0.25, -0.25],
            spread=[0.01, 0.01],
            impact=[0.5, 0.5],
            volatility=[0.2, 0.2],
            dollar_volume=[100, 100],
        )
    )
    assert both["total_cost_dollars"] == pytest.approx(3)


@pytest.mark.parametrize("gross_return", [0, -1, 0.0025])
def test_no_positive_net_scale_when_gross_does_not_cover_linear_cost(gross_return):
    row = load_lab().evaluate_schedule(**fixture(gross_return=gross_return))
    assert row["break_even"] == {"status": "no_positive_net_scale", "scale": None}
    assert row["positive_net"] is False
    assert row["net_expected_pnl_per_scale"] <= 0


def test_linear_only_cost_is_scale_invariant_and_has_no_finite_crossing():
    lab = load_lab()
    a = lab.evaluate_schedule(**fixture(impact=[0]))
    b = lab.evaluate_schedule(**fixture(impact=[0], scale=1e12))
    assert a["break_even"] == {"status": "no_finite_crossing", "scale": None}
    assert a["net_expected_pnl_per_scale"] == b["net_expected_pnl_per_scale"]
    assert a["impact_cost_dollars"] == b["impact_cost_dollars"] == 0


def test_zero_margin_without_impact_is_not_positive_at_any_scale():
    # Binary-exact fixture ensures G=L, including in unrounded arithmetic.
    row = load_lab().evaluate_schedule(**fixture(spread=[0.125], gross_return=0.03125, impact=[0]))
    assert row["net_expected_pnl_per_scale"] == 0
    assert row["positive_net"] is False
    assert row["break_even"]["status"] == "no_positive_net_scale"


def test_all_zero_trade_schedule_is_preserved_and_has_no_participation_limit():
    row = load_lab().evaluate_schedule(**fixture(trade_fractions=[0], participation=0))
    assert row["total_cost_dollars"] == 0
    assert row["normalized_cost"] == 0
    assert row["net_expected_pnl_dollars"] == pytest.approx(10)
    assert row["break_even"]["status"] == "no_finite_crossing"
    assert row["participation_cap"] == {"status": "no_trade_limit", "scale": None}
    assert row["within_participation_cap"] is True


def test_zero_participation_prevents_nonzero_trades_but_ignores_zero_trades():
    lab = load_lab()
    row = lab.evaluate_schedule(**fixture(participation=0))
    assert row["participation_cap"] == {"status": "finite", "scale": 0.0}
    assert row["within_participation_cap"] is False
    row = lab.evaluate_schedule(
        **fixture(
            trade_fractions=[0.25, 0],
            spread=[0.01, 0],
            impact=[0.5, 0],
            volatility=[0.2, 0],
            dollar_volume=[100, 1],
            participation=[0.1, 0],
        )
    )
    assert row["participation_cap"]["scale"] == pytest.approx(40)


def test_participation_equality_is_feasible_and_break_even_is_not_positive():
    # Exact binary fixture: L=0, I=.25, G=.5, zero-net scale=4 and cap=4.
    values = fixture(
        scale=4,
        trade_fractions=[1],
        spread=[0],
        impact=[0.5],
        volatility=[1],
        dollar_volume=[4],
        gross_return=0.5,
        participation=1,
    )
    row = load_lab().evaluate_schedule(**values)
    assert row["break_even"] == {"status": "finite", "scale": 4.0}
    assert row["net_expected_pnl_per_scale"] == 0
    assert row["positive_net"] is False
    assert row["within_participation_cap"] is True
    assert load_lab().evaluate_schedule(**{**values, "scale": 1})["positive_net"] is True
    assert load_lab().evaluate_schedule(**{**values, "scale": 16})["positive_net"] is False


@pytest.mark.parametrize(
    "field,value",
    [
        ("scale", 0),
        ("scale", -1),
        ("scale", math.inf),
        ("scale", math.nan),
        ("scale", True),
        ("scale", "100"),
        ("scale", [100]),
        ("gross_return", math.nan),
        ("gross_return", True),
        ("gross_return", 1j),
        ("trade_fractions", []),
        ("trade_fractions", 1),
        ("trade_fractions", [[[1]]]),
        ("trade_fractions", [math.nan]),
        ("trade_fractions", [True]),
        ("trade_fractions", ["0.25"]),
        ("trade_fractions", [0.25j]),
        ("spread", [-0.1]),
        ("impact", [-0.1]),
        ("volatility", [-0.1]),
        ("spread", [math.inf]),
        ("spread", [True]),
        ("impact", [None]),
        ("dollar_volume", [0]),
        ("dollar_volume", [-1]),
        ("dollar_volume", [None]),
        ("dollar_volume", [math.nan]),
        ("dollar_volume", [math.inf]),
        ("dollar_volume", None),
        ("dollar_volume", ["100"]),
        ("dollar_volume", [[100]]),
        ("dollar_volume", [100, 200]),
        ("impact", 0.5),
        ("volatility", [0.2, 0.2]),
        ("participation", -0.1),
        ("participation", 1.01),
        ("participation", math.inf),
        ("participation", True),
        ("participation", [True]),
        ("participation", [0.1, 0.1]),
    ],
)
def test_rejects_invalid_shapes_types_and_units_without_coercion(field, value):
    with pytest.raises((ValueError, TypeError)):
        load_lab().evaluate_schedule(**fixture(**{field: value}))


def test_unknown_volume_is_not_inferred_as_liquidity_even_with_zero_trade_and_impact():
    with pytest.raises((ValueError, TypeError)):
        load_lab().evaluate_schedule(**fixture(trade_fractions=[0], impact=[0], dollar_volume=[0]))


@pytest.mark.parametrize(
    "impact,want", [([1e-200], "above_float64_range"), ([1e200], "below_float64_range")]
)
def test_finite_crossing_outside_float_range_is_not_misclassified(impact, want):
    row = load_lab().evaluate_schedule(
        **fixture(
            trade_fractions=[1],
            spread=[0],
            impact=impact,
            volatility=[1],
            dollar_volume=[1],
            gross_return=1,
        )
    )
    assert row["break_even"] == {"status": want, "scale": None}
    json.dumps(row, allow_nan=False)


def test_underflowed_impact_coefficient_is_not_treated_as_zero_impact():
    row = load_lab().evaluate_schedule(
        **fixture(
            trade_fractions=[1e-200],
            spread=[0],
            impact=[1e-200],
            volatility=[1],
            dollar_volume=[1],
            gross_return=1,
        )
    )
    assert row["impact_coefficient"] is None
    assert row["representation_limits"]["impact_coefficient"] == "below_float64_range"
    assert row["break_even"]["status"] == "above_float64_range"


def test_overflowing_derived_costs_are_explicit_and_strict_json_safe():
    row = load_lab().evaluate_schedule(
        **fixture(
            scale=1e308,
            trade_fractions=[1e308],
            spread=[1],
            impact=[1],
            volatility=[1],
            dollar_volume=[1],
            gross_return=1,
        )
    )
    assert row["total_cost_dollars"] is None
    assert row["representation_limits"]["total_cost_dollars"] == "above_float64_range"
    assert row["positive_net"] is False
    json.dumps(row, allow_nan=False)


@pytest.mark.parametrize(
    "d,p,volume,want",
    [(1e-308, 1, 1e308, "above_float64_range"), (1e308, 1e-308, 1e-308, "below_float64_range")],
)
def test_participation_range_limits_remain_separate_from_absent_limit(d, p, volume, want):
    row = load_lab().evaluate_schedule(
        **fixture(
            trade_fractions=[d],
            spread=[0],
            impact=[0],
            volatility=[0],
            dollar_volume=[volume],
            participation=p,
        )
    )
    assert row["participation_cap"] == {"status": want, "scale": None}
    assert row["within_participation_cap"] is (want == "above_float64_range")


def test_fixed_grid_keeps_all_cells_and_replays_exactly_from_serialized_configuration():
    lab = load_lab()
    report = lab.run_lab()
    assert report["lab"] == "capacity_scaling"
    assert report["schema_version"] == 1
    assert report["synthetic_only"] is True
    assert len(report["cells"]) == 90
    keys = ("scale", "volume_scale", "impact_coefficient_input", "gross_return")
    observed = {tuple(row[key] for key in keys) for row in report["cells"]}
    expected = set(
        itertools.product([1e3, 1e4, 1e5, 1e6, 1e7], [0.1, 1, 10], [0, 0.1, 0.5], [0, 0.001])
    )
    assert observed == expected
    assert any(not row["within_participation_cap"] for row in report["cells"])
    assert any(row["net_expected_pnl_per_scale"] < 0 for row in report["cells"])
    assert all(not row["positive_net"] for row in report["cells"] if row["gross_return"] == 0)
    assert report["provenance"]["source_sha256"] == hashlib.sha256(MODULE.read_bytes()).hexdigest()
    assert report["provenance"]["numpy"] == np.__version__
    serialized = json.loads(json.dumps(report, allow_nan=False))
    replay = lab.run_lab(**serialized["config"])
    assert replay["cells"] == report["cells"]
    assert replay["controls"] == report["controls"]
    assert all(control["passed"] for control in report["controls"].values())
    assert report["negative_controls"]["shares_as_dollars"]["detected_by_input_validation"] is False
    assert report["negative_controls"]["omitted_scale_normalization"][
        "wrong_value"
    ] != pytest.approx(report["negative_controls"]["omitted_scale_normalization"]["correct_value"])


def test_custom_grid_and_schedule_are_replayed_without_hidden_defaults():
    lab = load_lab()
    kwargs = dict(
        scales=[3, 70],
        volume_scales=[2],
        impact_coefficients=[0.3],
        gross_returns=[-0.01, 0.2],
        trade_fractions=[[0, 0.5]],
        spread=[[0, 0.1]],
        volatility=[[0, 0.2]],
        dollar_volume=[[4, 16]],
        participation=[[0, 0.3]],
    )
    report = lab.run_lab(**kwargs)
    assert len(report["cells"]) == 4
    assert lab.run_lab(**report["config"])["cells"] == report["cells"]
    first = report["cells"][0]
    oracle = direct_dollar_oracle(
        dict(
            scale=3,
            trade_fractions=[[0, 0.5]],
            spread=[[0, 0.1]],
            impact=[[0.3, 0.3]],
            volatility=[[0, 0.2]],
            dollar_volume=[[8, 32]],
            gross_return=-0.01,
        )
    )
    assert first["net_expected_pnl_dollars"] == pytest.approx(oracle[2])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"scales": []},
        {"scales": [0]},
        {"scales": [1, 1]},
        {"scales": [True]},
        {"volume_scales": [0]},
        {"volume_scales": [math.nan]},
        {"impact_coefficients": [-1]},
        {"gross_returns": [math.inf]},
        {"volume_scales": [1e308]},
        {"dollar_volume": [1e-308, 1e-308], "volume_scales": [1e-308]},
    ],
)
def test_invalid_grid_and_unrepresentable_scaled_volumes_are_rejected(kwargs):
    with pytest.raises((ValueError, TypeError)):
        load_lab().run_lab(**kwargs)


def test_cli_writes_strict_json_then_refuses_overwrite(tmp_path):
    target = tmp_path / "nested" / "report.json"
    run = subprocess.run(
        [sys.executable, str(MODULE), "--out", str(target)], capture_output=True, text=True
    )
    assert run.returncode == 0, run.stderr
    original = target.read_bytes()
    report = json.loads(original)
    assert len(report["cells"]) == 90
    again = subprocess.run(
        [sys.executable, str(MODULE), "--out", str(target)], capture_output=True, text=True
    )
    assert again.returncode != 0
    assert "exist" in again.stderr.lower() or "overwrite" in again.stderr.lower()
    assert target.read_bytes() == original


def test_cli_refuses_symlink_output_without_touching_target(tmp_path):
    load_lab()
    sentinel = tmp_path / "existing.json"
    sentinel.write_text("preserve me")
    symlink = tmp_path / "report.json"
    symlink.symlink_to(sentinel)
    run = subprocess.run(
        [sys.executable, str(MODULE), "--out", str(symlink)], capture_output=True, text=True
    )
    assert run.returncode != 0
    assert sentinel.read_text() == "preserve me"


def test_tiny_negative_net_is_not_erased_by_adding_cost_to_large_linear_term():
    row = load_lab().evaluate_schedule(
        **fixture(
            scale=1,
            trade_fractions=[1],
            spread=[1],
            impact=[1e-100],
            volatility=[1],
            dollar_volume=[1],
            gross_return=1,
        )
    )
    assert row["net_expected_pnl_per_scale"] == pytest.approx(-1e-100, rel=1e-14, abs=0)
    assert row["positive_net"] is False


def test_tiny_linear_margin_is_not_erased_by_coefficient_summation():
    row = load_lab().evaluate_schedule(
        **fixture(
            scale=1,
            trade_fractions=[1, 1],
            spread=[1, 1e-100],
            impact=[0, 0],
            volatility=[1, 1],
            dollar_volume=[1, 1],
            gross_return=1,
        )
    )
    assert row["net_expected_pnl_per_scale"] == pytest.approx(-1e-100, rel=1e-14, abs=0)
    assert row["break_even"]["status"] == "no_positive_net_scale"


@pytest.mark.parametrize(
    "d,volume,eta,want",
    [
        (1e200, 1e200, 1e-100, 1e100),
        (1e308, 1e-308, 1e-308, 1e308),
        (1e-308, 1e308, 1e308, 1e-308),
    ],
)
def test_representable_impact_survives_intermediate_power_and_ratio_extremes(d, volume, eta, want):
    row = load_lab().evaluate_schedule(
        **fixture(
            scale=1,
            trade_fractions=[d],
            spread=[0],
            impact=[eta],
            volatility=[1],
            dollar_volume=[volume],
            gross_return=0,
        )
    )
    assert row["impact_cost_dollars"] == pytest.approx(want, rel=1e-14, abs=0)


def test_decimal_work_does_not_depend_on_callers_decimal_context():
    lab = load_lab()
    original = lab.evaluate_schedule(**fixture())
    with localcontext() as context:
        context.prec = 3
        context.rounding = "ROUND_DOWN"
        assert lab.evaluate_schedule(**fixture()) == original


def test_module_import_does_not_inherit_hostile_decimal_default_context():
    from decimal import DefaultContext, Inexact

    original = load_lab().evaluate_schedule(**fixture())
    saved = DefaultContext.copy()
    try:
        DefaultContext.rounding = "ROUND_UP"
        DefaultContext.prec = 3
        DefaultContext.traps[Inexact] = True
        assert load_lab().evaluate_schedule(**fixture()) == original
    finally:
        for attribute in ("prec", "rounding", "Emin", "Emax", "capitals", "clamp"):
            setattr(DefaultContext, attribute, getattr(saved, attribute))
        DefaultContext.traps = saved.traps
        DefaultContext.flags = saved.flags


def test_exact_radical_sum_preserves_geometric_series_net_after_1000_terms():
    row = load_lab().evaluate_schedule(
        **fixture(
            scale=1,
            trade_fractions=[1] * 1000,
            spread=[0] * 1000,
            impact=[2.0**-i for i in range(1, 1001)],
            volatility=[1] * 1000,
            dollar_volume=[1] * 1000,
            gross_return=1,
        )
    )
    assert row["net_expected_pnl_per_scale"] == 2.0**-1000
    assert row["positive_net"] is True
    assert row["net_evaluation"] == "exact_rational"
    assert row["precision_limits"] == []


def test_irrational_near_cancellation_is_explicitly_unresolved_not_valid_zero():
    with localcontext() as context:
        context.prec = 400
        remaining = Decimal(2).sqrt()
        terms = []
        for exponent in range(1001):
            term = 2.0**-exponent
            if Decimal.from_float(term) <= remaining:
                terms.append(term)
                remaining -= Decimal.from_float(term)
    # sum(eta) approximates sqrt(2) to ~300 places, and each impact term has sqrt(1/2).
    n = len(terms)
    row = load_lab().evaluate_schedule(
        **fixture(
            scale=1,
            trade_fractions=[1] * n,
            spread=[0] * n,
            impact=terms,
            volatility=[1] * n,
            dollar_volume=[2] * n,
            gross_return=1,
        )
    )
    assert row["net_evaluation"] == "near_cancellation_unresolved"
    assert row["net_expected_pnl_per_scale"] is None
    assert row["net_expected_pnl_dollars"] is None
    assert row["positive_net"] is None
    assert "net_expected_pnl_per_scale" in row["precision_limits"]
    json.dumps(row, allow_nan=False)


def test_scale_adjusted_exact_radical_can_resolve_net_even_with_irrational_coefficient():
    row = load_lab().evaluate_schedule(
        **fixture(
            scale=2,
            trade_fractions=[1],
            spread=[0],
            impact=[1],
            volatility=[1],
            dollar_volume=[2],
            gross_return=1,
        )
    )
    assert row["net_expected_pnl_per_scale"] == 0
    assert row["positive_net"] is False
    assert row["net_evaluation"] == "exact_rational"


def test_participation_exposes_exact_cap_when_display_rounding_changes_boundary_reading():
    from fractions import Fraction

    row = load_lab().evaluate_schedule(
        **fixture(
            scale=1e5,
            trade_fractions=[0.1],
            dollar_volume=[1e6],
            participation=0.01,
        )
    )
    expected_cap = Fraction(0.01) * 1000000 / Fraction(0.1)
    assert row["participation_cap_exact"] == str(expected_cap)
    assert row["participation_cap"]["scale"] == 1e5
    assert row["within_participation_cap"] is False
    assert row["participation_boundary_rounded"] is True
    exact = load_lab().evaluate_schedule(
        **fixture(
            scale=50,
            trade_fractions=[0.25],
            dollar_volume=[100],
            participation=0.125,
        )
    )
    assert exact["participation_cap_exact"] == "50"
    assert exact["within_participation_cap"] is True
    assert exact["participation_boundary_rounded"] is False

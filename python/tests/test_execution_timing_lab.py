"""Independent terminal cash-ledger and timing boundary checks."""

import importlib.util
import json
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import pytest

MODULE = Path(__file__).resolve().parents[2] / "experiments/execution_timing_lab.py"


def load_lab():
    assert MODULE.is_file(), "execution timing lab is not implemented"
    spec = importlib.util.spec_from_file_location("execution_timing_lab", MODULE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def inputs(**updates):
    return dict(
        target=2,
        horizon=3,
        alpha=1,
        decay=0.5,
        spread=0.25,
        impact=0.5,
        latency=0,
        schedule="uniform",
        **updates,
    )


def test_hand_check_and_terminal_completion():
    row = load_lab().evaluate_schedule(**inputs())
    # q=(2/3,2/3,2/3); x=(2/3,4/3,2); alpha=(1,1/2,1/4).
    assert row["exact"]["gross_pnl"] == "11/6"
    assert row["exact"]["spread_cost"] == "1/2"
    assert row["exact"]["impact_cost"] == "2/3"
    assert row["exact"]["net_pnl"] == "2/3"
    assert row["exact"]["terminal_inventory"] == "2"
    assert row["exact"]["opportunity_cost"] == "5/3"
    assert row["net_pnl"] == pytest.approx(2 / 3)


@pytest.mark.parametrize("schedule", ["front", "uniform", "back"])
@pytest.mark.parametrize("latency", [0, 1, 2])
def test_independent_purchase_cash_ledger(schedule, latency):
    kwargs = inputs()
    kwargs.update(schedule=schedule, latency=latency)
    row = load_lab().evaluate_schedule(**kwargs)
    trades = [Fraction(x) for x in row["exact"]["trades"]]
    price, cash = Fraction(100), Fraction(0)
    for t, q in enumerate(trades):
        cash -= q * price + q / 4 + q * q / 2
        price += Fraction(1, 2) ** t
    assert cash + 2 * price == Fraction(row["exact"]["net_pnl"])
    assert sum(trades) == 2
    assert not any(trades[:latency])


def test_latency_does_not_restart_alpha_clock():
    kwargs = inputs()
    kwargs.update(schedule="front", latency=2)
    row = load_lab().evaluate_schedule(**kwargs)
    assert row["exact"]["gross_pnl"] == "1/2"
    assert row["exact"]["impact_cost"] == "2"


def test_zero_decay_zero_alpha_and_one_slot():
    lab = load_lab()
    kwargs = inputs()
    kwargs.update(decay=0, schedule="front")
    assert lab.evaluate_schedule(**kwargs)["gross_pnl"] == 2
    kwargs.update(latency=1)
    assert lab.evaluate_schedule(**kwargs)["gross_pnl"] == 0
    kwargs.update(alpha=0)
    assert lab.evaluate_schedule(**kwargs)["net_pnl"] == -2.5
    kwargs.update(horizon=1, latency=0)
    rows = [
        lab.evaluate_schedule(**{**kwargs, "schedule": s}) for s in ("front", "uniform", "back")
    ]
    assert len({r["exact"]["net_pnl"] for r in rows}) == 1


@pytest.mark.parametrize(
    "key,value",
    [
        ("target", 0),
        ("target", True),
        ("alpha", -1),
        ("alpha", "1"),
        ("alpha", float("inf")),
        ("decay", 1.1),
        ("impact", -1),
        ("spread", -1),
        ("horizon", 0),
        ("horizon", 2.0),
        ("horizon", True),
        ("latency", -1),
        ("latency", 3),
        ("schedule", "optimize"),
    ],
)
def test_invalid_inputs(key, value):
    with pytest.raises((ValueError, TypeError)):
        load_lab().evaluate_schedule(**{**inputs(), key: value})


def test_extreme_range_keeps_exact_values_and_strict_json():
    kwargs = {**inputs(), "target": 1e308, "impact": 1e308}
    row = load_lab().evaluate_schedule(**kwargs)
    assert row["impact_cost"] is None
    assert row["representation_limits"]["impact_cost"] == "above_float64_range"
    assert Fraction(row["exact"]["impact_cost"]) > 0
    json.dumps(row, allow_nan=False)


def test_report_roundtrip_and_controls():
    lab = load_lab()
    report = lab.run_lab()
    replay = lab.run_lab(**report["config"])
    for key in report.keys() - {"performance", "provenance"}:
        assert report[key] == replay[key]
    assert len(report["cells"]) == 54
    assert all(item["passed"] for item in report["controls"].values())
    assert all(item["detected"] for item in report["negative_controls"].values())
    assert report["synthetic_only"] is True
    json.dumps(report, allow_nan=False)


def test_cli_exclusive_output(tmp_path):
    output = tmp_path / "report.json"
    command = [sys.executable, str(MODULE), "--out", str(output)]
    assert subprocess.run(command, capture_output=True).returncode == 0
    before = output.read_bytes()
    assert subprocess.run(command, capture_output=True).returncode != 0
    assert output.read_bytes() == before


def test_runner_runtime_provenance_contract():
    import numpy as np

    assert load_lab().run_lab()["provenance"]["numpy"] == np.__version__


@pytest.mark.parametrize(
    "updates",
    [{"decays": []}, {"impacts": []}, {"latencies": []}, {"schedules": []}, {"latencies": [4]}],
)
def test_invalid_grid(updates):
    with pytest.raises(ValueError):
        load_lab().run_lab(**updates)


def test_subnormal_cost_is_not_silently_zero():
    row = load_lab().evaluate_schedule(**{**inputs(), "target": 5e-324, "impact": 5e-324})
    assert row["impact_cost"] is None
    assert row["representation_limits"]["impact_cost"] == "below_float64_range"
    assert Fraction(row["exact"]["impact_cost"]) > 0


def test_restarted_clock_negative_control_keeps_same_horizon():
    negative = load_lab().run_lab()["negative_controls"]["restart_alpha_at_execution"]
    # At latency 2 of horizon 3, exactly one interval remains, incorrectly earning 2*1.
    assert negative["wrong_gross_pnl"] == "2"

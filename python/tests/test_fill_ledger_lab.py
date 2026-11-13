"""Report integration and negative controls for the exact fill ledger study."""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
from fractions import Fraction
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def load_module(name):
    path = ROOT / f"experiments/{name}.py"
    assert path.exists(), f"missing {name} implementation"
    spec = importlib.util.spec_from_file_location(f"_test_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_report_preserves_entire_grid_and_every_exact_row():
    m = load_module("fill_ledger_lab")
    report = m.run_lab()
    assert len(report["scenarios"]) == 48
    assert report["synthetic_only"] is True
    assert report["numeric_encoding"] == "exact rational strings from decimal inputs"
    for scenario in report["scenarios"]:
        ledger = scenario["ledger"]
        assert len(ledger["rows"]) == 4
        assert len(ledger["orders"]) == 2
        for row in ledger["rows"]:
            assert Fraction(row["equity_change"]) == (
                Fraction(row["market_pnl"])
                + Fraction(row["execution_pnl"])
                - Fraction(row["commission"])
            )
        assert "inventory" in ledger["terminal"]
    json.dumps(report, allow_nan=False)


def test_controls_detect_capacity_double_counting_and_residual_erasure():
    report = load_module("fill_ledger_lab").run_lab()
    assert all(control["passed"] for control in report["controls"].values())
    assert all(control["detected"] for control in report["negative_controls"].values())
    controls = report["controls"]
    assert controls["hand_entry_and_exit"]["observed"] == {
        "entry_cash": "79",
        "entry_equity": "101",
        "terminal_cash": "99",
    }
    assert controls["shared_capacity"]["observed"] == {"filled": "5", "remainders": ["1", "6"]}
    assert controls["partial_terminal"]["observed"] == {
        "inventory": "1",
        "cash": "181/2",
        "marked_equity": "203/2",
        "completed": False,
    }


@pytest.mark.parametrize(
    "kwargs",
    [
        {"participations": []},
        {"participations": [2]},
        {"fixed_fees": [-1]},
        {"fee_rates": [float("nan")]},
        {"terminal_volumes": [-1]},
    ],
)
def test_bad_grids_rejected_before_reporting(kwargs):
    with pytest.raises(ValueError):
        load_module("fill_ledger_lab").run_lab(**kwargs)


@pytest.mark.parametrize("grid", ["10", b"10", {"0": "ignored", "1": "ignored"}, {0, 1}])
def test_grid_requires_ordered_array_not_scalar_mapping_or_set(grid):
    with pytest.raises(ValueError, match="array"):
        load_module("fill_ledger_lab").run_lab(participations=grid)


def test_registered_report_replays_and_detects_all_evidence_mutations():
    replay = load_module("replay_offline_report")
    assert replay.LABS.get("fill_ledger") == "fill_ledger_lab.py", "register the reviewed lab"
    assert replay.DEPENDENCIES["fill_ledger"] == ("fill_ledger.py",)
    config = json.loads((ROOT / "experiments/configs/offline-smoke/fill_ledger.json").read_text())
    report = replay._load_lab("fill_ledger").run_lab(**config)
    assert replay.replay_report(report, lab="fill_ledger")["matched"]
    for where in ("science", "config"):
        altered = copy.deepcopy(report)
        if where == "science":
            altered["scenarios"][0]["ledger"]["terminal"]["cash"] = "999999"
        else:
            altered["config"]["participations"] = ["1"]
        assert not replay.replay_report(altered, lab="fill_ledger")["matched"]
    for key in ("source_sha256", "dependency_sha256"):
        altered = copy.deepcopy(report)
        if key == "source_sha256":
            altered["provenance"][key] = "0" * 64
        else:
            altered["provenance"][key]["fill_ledger.py"] = "0" * 64
        with pytest.raises(ValueError, match="source hash"):
            replay.replay_report(altered, lab="fill_ledger")


def test_runner_retains_exact_serialization_and_refuses_overwrite(tmp_path):
    runner = load_module("run_offline_lab")
    assert "fill_ledger" in runner.replay.LABS, "register the reviewed lab"
    config = ROOT / "experiments/configs/offline-smoke/fill_ledger.json"
    target = tmp_path / "ledger.json"
    runner.run_configured_lab(lab="fill_ledger", config=config, output=target)
    original = target.read_bytes()
    with pytest.raises((ValueError, FileExistsError)):
        runner.run_configured_lab(lab="fill_ledger", config=config, output=target)
    assert target.read_bytes() == original

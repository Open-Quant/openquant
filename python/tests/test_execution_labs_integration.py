"""New offline execution controls must participate in the shared evidence workflow."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def replay_module():
    spec = importlib.util.spec_from_file_location(
        "_execution_lab_integration_replay", ROOT / "experiments/replay_offline_report.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("name", ["execution_timing", "terminal_liquidation"])
def test_execution_lab_registered_with_complete_dependency_contract(name):
    replay = replay_module()
    assert replay.LABS[name] == f"{name}_lab.py"
    assert replay.DEPENDENCIES[name] == (
        ("self_financing_lab.py",) if name == "terminal_liquidation" else ()
    )


@pytest.mark.parametrize("name", ["execution_timing", "terminal_liquidation"])
def test_execution_lab_roundtrip_and_scientific_tamper_detection(name):
    replay = replay_module()
    config = json.loads((ROOT / f"experiments/configs/offline-smoke/{name}.json").read_text())
    module = replay._load_lab(name)
    report = json.loads(json.dumps(module.run_lab(**config), allow_nan=False))
    assert replay.check_report_identity(report, lab=name)["identity_matched"]
    assert replay.replay_report(report, lab=name)["matched"]
    tampered = copy.deepcopy(report)
    tampered["unrecorded_research_conclusion"] = "profitable"
    verdict = replay.replay_report(tampered, lab=name)
    assert verdict["matched"] is False
    assert verdict["first_difference"] is not None
    tampered = copy.deepcopy(report)
    tampered["provenance"]["source_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="source hash"):
        replay.replay_report(tampered, lab=name)


def test_terminal_liquidation_rejects_modified_dependency_identity():
    replay = replay_module()
    module = replay._load_lab("terminal_liquidation")
    config = json.loads(
        (ROOT / "experiments/configs/offline-smoke/terminal_liquidation.json").read_text()
    )
    report = module.run_lab(**config)
    report["provenance"]["dependency_sha256"]["self_financing_lab.py"] = "0" * 64
    with pytest.raises(ValueError, match="dependency source hash"):
        replay.replay_report(report, lab="terminal_liquidation")

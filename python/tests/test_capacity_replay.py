"""Capacity evidence must replay fully, including infeasibility and boundary status."""

import copy
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "experiments"


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / (name + ".py"))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_capacity_report_replays_exactly_with_full_input_convention():
    lab = load("capacity_scaling_lab")
    replay = load("replay_offline_report")
    assert "capacity_scaling" in replay.LABS, "capacity lab is not registered for replay"
    report = lab.run_lab(
        scales=[100000], volume_scales=[1], impact_coefficients=[0.1], gross_returns=[0.001]
    )
    assert report["cells"][0]["participation_boundary_rounded"]
    assert not report["cells"][0]["within_participation_cap"]
    assert replay.replay_report(report, lab="capacity_scaling")["matched"]


def test_changing_capacity_feasibility_or_threshold_fails_replay():
    lab = load("capacity_scaling_lab")
    replay = load("replay_offline_report")
    assert "capacity_scaling" in replay.LABS, "capacity lab is not registered for replay"
    report = lab.run_lab(
        scales=[100000], volume_scales=[1], impact_coefficients=[0.1], gross_returns=[0.001]
    )
    a = copy.deepcopy(report)
    a["cells"][0]["within_participation_cap"] = True
    assert not replay.replay_report(a, lab="capacity_scaling")["matched"]
    b = copy.deepcopy(report)
    b["cells"][0]["break_even"]["scale"] *= 2
    assert not replay.replay_report(b, lab="capacity_scaling")["matched"]

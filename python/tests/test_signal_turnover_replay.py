"""Replay must retain whole-path outputs and their actual independent-unit SE."""

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


def test_signal_report_replays_all_paths_and_controls():
    lab, replay = load("signal_turnover_lab"), load("replay_offline_report")
    assert "signal_turnover" in replay.LABS
    report = lab.run_lab(
        repetitions=4, horizon=8, rhos=[-0.5, 0.9], etas=[0, 1], betas=[0], cost_rates=[0.001]
    )
    assert len(report["cells"]) == 8
    assert replay.replay_report(report, lab="signal_turnover")["matched"]


def test_changed_path_or_standard_error_fails_replay():
    lab, replay = load("signal_turnover_lab"), load("replay_offline_report")
    assert "signal_turnover" in replay.LABS
    report = lab.run_lab(repetitions=4, horizon=8, rhos=[0], etas=[1], betas=[0], cost_rates=[0])
    a = copy.deepcopy(report)
    a["cells"][0]["per_replication"]["gross_pnl"][0] += 0.001
    assert not replay.replay_report(a, lab="signal_turnover")["matched"]
    b = copy.deepcopy(report)
    b["cells"][0]["summary"]["gross_pnl"]["mc_se"] = 0
    assert not replay.replay_report(b, lab="signal_turnover")["matched"]

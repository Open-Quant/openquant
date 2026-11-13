"""Stateful backtest segments need an explicit, paid-for boundary transition."""

from __future__ import annotations

import copy
import importlib.util
import sys
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from fractions import Fraction as F
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def module(name):
    path = ROOT / f"experiments/{name}.py"
    assert path.exists(), f"missing {name} implementation"
    spec = importlib.util.spec_from_file_location(f"_test_{name}", path)
    assert spec is not None and spec.loader is not None
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    return m


def tape(m, volume=2, policy="liquidate_at_declared_bid"):
    from experiments import fill_ledger as ledger

    at = datetime(2026, 1, 1, tzinfo=UTC)
    state = ledger.InitialState(at, F(100), F(0), F(10))
    d = ledger.Decision("buy", at, at, at, at + timedelta(seconds=1), F(2))
    entry = ledger.MarketBucket(at + timedelta(seconds=1), F(11), F("9.5"), F(10), F(10))
    terminal = ledger.MarketBucket(at + timedelta(seconds=2), F(11), F("10.5"), F(11), F(volume))
    result = ledger.run_ledger(
        state,
        [d],
        [entry],
        fixed_fee=1,
        terminal_policy=policy,
        terminal_bucket=terminal if policy != "mark_only" else None,
    )
    return result, at + timedelta(seconds=2)


def test_cash_handoff_preserves_realized_cash_and_units_without_deposit():
    m = module("execution_segments_lab")
    result, at = tape(m)
    next_state = m.cash_handoff(result, at=at, mark=F(11))
    assert next_state.cash == 99 and next_state.inventory == 0
    assert next_state.instrument == "SYNTH" and next_state.currency == "USD"


@pytest.mark.parametrize("volume", [0, 1])
def test_partial_terminal_cannot_reset_into_cash_only_segment(volume):
    m = module("execution_segments_lab")
    result, at = tape(m, volume=volume)
    with pytest.raises(ValueError, match="inventory|liquidation"):
        m.cash_handoff(result, at=at, mark=F(11))


def test_mark_only_policy_cannot_be_reinterpreted_as_cash():
    m = module("execution_segments_lab")
    result, at = tape(m, policy="mark_only")
    with pytest.raises(ValueError, match="liquidation"):
        m.cash_handoff(result, at=at, mark=F(11))


def test_handoff_rejects_time_reversal_and_inconsistent_terminal_summary():
    m = module("execution_segments_lab")
    result, at = tape(m)
    with pytest.raises(ValueError, match="timestamp"):
        m.cash_handoff(result, at=at - timedelta(seconds=1), mark=F(11))
    with pytest.raises(ValueError, match="summary"):
        m.cash_handoff(
            replace(result, terminal=replace(result.terminal, cash=F(101))), at=at, mark=F(11)
        )


def test_unfinished_decisions_need_an_explicit_transition_policy():
    m = module("execution_segments_lab")
    result, at = tape(m)
    with pytest.raises(ValueError, match="decisions"):
        m.cash_handoff(replace(result, decisions_completed=False), at=at, mark=F(11))


def test_segment_lab_retains_rejected_handoffs_and_catches_naive_stitching():
    report = module("execution_segments_lab").run_lab()
    assert [s["handoff_accepted"] for s in report["scenarios"]] == [False, False, True, True]
    assert report["controls"]["cash_after_paid_boundary"]["observed"] == "99"
    assert report["negative_controls"]["omit_boundary_costs"]["wrong_final_equity"] == "101"
    assert report["negative_controls"]["carry_unrequested_stock"]["wrong_final_equity"] == "103"
    assert all(c["passed"] for c in report["controls"].values())
    assert all(c["detected"] for c in report["negative_controls"].values())


def test_segment_lab_replays_and_includes_all_transitive_sources():
    replay = module("replay_offline_report")
    assert replay.LABS.get("execution_segments") == "execution_segments_lab.py"
    assert replay.DEPENDENCIES["execution_segments"] == ("fill_ledger.py",)
    report = replay._load_lab("execution_segments").run_lab()
    assert replay.replay_report(report, lab="execution_segments")["matched"]
    changed = copy.deepcopy(report)
    changed["scenarios"][2]["second_segment"]["terminal"]["cash"] = "101"
    assert not replay.replay_report(changed, lab="execution_segments")["matched"]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"terminal_volumes": "12"},
        {"terminal_volumes": []},
        {"terminal_volumes": [-1]},
        {"fixed_fee": -1},
        {"fixed_fee": float("nan")},
    ],
)
def test_malformed_segment_study_inputs_fail(kwargs):
    with pytest.raises(ValueError):
        module("execution_segments_lab").run_lab(**kwargs)

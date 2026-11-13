"""Exact pre-trade envelope tests; expectations are independent arithmetic."""

from __future__ import annotations

import copy
import importlib
import itertools
import json
import sys
from datetime import UTC, datetime, timedelta
from fractions import Fraction as F
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
T = datetime(2026, 1, 1, tzinfo=UTC)


def module():
    assert (ROOT / "experiments/cash_budget_lab.py").exists(), "cash budget lab missing"
    return importlib.import_module("experiments.cash_budget_lab")


def budget(**kwargs):
    args = dict(
        cash=100,
        desired_shares=20,
        price_cap=10,
        lot_size=1,
        fixed_fee=1,
        fee_rate="0.01",
        max_fills=2,
        information_at=T,
        decision_at=T,
    )
    args.update(kwargs)
    return module().size_buy(**args)


def test_hand_budget_reserves_each_fill_and_rounds_down():
    b = budget()
    assert b.shares == 9
    assert b.worst_case_spend == F("92.9")
    assert b.cash_floor == F("7.1")
    assert b.price_cap == 10 and b.max_fills == 2


def test_every_grid_budget_is_maximal_under_declared_envelope():
    for cash, desired, lot, fee, rate, count in itertools.product(
        [0, 5, 25, 100],
        [0, F("0.5"), 12],
        [F("0.25"), 1, 3],
        [0, 2],
        [0, F("0.1")],
        [1, 3],
    ):
        b = budget(
            cash=cash,
            desired_shares=desired,
            lot_size=lot,
            fixed_fee=fee,
            fee_rate=rate,
            max_fills=count,
        )
        candidates = [lot * i for i in range(int(F(desired) // lot) + 1)]
        feasible = [q for q in candidates if q == 0 or q * 10 * (1 + rate) + count * fee <= cash]
        assert b.shares == max(feasible)
        assert b.shares % lot == 0 and b.shares <= desired
        assert b.cash_floor >= 0
        if b.shares == 0:
            assert b.worst_case_spend == 0 and b.cash_floor == cash


@pytest.mark.parametrize(
    "kwargs,name",
    [
        ({"cash": -1}, "cash"),
        ({"desired_shares": -1}, "desired_shares"),
        ({"price_cap": 0}, "price_cap"),
        ({"lot_size": 0}, "lot_size"),
        ({"fixed_fee": -1}, "fixed_fee"),
        ({"fee_rate": -1}, "fee_rate"),
        ({"cash_reserve": -1}, "cash_reserve"),
        ({"cash_reserve": 101}, "cash_reserve"),
        ({"cash": True}, "cash"),
        ({"price_cap": float("inf")}, "price_cap"),
        ({"max_fills": 0}, "max_fills"),
        ({"max_fills": True}, "max_fills"),
        ({"max_fills": 2.0}, "max_fills"),
        ({"max_fills": "2"}, "max_fills"),
        ({"information_at": T + timedelta(seconds=1)}, "information_at"),
        ({"decision_at": T.replace(tzinfo=None)}, "decision_at"),
    ],
)
def test_bad_contract_inputs_rejected(kwargs, name):
    with pytest.raises(ValueError, match=name):
        budget(**kwargs)


def test_reserve_cash_and_decimal_extremes_are_exact():
    assert budget(cash_reserve=20).shares == 7
    assert budget(cash=1e-200, price_cap=1e-201, fixed_fee=0, fee_rate=0).shares == 10
    assert budget(cash=1e200, price_cap=1e199, fixed_fee=0, fee_rate=0).shares == 10
    assert budget(cash=2, fixed_fee=1, max_fills=2).shares == 0


def fill(shares=4, price=10, fee="1.4", second=1):
    from experiments.fill_ledger import Fill

    return Fill("buy", T + timedelta(seconds=second), F(shares), F(price), F(fee), False)


def test_audit_preserves_conditional_guarantee_and_completion_separately():
    b = budget()
    a = module().audit_fills(b, [fill(), fill(5, fee="1.5", second=2)])
    assert a["within_envelope"] and a["order_completed"]
    assert a["actual_spend"] == "929/10" and a["cash_after"] == "71/10"
    partial = module().audit_fills(b, [fill()])
    assert partial["within_envelope"] and not partial["order_completed"]
    assert partial["remaining_shares"] == "5"
    assert module().audit_fills(b, [])["actual_spend"] == "0"


@pytest.mark.parametrize(
    "fills,reason",
    [
        ([fill(price=11, fee="1.44")], "price_cap"),
        ([fill(3, fee="1.3", second=i) for i in (1, 2, 3)], "max_fills"),
        ([fill(10, fee=2)], "shares"),
        ([fill(fee=2)], "commission"),
        ([fill(second=0)], "decision_at"),
        ([fill(second=2), fill(second=1)], "chronology"),
        ([fill(-1)], "positive"),
    ],
)
def test_audit_records_broken_assumptions_instead_of_certifying(fills, reason):
    a = module().audit_fills(budget(), fills)
    assert not a["within_envelope"]
    assert any(reason in value for value in a["violations"])


def test_lab_preserves_no_orders_partial_fills_and_negative_controls():
    r = module().run_lab()
    assert len(r["scenarios"]) == 72
    assert all(s["audit"]["within_envelope"] for s in r["scenarios"])
    assert any(s["budget"]["shares"] == "0" for s in r["scenarios"])
    assert any(not s["audit"]["order_completed"] for s in r["scenarios"])
    assert all(v["detected"] for v in r["negative_controls"].values())
    assert len(r["negative_controls"]) == 4
    json.dumps(r, allow_nan=False)


def test_registered_lab_replay_and_mutations(tmp_path):
    replay = importlib.import_module("experiments.replay_offline_report")
    runner = importlib.import_module("experiments.run_offline_lab")

    assert replay.LABS.get("cash_budget") == "cash_budget_lab.py", "register cash budget"
    assert replay.DEPENDENCIES["cash_budget"] == ("fill_ledger.py",)
    cfg = ROOT / "experiments/configs/offline-smoke/cash_budget.json"
    out = tmp_path / "cash.json"
    runner.run_configured_lab(lab="cash_budget", config=cfg, output=out)
    report = replay.read_report(out)
    assert replay.replay_report(report, lab="cash_budget")["matched"]
    for key in ("cash_floor", "shares"):
        altered = copy.deepcopy(report)
        altered["scenarios"][0]["budget"][key] = "999"
        assert not replay.replay_report(altered, lab="cash_budget")["matched"]
    altered = copy.deepcopy(report)
    altered["provenance"]["dependency_sha256"]["fill_ledger.py"] = "0" * 64
    with pytest.raises(ValueError, match="dependency source hash"):
        replay.replay_report(altered, lab="cash_budget")
    with pytest.raises((ValueError, FileExistsError)):
        runner.run_configured_lab(lab="cash_budget", config=cfg, output=out)


@pytest.mark.parametrize("kwargs", [{"cash_values": "100"}, {"lot_sizes": []}, {"price_caps": [0]}])
def test_lab_rejects_malformed_grid(kwargs):
    with pytest.raises(ValueError):
        module().run_lab(**kwargs)


def test_negative_controls_do_not_hide_unrelated_ledger_errors(monkeypatch):
    m = module()

    def fail(*args, **kwargs):
        raise ValueError("unrelated malformed quote")

    monkeypatch.setattr(m, "_execute", fail)
    with pytest.raises(ValueError, match="unrelated malformed quote"):
        m._negative_controls(T)

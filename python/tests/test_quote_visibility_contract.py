"""Original hand calculations and causal controls for an offline quote contract."""

import copy
import importlib
import importlib.util
import json
import subprocess
import sys

import pytest


def scenario():
    return {
        "schema": "openquant_quote_visibility_v1",
        "cash": "100",
        "inventory": "0",
        "fixed_fee": "1",
        "fee_rate": "1/1000",
        "order": {
            "side": "buy",
            "shares": 3,
            "decision_at": 0,
            "earliest_fill_at": 1,
            "deadline": 3,
            "limit": "10",
        },
        "buckets": [
            {
                "at": 1,
                "available_at": 1,
                "bid": "9",
                "ask": "10",
                "capacity": 0,
                "reference_close": "8",
            },
            {
                "at": 2,
                "available_at": 2,
                "bid": "8",
                "ask": "9",
                "capacity": 2,
                "reference_close": "8",
            },
            {
                "at": 3,
                "available_at": 3,
                "bid": "9",
                "ask": "10",
                "capacity": 1,
                "reference_close": "8",
            },
            {
                "at": 4,
                "available_at": 4,
                "bid": "10",
                "ask": "11",
                "capacity": 9,
                "reference_close": "8",
            },
        ],
        "marks": [
            {"observed_at": 0, "available_at": 0, "price": "8"},
            {"observed_at": 2, "available_at": 3, "price": "12"},
        ],
    }


def replay(doc):
    return importlib.import_module("experiments.quote_visibility_contract").replay(doc)


def test_module_exists():
    assert importlib.util.find_spec("experiments.quote_visibility_contract") is not None


def test_limit_eligibility_capacity_and_mark_are_separate():
    r = replay(scenario())
    assert r["external_engine_validated"] is False
    assert r["synthetic_only"] is True
    first, second, third, final = r["rows"]
    assert (first["limit_eligible"], first["eligible"], first["reason"]) == (
        True,
        True,
        "no_capacity",
    )
    assert first["fill_shares"] == 0 and first["commission"] == "0"
    assert (second["fill_shares"], second["fill_price"], second["cash"]) == (2, "9", "40491/500")
    assert (second["mark"], second["equity"]) == ("8", "48491/500")
    # 100 - (18 + 1 + .018) - (10 + 1 + .010) = 69.972; mark adds 36.
    assert (third["fill_shares"], third["cash"], third["equity"]) == (1, "17493/250", "26493/250")
    assert final["status"] == "completed" and final["remaining"] == 0 and not final["active"]


def test_sell_uses_bid_not_ask_or_mark_and_respects_limit():
    d = scenario()
    d["inventory"] = "3"
    d["order"].update(side="sell", limit="9")
    rows = replay(d)["rows"]
    assert rows[1]["reason"] == "limit_not_met" and rows[1]["fill_shares"] == 0
    assert rows[2]["fill_shares"] == -1 and rows[2]["fill_price"] == "9"
    assert rows[2]["cash"] == "107991/1000" and rows[2]["inventory"] == "2"
    assert rows[3]["status"] == "expired" and rows[3]["inventory"] == "2"


def test_mark_change_is_not_cash_or_fill_change():
    a = scenario()
    b = copy.deepcopy(a)
    b["marks"][1]["price"] = "99"
    ra, rb = replay(a), replay(b)
    assert ra["fills"] == rb["fills"]
    assert [x["cash"] for x in ra["rows"]] == [x["cash"] for x in rb["rows"]]
    assert ra["rows"][:2] == rb["rows"][:2]
    assert ra["rows"][2]["equity"] != rb["rows"][2]["equity"]


def test_same_reference_close_different_quotes_change_execution():
    a = scenario()
    b = copy.deepcopy(a)
    b["buckets"][1]["ask"] = "11"
    assert replay(a)["rows"][1]["fill_shares"] == 2
    assert replay(b)["rows"][1]["reason"] == "limit_not_met"
    c = copy.deepcopy(a)
    for bucket in c["buckets"]:
        bucket["reference_close"] = "1000"
    assert replay(a) == replay(c)


def test_delayed_quote_opportunity_is_missed_not_backfilled():
    d = scenario()
    d["buckets"][1]["available_at"] = 3
    rows = replay(d)["rows"]
    assert rows[1]["reason"] == "quote_unavailable" and rows[1]["limit_eligible"] is None
    assert rows[2]["fill_shares"] == 1
    assert rows[3]["status"] == "expired" and rows[3]["remaining"] == 2


def test_late_old_mark_does_not_replace_newer_visible_mark():
    d = scenario()
    d["marks"] = [
        {"observed_at": 0, "available_at": 3, "price": "100"},
        {"observed_at": 1, "available_at": 1, "price": "8"},
    ]
    assert [x["mark"] for x in replay(d)["rows"]] == ["8"] * 4


def test_no_mark_with_inventory_is_unknown_not_cash_equity():
    d = scenario()
    d["marks"] = []
    rows = replay(d)["rows"]
    assert rows[0]["equity"] == "100"
    assert rows[1]["equity"] is None and rows[1]["mark"] is None


def test_decision_view_has_arrivals_before_decision_but_no_future():
    d = scenario()
    d["order"].update(decision_at=2, earliest_fill_at=2)
    d["buckets"][1]["available_at"] = 3
    view = replay(d)["decision_view"]
    assert view["quote_at"] == 1 and view["ask"] == "10"
    assert view["mark_observed_at"] == 0 and view["mark"] == "8"
    assert replay(d)["rows"][1]["reason"] == "outside_window"
    d["buckets"][1]["available_at"] = 2
    assert replay(d)["decision_view"]["quote_at"] == 2
    assert replay(d)["rows"][1]["fill_shares"] == 0


def test_feed_end_at_deadline_can_leave_active_residual():
    d = scenario()
    d["order"]["shares"] = 5
    full = replay(d)
    d["buckets"] = d["buckets"][:3]
    short = replay(d)
    assert short["rows"][-1]["fill_shares"] == 1
    assert short["rows"][-1]["active"] and short["rows"][-1]["remaining"] == 2
    assert full["rows"][-1]["status"] == "expired"
    assert full["rows"][:3] == short["rows"]


def test_future_quote_and_mark_perturbation_preserves_prefix():
    d = scenario()
    changed = copy.deepcopy(d)
    changed["buckets"][3].update(bid="500", ask="600", capacity=100)
    changed["marks"].append({"observed_at": 4, "available_at": 5, "price": "900"})
    assert replay(d)["rows"][:3] == replay(changed)["rows"][:3]
    assert replay(d)["decision_view"] == replay(changed)["decision_view"]


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d.update(cash=100.0),
        lambda d: d.update(cash="NaN"),
        lambda d: d.update(fee_rate="2"),
        lambda d: d["order"].update(shares=True),
        lambda d: d["order"].update(deadline=0),
        lambda d: d["order"].update(limit="0"),
        lambda d: d["buckets"][0].update(bid="11"),
        lambda d: d["buckets"][0].update(available_at=0),
        lambda d: d["buckets"][1].update(at=1),
        lambda d: d["buckets"][0].update(capacity=-1),
        lambda d: d["marks"].append(dict(d["marks"][0])),
        lambda d: d["marks"][0].update(available_at=-1),
        lambda d: d.update(unrecognized=True),
    ],
)
def test_invalid_contract_rejected(mutate):
    d = scenario()
    mutate(d)
    with pytest.raises(ValueError):
        replay(d)


@pytest.mark.parametrize(
    "side,cash,inventory,reason",
    [
        ("buy", "1", "0", "cash"),
        ("sell", "100", "0", "inventory"),
    ],
)
def test_refusal_does_not_silently_clip_size(side, cash, inventory, reason):
    d = scenario()
    d.update(cash=cash, inventory=inventory)
    d["order"].update(side=side, limit=None)
    with pytest.raises(ValueError, match=reason):
        replay(d)


@pytest.mark.parametrize("value", ["1e2", " 100 ", "1_00", "١٠٠"])
def test_only_bounded_ascii_rational_or_decimal_grammar(value):
    d = scenario()
    d["cash"] = value
    with pytest.raises(ValueError, match="rational"):
        replay(d)


def test_cli_runs_the_bundled_demo():
    result = subprocess.run(
        [sys.executable, "-m", "experiments.quote_visibility_contract"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == replay(scenario())


def test_cli_rejects_duplicate_json_fields(tmp_path):
    path = tmp_path / "duplicate.json"
    path.write_text(json.dumps(scenario()).replace('"cash": "100"', '"cash":"1", "cash":"100"'))
    result = subprocess.run(
        [sys.executable, "-m", "experiments.quote_visibility_contract", str(path)],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "duplicate" in result.stderr


def comparison(doc):
    module = importlib.import_module("experiments.quote_visibility_contract")
    assert hasattr(module, "compare_accounting_subset")
    return module.compare_accounting_subset(doc)


def test_existing_ledger_matches_representable_accounting_not_lifecycle():
    d = scenario()
    d["order"].update(limit=None, shares=5)
    r = comparison(d)
    assert r["status"] == "matched_accounting_subset"
    assert r["external_engine_validated"] is False and r["full_contract_compatible"] is False
    assert r["differing_fields"] == []
    assert r["lifecycle"] == {"quote_contract": "expired", "fill_ledger": "cancelled_at_deadline"}
    assert r["source_sha256"].keys() == {"quote_visibility_contract", "fill_ledger"}


def test_comparison_exposes_horizon_liveness_difference():
    d = scenario()
    d["order"].update(limit=None, shares=5)
    d["buckets"] = d["buckets"][:2]
    r = comparison(d)
    assert r["status"] == "matched_accounting_subset"
    assert r["lifecycle"] == {"quote_contract": "partial", "fill_ledger": "cancelled_at_horizon"}


@pytest.mark.parametrize(
    "mutate,reason",
    [
        (lambda d: None, "limit_order"),
        (
            lambda d: (d["order"].update(limit=None), d["buckets"][1].update(available_at=3)),
            "delayed_quote",
        ),
        (lambda d: (d["order"].update(limit=None), d.update(marks=[])), "initial_mark_unavailable"),
        (
            lambda d: (d["order"].update(limit=None), d["buckets"][0].update(at=0, available_at=0)),
            "bucket_not_after_initial_state",
        ),
    ],
)
def test_ledger_comparison_does_not_translate_unsupported_contract(mutate, reason):
    d = scenario()
    mutate(d)
    r = comparison(d)
    assert r["status"] == "unsupported" and reason in r["reasons"]
    assert r["full_contract_compatible"] is False


def test_ledger_accounting_comparison_catches_changed_observation(monkeypatch):
    from dataclasses import replace

    from experiments import fill_ledger

    original = fill_ledger.run_ledger

    def wrong(*args, **kwargs):
        result = original(*args, **kwargs)
        rows = list(result.rows)
        rows[-1] = replace(rows[-1], equity=rows[-1].equity + 1)
        return replace(result, rows=tuple(rows))

    monkeypatch.setattr(fill_ledger, "run_ledger", wrong)
    d = scenario()
    d["order"]["limit"] = None
    r = comparison(d)
    assert r["status"] == "different_accounting_subset"
    assert "equity" in r["differing_fields"]


@pytest.mark.parametrize("field", ["shares", "at"])
def test_comparison_never_truncates_fractional_fill_observations(monkeypatch, field):
    from dataclasses import replace
    from datetime import timedelta
    from fractions import Fraction

    from experiments import fill_ledger

    original = fill_ledger.run_ledger

    def wrong(*args, **kwargs):
        result = original(*args, **kwargs)
        fills = list(result.fills)
        fills[0] = (
            replace(fills[0], shares=fills[0].shares + Fraction(1, 2))
            if field == "shares"
            else replace(fills[0], at=fills[0].at + timedelta(microseconds=1))
        )
        return replace(result, fills=tuple(fills))

    monkeypatch.setattr(fill_ledger, "run_ledger", wrong)
    d = scenario()
    d["order"]["limit"] = None
    assert "fills" in comparison(d)["differing_fields"]


def test_comparison_checks_no_fill_row_time(monkeypatch):
    from dataclasses import replace
    from datetime import timedelta

    from experiments import fill_ledger

    original = fill_ledger.run_ledger

    def wrong(*args, **kwargs):
        result = original(*args, **kwargs)
        rows = list(result.rows)
        assert rows[0].executed_volume == 0
        rows[0] = replace(rows[0], at=rows[0].at + timedelta(microseconds=1))
        return replace(result, rows=tuple(rows))

    monkeypatch.setattr(fill_ledger, "run_ledger", wrong)
    d = scenario()
    d["order"]["limit"] = None
    assert "at" in comparison(d)["differing_fields"]

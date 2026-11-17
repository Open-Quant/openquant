"""Small hand-calculated and seeded controls; no external parity assertion."""

import copy
import math
import random
from fractions import Fraction as Q
from typing import Any

import pytest

pytest.importorskip("backtrader", reason="optional pinned external-engine probe")

from experiments.backtrader_admission_probe import audit_trace, observe, run


def one(**changes):
    args = dict(
        cash=25,
        prices=[10, 10, 8, 1],
        volumes=[0, 2, 3, 10],
        orders=[dict(id="large", bar=0, shares=5)],
        fixed=1,
        rate=0.001,
    )
    args.update(changes)
    return observe(**args)


def test_same_cycle_shadow_cash_is_not_live_cash():
    result = one(orders=[dict(id="large", bar=0, shares=5), dict(id="small", bar=0, shares=1)])
    assert [x["status"] for x in result["bars"][-1]["orders"]] == ["Margin", "Margin"]
    first, second = result["execution_attempts"]
    assert math.isclose(first["cash_after"], -26.05)
    assert math.isclose(second["cash_before"], -26.05)
    assert math.isclose(second["cash_after"], -37.06)
    assert (first["inventory_before"], first["inventory_after"]) == (0, 5)
    assert (second["inventory_before"], second["inventory_after"]) == (5, 6)
    assert all(x["cash"] == 25 and x["inventory"] == 0 for x in result["bars"])
    assert audit_trace(result)["terminal_residual_ids"] == ["large", "small"]


def test_order_and_cycle_controls_are_distinct():
    large, small = dict(id="large", bar=0, shares=5), dict(id="small", bar=0, shares=1)
    reversed_result = one(orders=[small, large])
    next_cycle = one(orders=[large, dict(small, bar=1)])
    assert reversed_result["bars"][-1]["cash"] == 13.99
    # The later order fills at the next bar's price 8, not its creation price 10.
    assert math.isclose(next_cycle["bars"][-1]["cash"], 15.992)
    for result in (reversed_result, next_cycle):
        assert result["bars"][-1]["inventory"] == 1
        assert audit_trace(result)["terminal_residual_ids"] == ["large"]


def test_disabling_precheck_does_not_make_residual_retryable():
    result = one(checksubmit=False)
    assert math.isclose(result["bars"][-1]["cash"], 3.98, abs_tol=1e-12)
    assert result["bars"][-1]["inventory"] == 2
    assert [x["bar"] for x in result["fills"]] == [1]
    final = result["bars"][-1]["orders"][0]
    assert (final["status"], final["remaining"], final["alive"]) == ("Margin", 3, False)
    assert audit_trace(result)["terminal_residual_ids"] == ["large"]


def test_whole_order_one_fee_is_not_a_budget_for_repeated_fill_fees():
    result = one(cash=51.5, prices=[10, 10, 10, 1])
    assert math.isclose(result["execution_attempts"][0]["cash_after"], 0.45)
    assert math.isclose(result["bars"][-1]["cash"], 30.48)
    assert result["bars"][-1]["orders"][0]["status"] == "Margin"
    enough = one(cash=52.1, prices=[10, 10, 10, 1])
    assert enough["bars"][-1]["orders"][0]["status"] == "Completed"
    assert math.isclose(enough["bars"][-1]["cash"], 0.05, abs_tol=1e-12)


def test_exact_decimal_boundary_is_preserved_not_silently_relaxed():
    result = one(cash=51.05)
    assert Q("51.05") - 5 * 10 - 1 - Q(".001") * 50 == 0
    assert result["execution_attempts"][0]["cash_after"] < 0
    assert result["bars"][-1]["orders"][0]["status"] == "Margin"
    assert not result["fills"]


def test_truncation_is_active_residual_not_terminal_refusal():
    result = one(checksubmit=False, prices=[10, 10], volumes=[0, 2])
    analysis = audit_trace(result)
    assert analysis["active_residual_ids"] == ["large"]
    assert not analysis["terminal_residual_ids"]
    assert analysis["completed_ids"] == []
    assert result["bars"][-1]["orders"][0]["alive"] is True


def test_pending_end_has_an_undelivered_emission():
    result = one(orders=[dict(id="last", bar=3, shares=1)])
    analysis = audit_trace(result)
    assert analysis["active_residual_ids"] == ["last"]
    assert len(result["broker_events"]) == 1
    assert not result["notifications"]
    assert analysis["undelivered_notifications"] == 1


def test_volume_limit_is_per_order_not_shared_capacity():
    result = one(
        cash=100,
        volumes=[0, 2, 0, 0],
        orders=[dict(id="a", bar=0, shares=2), dict(id="b", bar=0, shares=2)],
    )
    assert sum(x["size"] for x in result["fills"] if x["bar"] == 1) == 4
    analysis = audit_trace(result)
    assert analysis["shared_volume_excess_bars"] == [1]
    assert analysis["external_engine_validated"] is False


def test_notification_timestamps_do_not_move_fills():
    result = one(cash=100)
    assert result["broker_events"][0]["bar"] == 0
    assert result["notifications"][0]["delivered_bar"] == 1
    assert result["notifications"][0]["filled"] == 0
    assert [(x["bar"], x["size"]) for x in result["fills"]] == [(1, 2), (2, 3)]
    assert audit_trace(result)["consistent"] is True


def test_random_small_paths_reconcile_with_independent_rational_fill_ledger():
    rng = random.Random(20261117)
    for _ in range(200):
        prices = [rng.randint(1, 20) for _ in range(6)]
        volumes = [rng.randint(0, 5) for _ in range(6)]
        config: dict[str, Any] = dict(
            cash=rng.randint(0, 120),
            prices=prices,
            volumes=volumes,
            fixed=rng.randint(0, 2),
            rate=rng.choice([0, 0.001]),
            orders=[
                dict(id=str(i), bar=rng.randint(0, 4), shares=rng.randint(1, 8))
                for i in range(rng.randint(1, 4))
            ],
            checksubmit=rng.choice([True, False]),
        )
        result = observe(**config)
        assert audit_trace(result)["consistent"]
        cash, inventory = Q(config["cash"]), 0
        for bar in result["bars"]:
            for fill in result["fills"]:
                if fill["bar"] == bar["bar"]:
                    n, price = fill["size"], Q(str(fill["price"]))
                    fee = Q(config["fixed"]) + n * price * Q(str(config["rate"]))
                    cash -= n * price + fee
                    inventory += n
            assert math.isclose(bar["cash"], float(cash), rel_tol=1e-12, abs_tol=1e-10)
            assert bar["inventory"] == inventory


@pytest.mark.parametrize(
    "field,value",
    [
        ("checksubmit", 1),
        ("cash", True),
        ("cash", float("nan")),
        ("cash", -1),
        ("cash", 0),
        ("cash", 5e-324),
        ("cash", 10**1000),
        ("cash", 1e13),
        ("rate", 2),
        ("fixed", float("inf")),
        ("prices", [10, 0]),
        ("volumes", [0, 2.0, 3, 1]),
        ("orders", []),
        ("orders", [dict(id="x", bar=True, shares=1)]),
        ("orders", [dict(id="x", bar=0, shares=False)]),
        ("orders", [dict(id="x", bar=0, shares=1), dict(id="x", bar=1, shares=1)]),
    ],
)
def test_invalid_inputs_fail_before_observation(field, value):
    with pytest.raises(ValueError):
        one(**{field: value})


@pytest.mark.parametrize(
    "mutation",
    [
        lambda x: x.update(schema="unknown"),
        lambda x: x.update(external_engine_validated=True),
        lambda x: x["fills"][0].update(bar=0),
        lambda x: x["fills"][0].update(commission=0),
        lambda x: x["fills"][0].update(size=3),
        lambda x: x["bars"][-1].update(cash=0),
        lambda x: x["bars"][-1]["orders"][0].update(remaining=1),
        lambda x: x["notifications"][0].update(filled=2),
        lambda x: x["notifications"][0].update(delivered_bar=0),
        lambda x: x["fills"].append(dict(x["fills"][0])),
        lambda x: x.update(broker_events=[]),
        lambda x: x.update(execution_attempts=[]),
        lambda x: x["execution_attempts"][-1].update(cash_after=0),
        lambda x: x["execution_attempts"][-1].update(remaining_after=100),
        lambda x: x["bars"][0].pop("cash"),
        lambda x: x["config"].pop("cash"),
        lambda x: x.update(fills=[None]),
        lambda x: x.update(source_sha256={}),
        lambda x: x["broker_events"][0].update(phase="execution"),
        lambda x: x["broker_events"][0].update(live_cash=float("nan")),
        lambda x: x["broker_events"][0].update(live_cash=987),
        lambda x: x["notifications"][0].update(delivered_bar=3),
        lambda x: x["notifications"].pop(),
    ],
)
def test_corrupted_trace_fails_closed(mutation):
    result = copy.deepcopy(one(cash=100))
    mutation(result)
    with pytest.raises(ValueError):
        audit_trace(result)


def test_fixed_study_cannot_emit_qualification():
    result = run()
    assert result["external_engine_validated"] is False
    assert len(result["cases"]) >= 7
    for trace in result["cases"].values():
        assert audit_trace(trace)["consistent"] is True


def test_last_bar_completion_must_be_delivered():
    result = one(cash=100, prices=[10, 10], volumes=[0, 5])
    result["notifications"].pop()
    with pytest.raises(ValueError, match="missing delivered"):
        audit_trace(result)


def test_zero_volume_attempt_cannot_be_omitted():
    result = one(cash=100, volumes=[0, 0, 5, 0])
    result["execution_attempts"] = [
        x for x in result["execution_attempts"] if not (x["phase"] == "execution" and x["bar"] == 1)
    ]
    with pytest.raises(ValueError, match="lifecycle"):
        audit_trace(result)


def test_terminal_order_cannot_have_a_later_noop_attempt():
    result = one(cash=100, volumes=[0, 5, 0, 0])
    last = result["bars"][-1]
    result["execution_attempts"].append(
        dict(
            id="large",
            bar=3,
            phase="execution",
            cash_before=last["cash"],
            cash_after=last["cash"],
            inventory_before=5,
            inventory_after=5,
            remaining_before=0,
            remaining_after=0,
        )
    )
    with pytest.raises(ValueError, match="lifecycle"):
        audit_trace(result)

"""Independent controls for the optional, pinned Backtrader study."""

from __future__ import annotations

import pytest

pytest.importorskip("backtrader")
from experiments.backtrader_sell_quote_probe import observe


@pytest.mark.parametrize(
    ("slip_match", "slip_out", "first_price", "buy_cash"),
    [(True, False, 10, 80), (True, True, 11, 78), (False, False, None, 100)],
)
def test_proxy_ask_slippage_is_not_intrinsic_quote_support(
    slip_match: bool, slip_out: bool, first_price: int | None, buy_cash: int
) -> None:
    result = observe(slip_match=slip_match, slip_out=slip_out)
    bar = result["bars"][1]
    first_fills = [fill for fill in result["fills"] if fill["bar"] == 1]
    assert [fill["price"] for fill in first_fills] == (
        [first_price] if first_price is not None else []
    )
    assert bar["cash"] == buy_cash
    assert bar["engine_value"] == buy_cash + bar["inventory"] * 9
    assert bar["reference_mark_value"] == buy_cash + bar["inventory"] * 12
    assert result["external_engine_validated"] is False


def test_sell_expiry_is_not_a_last_fill_or_liquidation() -> None:
    result = observe(slip_match=True, slip_out=False)
    assert result["fills"] == [
        {"order": "entry", "bar": 1, "size": 2, "price": 10.0},
        {"order": "exit", "bar": 2, "size": -1, "price": 9.0},
    ]
    assert [bar.get("exit_status") for bar in result["bars"]] == [
        None,
        "Submitted",
        "Partial",
        "Expired",
    ]
    assert [bar["inventory"] for bar in result["bars"]] == [0, 2, 1, 1]
    assert [bar["cash"] for bar in result["bars"]] == [100, 80, 89, 89]
    assert result["bars"][3]["exit_remaining"] == -1
    assert result["bars"][3]["exit_active"] is False


def test_same_valid_timestamp_does_not_expire_pinned_market_order() -> None:
    result = observe(slip_match=True, slip_out=False, exit_market=True)
    assert [bar["exit_status"] for bar in result["bars"]] == [
        None,
        "Submitted",
        "Partial",
        "Completed",
    ]
    assert [bar["inventory"] for bar in result["bars"]] == [0, 2, 1, 0]
    assert [fill["price"] for fill in result["fills"] if fill["order"] == "exit"] == [8, 11]
    assert result["bars"][-1]["cash"] == 99


def test_outside_slippage_proxy_can_cross_sell_limit() -> None:
    clamped = observe(slip_match=True, slip_out=False)
    outside = observe(slip_match=True, slip_out=True)
    assert (clamped["fills"][1]["price"], outside["fills"][1]["price"]) == (9, 8)
    assert outside["fills"][1]["price"] < outside["settings"]["exit_limit"]
    assert outside["bars"][-1]["exit_status"] == "Expired"


@pytest.mark.parametrize(
    ("match", "outside", "market", "expected"),
    [
        (True, False, False, {1: (2, 10), 2: (-1, 9)}),
        (True, True, False, {1: (2, 11), 2: (-1, 8)}),
        (False, False, False, {2: (1, 10)}),
        (True, False, True, {1: (2, 10), 2: (-1, 8), 3: (-1, 11)}),
    ],
)
def test_exact_cash_inventory_and_two_value_bases(
    match: bool, outside: bool, market: bool, expected: dict[int, tuple[int, int]]
) -> None:
    result = observe(slip_match=match, slip_out=outside, exit_market=market)
    assert {f["bar"]: (f["size"], f["price"]) for f in result["fills"]} == expected
    assert len(result["fills"]) == len(expected)
    cash, inventory = 100, 0
    for bar, (close, mark) in enumerate(zip([10, 9, 10, 11], [10, 12, 12, 13], strict=True)):
        shares, price = expected.get(bar, (0, 0))
        cash -= shares * price
        inventory += shares
        state = result["bars"][bar]
        assert (state["cash"], state["inventory"]) == (cash, inventory)
        assert state["engine_value"] == cash + inventory * close
        assert state["reference_mark_value"] == cash + inventory * mark
        assert state["reference_mark_value"] - state["engine_value"] == inventory * (mark - close)
    assert result["settings"]["fixed_slippage"] == 1
    assert result["settings"]["exit_valid_for_days"] == 1
    assert result["settings"]["valuation_bases"] == ["broker_close", "external_mark"]


def test_only_validity_changes_outcome_of_matched_limit_orders() -> None:
    timed = observe(slip_match=True, slip_out=False)
    untimed = observe(slip_match=True, slip_out=False, exit_no_expiry=True)
    assert timed["bars"][:3] == untimed["bars"][:3]
    assert untimed["fills"][:2] == timed["fills"]
    assert untimed["fills"][-1] == {"order": "exit", "bar": 3, "size": -1, "price": 10}
    assert (untimed["bars"][-1]["exit_status"], timed["bars"][-1]["exit_status"]) == (
        "Completed",
        "Expired",
    )
    assert untimed["bars"][-1]["cash"] == 99
    assert untimed["bars"][-1]["inventory"] == 0
    assert untimed["settings"]["exit_valid_for_days"] is None

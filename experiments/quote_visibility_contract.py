"""Exact synthetic quote/mark visibility contract, not an external broker adapter.

Each quote bucket is one instantaneous declared capacity opportunity. A quote
arriving after that time is missed, never retrospectively executed or reused.
Externally supplied orders do not establish the causality of a trading strategy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import UTC, datetime, timedelta
from fractions import Fraction as Q
from pathlib import Path
from typing import Any

SCHEMA = "openquant_quote_visibility_v1"
DEMO = Path(__file__).with_name("fixtures") / "quote_visibility_demo_v1.json"


def _fields(value: Any, keys: tuple[str, ...]) -> None:
    if not isinstance(value, dict) or set(value) != set(keys):
        raise ValueError("unexpected contract fields")


def _time(value: Any) -> int:
    if type(value) is not int or not 0 <= value <= 1_000_000:
        raise ValueError("timestamp must be an integer in [0, 1000000]")
    return value


def _amount(value: Any, *, positive: bool = False, maximum: int = 10**12) -> Q:
    if (
        not isinstance(value, str)
        or not 1 <= len(value) <= 64
        or re.fullmatch(r"[0-9]+(?:/[0-9]+|\.[0-9]+)?", value) is None
    ):
        raise ValueError("amounts must be bounded exact rational strings")
    try:
        result = Q(value)
    except (ValueError, ZeroDivisionError) as exc:
        raise ValueError("invalid exact rational amount") from exc
    if result < 0 or result > maximum or (positive and result == 0):
        raise ValueError("amount out of contract range")
    return result


def _quantity(value: Any, *, positive: bool = False) -> int:
    if type(value) is not int or not (1 if positive else 0) <= value <= 1_000_000:
        raise ValueError("quantity must be a bounded integer")
    return value


def _validate(document: Any) -> None:
    _fields(
        document,
        ("schema", "cash", "inventory", "fixed_fee", "fee_rate", "order", "buckets", "marks"),
    )
    if document["schema"] != SCHEMA:
        raise ValueError("unsupported quote contract")
    _amount(document["cash"])
    initial_inventory = _amount(document["inventory"], maximum=1_000_000)
    if initial_inventory.denominator != 1:
        raise ValueError("inventory must be whole shares")
    _amount(document["fixed_fee"], maximum=1_000_000)
    _amount(document["fee_rate"], maximum=1)
    order = document["order"]
    _fields(order, ("side", "shares", "decision_at", "earliest_fill_at", "deadline", "limit"))
    if order["side"] not in ("buy", "sell"):
        raise ValueError("unsupported order side")
    _quantity(order["shares"], positive=True)
    decision, earliest, deadline = (
        _time(order[x]) for x in ("decision_at", "earliest_fill_at", "deadline")
    )
    if not decision <= earliest <= deadline or deadline <= decision:
        raise ValueError("invalid order window")
    if order["limit"] is not None:
        _amount(order["limit"], positive=True)
    buckets, marks = document["buckets"], document["marks"]
    if not isinstance(buckets, list) or not 1 <= len(buckets) <= 64:
        raise ValueError("1 to 64 quote buckets required")
    previous = -1
    for bucket in buckets:
        _fields(bucket, ("at", "available_at", "bid", "ask", "capacity", "reference_close"))
        at, available = _time(bucket["at"]), _time(bucket["available_at"])
        if at <= previous or available < at:
            raise ValueError(
                "quote times must increase and availability cannot precede observation"
            )
        previous = at
        if _amount(bucket["bid"], positive=True) > _amount(bucket["ask"], positive=True):
            raise ValueError("crossed bid/ask")
        _amount(bucket["reference_close"], positive=True)
        _quantity(bucket["capacity"])
    if not isinstance(marks, list) or len(marks) > 64:
        raise ValueError("at most 64 independent marks required")
    observed = set()
    for mark in marks:
        _fields(mark, ("observed_at", "available_at", "price"))
        at, available = _time(mark["observed_at"]), _time(mark["available_at"])
        if at in observed or available < at:
            raise ValueError("duplicate mark or invalid mark availability")
        observed.add(at)
        _amount(mark["price"], positive=True)


def _visible_mark(marks: list[dict[str, Any]], at: int) -> dict[str, Any] | None:
    known = [x for x in marks if x["observed_at"] <= at and x["available_at"] <= at]
    return max(known, key=lambda x: x["observed_at"]) if known else None


def replay(document: dict[str, Any]) -> dict[str, Any]:
    """Replay bounded, exact-rational inputs and disclose every execution gate.

    Arrivals at a timestamp precede its decision view, but fills strictly follow
    the order decision. Deadline equality is eligible. Expiry is observed only
    at a later replay bucket; it does not liquidate inventory. Marks never affect
    fills or cash. Cash/inventory shortfall refuses the run without size clipping.
    """
    _validate(document)
    cash, inventory = Q(document["cash"]), Q(document["inventory"])
    fee, rate = Q(document["fixed_fee"]), Q(document["fee_rate"])
    order, buckets, marks = document["order"], document["buckets"], document["marks"]
    remaining = order["shares"]
    direction = 1 if order["side"] == "buy" else -1
    limit = Q(order["limit"]) if order["limit"] is not None else None
    decision = order["decision_at"]
    visible_quotes = [x for x in buckets if x["at"] <= decision and x["available_at"] <= decision]
    quote = visible_quotes[-1] if visible_quotes else None
    mark = _visible_mark(marks, decision)
    decision_view = {
        "at": decision,
        "quote_at": quote["at"] if quote else None,
        "quote_available_at": quote["available_at"] if quote else None,
        "bid": str(Q(quote["bid"])) if quote else None,
        "ask": str(Q(quote["ask"])) if quote else None,
        "mark": str(Q(mark["price"])) if mark else None,
        "mark_observed_at": mark["observed_at"] if mark else None,
        "mark_available_at": mark["available_at"] if mark else None,
    }
    rows, fills = [], []
    for bucket in buckets:
        at = bucket["at"]
        price = Q(bucket["ask"] if direction > 0 else bucket["bid"])
        visible = bucket["available_at"] <= at
        limit_ok = (
            (limit is None or (price <= limit if direction > 0 else price >= limit))
            if visible
            else None
        )
        window = decision < at and order["earliest_fill_at"] <= at <= order["deadline"]
        eligible = bool(remaining and window and visible and limit_ok)
        amount, commission = 0, Q(0)
        if not remaining:
            reason = "completed"
        elif at > order["deadline"]:
            reason = "expired"
        elif not window:
            reason = "outside_window"
        elif not visible:
            reason = "quote_unavailable"
        elif not limit_ok:
            reason = "limit_not_met"
        elif not bucket["capacity"]:
            reason = "no_capacity"
        else:
            amount = direction * min(remaining, bucket["capacity"])
            commission = fee + abs(amount) * price * rate
            next_cash, next_inventory = cash - amount * price - commission, inventory + amount
            if next_cash < 0:
                raise ValueError(f"cash shortfall at {at}")
            if next_inventory < 0:
                raise ValueError(f"inventory shortfall at {at}")
            cash, inventory = next_cash, next_inventory
            remaining -= abs(amount)
            reason = "filled"
            fills.append(
                {"at": at, "shares": amount, "price": str(price), "commission": str(commission)}
            )
        status = (
            "completed"
            if remaining == 0
            else "expired"
            if at > order["deadline"]
            else "pending"
            if at < decision
            else "partial"
            if remaining < order["shares"]
            else "active"
        )
        mark = _visible_mark(marks, at)
        mark_price = Q(mark["price"]) if mark else None
        equity = (
            cash + inventory * mark_price
            if mark_price is not None
            else cash
            if inventory == 0
            else None
        )
        rows.append(
            {
                "at": at,
                "quote_visible": visible,
                "limit_eligible": limit_ok,
                "eligible": eligible,
                "reason": reason,
                "fill_shares": amount,
                "fill_price": str(price) if amount else None,
                "commission": str(commission),
                "cash": str(cash),
                "inventory": str(inventory),
                "remaining": remaining,
                "status": status,
                "active": status in ("active", "partial"),
                "mark": str(mark_price) if mark_price is not None else None,
                "mark_observed_at": mark["observed_at"] if mark else None,
                "mark_available_at": mark["available_at"] if mark else None,
                "equity": str(equity) if equity is not None else None,
            }
        )
    return {
        "schema": SCHEMA,
        "synthetic_only": True,
        "external_engine_validated": False,
        "decision_view": decision_view,
        "rows": rows,
        "fills": fills,
    }


def compare_accounting_subset(document: dict[str, Any]) -> dict[str, Any]:
    """Compare representable financial rows with the existing local fill ledger.

    This is not full-contract compatibility: the existing ledger reports a final
    cancellation outcome while this contract can retain an active residual. Limits, delayed
    quotes and absent initial marks are unsupported, not translated or invented.
    """
    _validate(document)
    order, buckets, marks = document["order"], document["buckets"], document["marks"]
    reasons = []
    initial_mark = _visible_mark(marks, 0)
    if order["limit"] is not None:
        reasons.append("limit_order")
    if any(x["available_at"] != x["at"] for x in buckets):
        reasons.append("delayed_quote")
    if initial_mark is None:
        reasons.append("initial_mark_unavailable")
    if buckets[0]["at"] == 0:
        reasons.append("bucket_not_after_initial_state")
    report: dict[str, Any] = {
        "schema": "openquant_quote_accounting_comparison_v1",
        "synthetic_only": True,
        "external_engine_validated": False,
        "full_contract_compatible": False,
    }
    if reasons:
        return dict(report, status="unsupported", reasons=reasons, case_comparison_executed=False)
    from experiments import fill_ledger

    expected = replay(document)
    origin = datetime(2026, 1, 1, tzinfo=UTC)

    def time(at: int) -> datetime:
        return origin + timedelta(seconds=at)

    # A time-zero initial mark was required above, never borrowed from the future.
    assert initial_mark is not None
    initial = fill_ledger.InitialState(
        origin, Q(document["cash"]), Q(document["inventory"]), Q(initial_mark["price"])
    )
    decision = fill_ledger.Decision(
        "synthetic",
        time(order["decision_at"]),
        time(order["decision_at"]),
        time(order["earliest_fill_at"]),
        time(order["deadline"]),
        Q(order["shares"] * (1 if order["side"] == "buy" else -1)),
    )
    quotes = []
    for bucket in buckets:
        mark = _visible_mark(marks, bucket["at"])
        assert mark is not None
        quotes.append(
            fill_ledger.MarketBucket(
                time(bucket["at"]),
                Q(mark["price"]),
                Q(bucket["bid"]),
                Q(bucket["ask"]),
                Q(bucket["capacity"]),
            )
        )
    actual = fill_ledger.run_ledger(
        initial, [decision], quotes, fixed_fee=document["fixed_fee"], fee_rate=document["fee_rate"]
    )
    compared = ["cash", "inventory", "equity", "commission"]
    differences = [
        name
        for name in compared
        if [str(getattr(row, name)) for row in actual.rows]
        != [row[name] for row in expected["rows"]]
    ]
    if [row.at for row in actual.rows] != [time(row["at"]) for row in expected["rows"]]:
        differences.append("at")
    # Preserve fractional quantities and subsecond times in comparisons. Casting
    # either to int would hide a divergent observation from a future implementation.
    fills = [(x.at, x.shares, x.price, x.commission) for x in actual.fills]
    expected_fills = [
        (time(x["at"]), Q(x["shares"]), Q(x["price"]), Q(x["commission"]))
        for x in expected["fills"]
    ]
    if fills != expected_fills:
        differences.append("fills")
    return dict(
        report,
        status="different_accounting_subset" if differences else "matched_accounting_subset",
        case_comparison_executed=True,
        compared_fields=["at", *compared, "fills"],
        differing_fields=differences,
        lifecycle={
            "quote_contract": expected["rows"][-1]["status"],
            "fill_ledger": actual.orders[0].status,
        },
        source_sha256={
            "quote_visibility_contract": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "fill_ledger": hashlib.sha256(Path(fill_ledger.__file__).read_bytes()).hexdigest(),
        },
    )


def load_fixture(path: Path) -> dict[str, Any]:
    """Read a bounded local JSON fixture without ambiguous duplicate fields."""
    with path.open("rb") as source:
        raw = source.read(256_001)
    if len(raw) > 256_000:
        raise ValueError("fixture exceeds bounded diagnostic size")

    def unique_fields(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for name, value in pairs:
            if name in result:
                raise ValueError(f"duplicate JSON field: {name}")
            result[name] = value
        return result

    document = json.loads(raw, object_pairs_hook=unique_fields)
    if not isinstance(document, dict):
        raise ValueError("fixture must be a JSON object")
    return document


def main() -> None:
    """Emit a controlled local JSON replay, never a qualification receipt."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, nargs="?", default=DEMO)
    print(json.dumps(replay(load_fixture(parser.parse_args().path)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

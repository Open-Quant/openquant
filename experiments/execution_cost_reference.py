"""Small independent rational interpreter for synthetic single-order cost fixtures.

No imports from the ledger or its record types; no native/library calls.
Not an external engine, and not a realistic market simulator.
"""

from __future__ import annotations

from fractions import Fraction as Q
from typing import Any


def observe(document: dict[str, Any], case: dict[str, Any]) -> dict[str, Any]:
    """Interpret an already schema-validated single-order scenario.

    Inputs only are read; expected paths are never used to calculate outputs.
    """
    s = document["scenario"]
    cash = Q(s["initial"]["cash"])
    inventory = Q(s["initial"]["inventory"])
    order = s["decision"]
    remaining = Q(order["shares"])
    fee, rate = Q(case["fixed_fee"]), Q(case["fee_rate"])
    out: dict[str, Any] = {
        key: []
        for key in (
            "cash",
            "inventory",
            "equity",
            "commission",
            "fill_shares",
            "fill_prices",
            "fill_commissions",
        )
    }
    events = [
        (bucket, volume, False)
        for bucket, volume in zip(s["buckets"], case["volumes"], strict=True)
    ]
    if case["terminal_volume"] is not None:
        events.append((s["terminal_bucket"], case["terminal_volume"], True))
    for quote, volume, terminal in events:
        capacity = Q(s["participation"]) * Q(volume)
        amount = Q(0)
        if terminal:
            amount = -min(inventory, capacity)
        elif order["decision_at"] < quote["at"] and (
            order["earliest_fill_at"] <= quote["at"] <= order["deadline"]
        ):
            amount = min(abs(remaining), capacity) * (1 if remaining > 0 else -1)
        charge = Q(0)
        if amount:
            price = Q(quote["ask"] if amount > 0 else quote["bid"])
            charge = fee + abs(amount) * price * rate
            # State update is deliberately expressed as ledger debits/credits.
            debit = amount * price + charge
            next_cash, next_inventory = cash - debit, inventory + amount
            if next_cash < 0:
                raise ValueError("cash would become negative")
            if next_inventory < 0:
                raise ValueError("inventory would become negative")
            cash, inventory = next_cash, next_inventory
            if not terminal:
                remaining -= amount
            for key, value in (
                ("fill_shares", amount),
                ("fill_prices", price),
                ("fill_commissions", charge),
            ):
                out[key].append(str(value))
        for key, value in (
            ("cash", cash),
            ("inventory", inventory),
            ("equity", cash + inventory * Q(quote["mark"])),
            ("commission", charge),
        ):
            out[key].append(str(value))
    out["terminal_completed"] = case["terminal_volume"] is None or inventory == 0
    out["decisions_completed"] = remaining == 0
    return out

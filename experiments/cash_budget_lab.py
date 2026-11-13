"""Conditional pre-trade cash envelopes; not a live limit-order or sizing API.

The supplied price/fill bounds are assumptions, not forecasts or enforced venue
rules. This module never derives them from future quotes. See CASH_BUDGET_LAB.md.
"""

from __future__ import annotations

import hashlib
import itertools
import platform
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from fractions import Fraction as F
from pathlib import Path
from typing import Any

import numpy as np

from experiments import fill_ledger as ledger


@dataclass(frozen=True)
class BuyBudget:
    cash: F
    desired_shares: F
    price_cap: F
    lot_size: F
    fixed_fee: F
    fee_rate: F
    max_fills: int
    cash_reserve: F
    information_at: datetime
    decision_at: datetime
    shares: F
    worst_case_spend: F
    cash_floor: F


def size_buy(
    *,
    cash: object,
    desired_shares: object,
    price_cap: object,
    lot_size: object,
    fixed_fee: object,
    fee_rate: object,
    max_fills: int,
    information_at: datetime,
    decision_at: datetime,
    cash_reserve: object = 0,
) -> BuyBudget:
    """Maximize order quantity on a lot grid under a declared worst-case envelope.

    The quantity is in [0, desired_shares]. Positive quantities reserve fixed_fee
    for max_fills executions plus proportional fees at price_cap. No order costs
    zero. cash_reserve is unavailable for this order, not a terminal-fee model.
    The bound covers this isolated buy only, without concurrent cash consumers.
    """
    names = (
        "cash",
        "desired_shares",
        "price_cap",
        "lot_size",
        "fixed_fee",
        "fee_rate",
        "cash_reserve",
    )
    values = (cash, desired_shares, price_cap, lot_size, fixed_fee, fee_rate, cash_reserve)
    c, desired, cap, lot, fixed, rate, reserve = (
        ledger._number(value, name, positive=name in {"price_cap", "lot_size"})
        for name, value in zip(names, values)
    )
    if type(max_fills) is not int or max_fills <= 0:
        raise ValueError("max_fills must be a positive integer")
    if reserve > c:
        raise ValueError("cash_reserve must not exceed cash")
    ledger._timestamp(information_at, "information_at")
    ledger._timestamp(decision_at, "decision_at")
    if information_at > decision_at:
        raise ValueError("information_at must not follow decision_at")
    available = max(F(0), c - reserve - max_fills * fixed)
    quantity = min(desired // lot, available // (lot * cap * (1 + rate))) * lot
    spend = quantity * cap * (1 + rate) + max_fills * fixed if quantity else F(0)
    return BuyBudget(
        c,
        desired,
        cap,
        lot,
        fixed,
        rate,
        max_fills,
        reserve,
        information_at,
        decision_at,
        quantity,
        spend,
        c - spend,
    )


def audit_fills(
    budget: BuyBudget, fills: list[ledger.Fill] | tuple[ledger.Fill, ...]
) -> dict[str, Any]:
    """Check genuine ledger fills against one isolated budget, retaining violations.

    This is not a history/authenticity validator. It ignores no extra fills: the
    caller supplies all fills charged to this cash envelope. The audit accepts
    partial/no completion separately from affordability. Order lot rounding does
    not assert that every partial fill is a lot multiple.
    """
    if not isinstance(budget, BuyBudget):
        raise ValueError("budget must be produced by size_buy")
    if not isinstance(fills, (list, tuple)) or any(not isinstance(f, ledger.Fill) for f in fills):
        raise ValueError("fills must be an ordered array of Fill records")
    violations = []
    if len(fills) > budget.max_fills:
        violations.append("max_fills exceeded")
    previous = budget.decision_at
    for f in fills:
        ledger._timestamp(f.at, "fill.at")
        if f.shares <= 0:
            violations.append("fill shares must be positive buys")
        if f.terminal:
            violations.append("terminal fill outside buy envelope")
        if f.price <= 0 or f.price > budget.price_cap:
            violations.append("price_cap violated")
        if f.commission != budget.fixed_fee + f.shares * f.price * budget.fee_rate:
            violations.append("commission differs from declared fees")
        if f.at <= budget.decision_at:
            violations.append("fill must follow decision_at")
        if f.at < previous:
            violations.append("fill chronology is not ordered")
        previous = f.at
    quantity = sum((f.shares for f in fills), F(0))
    spend = sum((f.shares * f.price + f.commission for f in fills), F(0))
    if quantity > budget.shares:
        violations.append("filled shares exceed budget")
    cash_after = budget.cash - spend
    if cash_after < budget.cash_reserve:
        violations.append("cash reserve breached")
    return {
        "within_envelope": not violations,
        "violations": violations,
        "filled_shares": str(quantity),
        "remaining_shares": str(budget.shares - quantity),
        "order_completed": quantity == budget.shares,
        "actual_spend": str(spend),
        "cash_after": str(cash_after),
        "fill_count": len(fills),
    }


def _encode(value: Any) -> Any:
    if isinstance(value, F):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {key: _encode(v) for key, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_encode(v) for v in value]
    return value


def _grid(values: Any, name: str, *, positive: bool = False) -> list[str]:
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError(f"{name} must be a nonempty ordered array")
    return [str(ledger._number(v, name, positive=positive)) for v in values]


def _execute(b: BuyBudget, prices: list[F], volumes: list[F]) -> ledger.LedgerResult:
    state = ledger.InitialState(b.decision_at, b.cash, F(0), b.price_cap)
    decisions = (
        []
        if not b.shares
        else [
            ledger.Decision(
                "buy",
                b.information_at,
                b.decision_at,
                b.decision_at,
                b.decision_at + timedelta(seconds=len(prices)),
                b.shares,
            )
        ]
    )
    quotes = [
        ledger.MarketBucket(
            b.decision_at + timedelta(seconds=i + 1),
            price,
            price,
            price,
            volume,
        )
        for i, (price, volume) in enumerate(zip(prices, volumes, strict=True))
    ]
    return ledger.run_ledger(state, decisions, quotes, fixed_fee=b.fixed_fee, fee_rate=b.fee_rate)


def _negative_controls(at: datetime) -> dict[str, Any]:
    common: dict[str, Any] = dict(
        cash=100,
        desired_shares=20,
        price_cap=10,
        lot_size=1,
        fixed_fee=1,
        fee_rate=0,
        max_fills=2,
        information_at=at,
        decision_at=at,
    )
    controls = {}
    # Independently specified wrong assumptions. Retain the concrete ledger rejection.
    cases = {
        "future_price_exceeds_cap": (size_buy(**common), [F(11), F(11)], [F(4), F(5)]),
        "extra_fixed_fee_fill": (
            size_buy(**{**common, "cash": 95, "fixed_fee": 5, "max_fills": 1}),
            [F(10), F(10)],
            [F(4), F(5)],
        ),
    }
    for name, (b, prices, volumes) in cases.items():
        try:
            result = _execute(b, prices, volumes)
            rejection = None
            detected = not audit_fills(b, result.fills)["within_envelope"]
        except ValueError as exc:
            rejection = str(exc)
            if rejection != "cash would become negative for buy":
                raise
            detected = True
        controls[name] = {
            "detected": detected,
            "ledger_rejection": rejection,
            "budget": _encode(asdict(b)),
            "prices": _encode(prices),
            "volumes": _encode(volumes),
        }
    # A request with no fixed-fee reserve would buy 10; that requires 102 cash.
    controls["ignore_fill_fees"] = {
        "wrong_shares": "10",
        "actual_required_cash": "102",
        "available_cash": "100",
        "detected": F(102) > 100,
    }
    # Rounding up the affordable lot quantity would buy 12 for a 105 budget.
    controls["round_lot_up"] = {
        "wrong_shares": "12",
        "actual_required_cash": "120",
        "available_cash": "105",
        "detected": F(120) > 105,
    }
    return controls


def run_lab(
    *,
    cash_values: Any = (0, 5, 100),
    price_caps: Any = (10, 11),
    lot_sizes: Any = (1, 2),
    fixed_fees: Any = (1, 3),
) -> dict[str, Any]:
    """Retain every prescribed envelope and no/partial/full execution scenario."""
    start = time.perf_counter()
    config = {
        "cash_values": _grid(cash_values, "cash_values"),
        "price_caps": _grid(price_caps, "price_caps", positive=True),
        "lot_sizes": _grid(lot_sizes, "lot_sizes", positive=True),
        "fixed_fees": _grid(fixed_fees, "fixed_fees"),
    }
    at = datetime(2026, 1, 1, tzinfo=UTC)
    scenarios = []
    for cash, cap, lot, fee in itertools.product(*config.values()):
        b = size_buy(
            cash=cash,
            desired_shares=20,
            price_cap=cap,
            lot_size=lot,
            fixed_fee=fee,
            fee_rate="0.01",
            max_fills=2,
            information_at=at,
            decision_at=at,
        )
        for name, volumes in (
            ("no_fills", [F(0), F(0)]),
            ("partial", [b.shares / 2, F(0)]),
            ("complete", [b.shares / 2, b.shares / 2]),
        ):
            result = _execute(b, [b.price_cap * F("0.9"), b.price_cap], volumes)
            scenarios.append(
                {
                    "pattern": name,
                    "budget": _encode(asdict(b)),
                    "ledger": _encode(asdict(result)),
                    "audit": audit_fills(b, result.fills),
                }
            )
    source = Path(__file__).resolve()
    try:
        git_sha = subprocess.check_output(
            ["git", "-C", str(source.parent), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        git_sha = "unavailable"
    return {
        "lab": "cash_budget",
        "schema_version": 1,
        "synthetic_only": True,
        "numeric_encoding": "exact rational strings from decimal inputs",
        "config": config,
        "scenarios": scenarios,
        "negative_controls": _negative_controls(at),
        "limitations": [
            "Conditional affordability, not investment evidence or an optimal policy.",
            "Price cap and maximum fill count must hold; this does not enforce them at a venue.",
            "Single isolated buy envelope; concurrent orders cannot reuse its reserved cash.",
            "No terminal execution, financing, currencies, tick rounding or fee tiers.",
            "Order lot rounding does not impose a lot grid on partial fills.",
            "No completion, liquidity or full-history authenticity guarantee.",
            "Trusted local configurations have no computational resource sandbox.",
        ],
        "provenance": {
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "dependency_sha256": {
                "fill_ledger.py": hashlib.sha256(Path(ledger.__file__).read_bytes()).hexdigest()
            },
            "git_sha": git_sha,
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        "performance": {"wall_seconds": time.perf_counter() - start},
    }

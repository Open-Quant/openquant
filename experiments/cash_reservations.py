"""In-process offline cash reservation prototype, not a broker or durable service.

A lock serializes reserve/fill/finalize transitions in this one Python object.
Only genuine, complete, ordered fills satisfying declared envelopes are modeled.
Finalization asserts that all executions are known: it is NOT a cancel request.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as F
from threading import RLock
from typing import Any

from experiments.cash_budget_lab import BuyBudget, audit_fills, size_buy
from experiments.fill_ledger import Fill, _identifier, _number


@dataclass(frozen=True)
class CashSnapshot:
    cash: F
    reserved: F
    cash_floor: F
    available: F
    revision: int


@dataclass(frozen=True)
class _Order:
    budget: BuyBudget
    fills: tuple[Fill, ...] = ()
    finalized: bool = False

    @property
    def remaining(self) -> F:
        quantity = self.budget.shares - sum((f.shares for f in self.fills), F(0))
        if self.finalized or quantity == 0:
            return F(0)
        b = self.budget
        return (
            quantity * b.price_cap * (1 + b.fee_rate)
            + (b.max_fills - len(self.fills)) * b.fixed_fee
        )


class CashReservations:
    """One currency, buy-only toy account; every cash consumer must use this object.

    No settlement, selling, margin, FX, persistence, external synchronization,
    market impact or source authentication. Snapshot availability is advisory;
    reserve recomputes the budget inside the lock. Never share snapshots as funds.
    """

    def __init__(self, cash: object, *, cash_floor: object = 0):
        self._cash = _number(cash, "cash")
        self._floor = _number(cash_floor, "cash_floor")
        if self._floor > self._cash:
            raise ValueError("cash_floor must not exceed cash")
        self._orders: dict[str, _Order] = {}
        self._events: dict[str, Fill] = {}
        self._revision = 0
        self._lock = RLock()

    def snapshot(self) -> CashSnapshot:
        with self._lock:
            reserved = sum((order.remaining for order in self._orders.values()), F(0))
            return CashSnapshot(
                self._cash,
                reserved,
                self._floor,
                self._cash - self._floor - reserved,
                self._revision,
            )

    def reserve(self, order_id: str, **bounds: Any) -> BuyBudget:
        """Size and reserve from unencumbered cash, in one locked transition.

        Bounds are size_buy arguments except cash/cash_reserve, owned here.
        Unique order IDs are retained even for a no-order decision.
        """
        _identifier(order_id, "order_id")
        if "cash" in bounds or "cash_reserve" in bounds:
            raise ValueError("cash and cash_reserve are owned by the account")
        with self._lock:
            if order_id in self._orders:
                raise ValueError("order_id already exists")
            budget = size_buy(cash=self.snapshot().available, **bounds)
            self._orders[order_id] = _Order(budget)
            self._revision += 1
            return budget

    def fill(self, event_id: str, fill: Fill) -> CashSnapshot:
        """Apply one exact execution once, rejecting conflicting redelivery.

        Rejections leave the account unchanged. A breach in a real execution must
        instead trigger reconciliation; refusing a record cannot undo a trade.
        """
        _identifier(event_id, "event_id")
        if not isinstance(fill, Fill):
            raise ValueError("fill must be a Fill")
        # Fill records from the ledger are exact; do not accept forged float/NaN
        # fields that would contaminate this prototype's arithmetic.
        if any(type(v) is not F for v in (fill.shares, fill.price, fill.commission)):
            raise ValueError("fill numeric fields must be exact Fractions")
        with self._lock:
            if event_id in self._events:
                if self._events[event_id] != fill:
                    raise ValueError("conflicting duplicate event_id")
                return self.snapshot()
            if fill.decision_id not in self._orders:
                raise ValueError("unknown order_id")
            order = self._orders[fill.decision_id]
            if order.finalized:
                raise ValueError("fill after finalized order")
            fills = (*order.fills, fill)
            audit = audit_fills(order.budget, fills)
            if not audit["within_envelope"]:
                raise ValueError("; ".join(audit["violations"]))
            updated = _Order(order.budget, fills)
            spend = fill.shares * fill.price + fill.commission
            after_cash = self._cash - spend
            after_reserved = self.snapshot().reserved - order.remaining + updated.remaining
            if after_cash - after_reserved < self._floor:
                raise ValueError("account cash floor breached")
            self._cash = after_cash
            self._orders[fill.decision_id] = updated
            self._events[event_id] = fill
            self._revision += 1
            return self.snapshot()

    def finalize(self, order_id: str) -> CashSnapshot:
        """Assert a complete execution history and release unfilled reservation.

        Synthetic terminal event only. In live systems a cancel acknowledgment
        alone may be insufficient: reconcile executions, sequence gaps and races.
        No automatic deadline, cancellation request, or exhausted-fill release.
        """
        with self._lock:
            if order_id not in self._orders:
                raise ValueError("unknown order_id")
            order = self._orders[order_id]
            if not order.finalized:
                self._orders[order_id] = _Order(order.budget, order.fills, True)
                self._revision += 1
            return self.snapshot()

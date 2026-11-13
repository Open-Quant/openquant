"""Exact, offline, long-only accounting for one instrument and one currency.

Finite float inputs mean their decimal repr, not their binary64 expansion. Values
are normalized to Fraction in frozen input records. This is neither an exchange
simulator nor a live execution API. Read FILL_LEDGER_LAB.md before interpretation.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta
from fractions import Fraction


def _number(value: object, name: str, *, positive: bool = False) -> Fraction:
    if isinstance(value, bool) or not isinstance(value, (int, float, str, Fraction)):
        raise ValueError(f"{name} must be a finite numeric value")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    try:
        result = Fraction(str(value)) if isinstance(value, float) else Fraction(value)
    except (ValueError, TypeError, ZeroDivisionError, OverflowError) as exc:
        raise ValueError(f"{name} must be finite and numeric") from exc
    if result < 0 or (positive and result == 0):
        raise ValueError(f"{name} must be {'positive' if positive else 'nonnegative'}")
    return result


def _timestamp(value: object, name: str) -> None:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise ValueError(f"{name} must be an aware UTC datetime")


def _identifier(value: object, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")


def _units(record: object) -> None:
    for name in ("instrument", "currency"):
        _identifier(getattr(record, name), name)


@dataclass(frozen=True)
class InitialState:
    at: datetime
    cash: Fraction
    inventory: Fraction
    mark: Fraction
    instrument: str = "SYNTH"
    currency: str = "USD"

    def __post_init__(self) -> None:
        _timestamp(self.at, "initial.at")
        _units(self)
        for name in ("cash", "inventory", "mark"):
            object.__setattr__(
                self, name, _number(getattr(self, name), name, positive=name == "mark")
            )


@dataclass(frozen=True)
class Decision:
    id: str
    information_at: datetime
    decision_at: datetime
    earliest_fill_at: datetime
    deadline: datetime
    shares: Fraction
    instrument: str = "SYNTH"
    currency: str = "USD"

    def __post_init__(self) -> None:
        _identifier(self.id, "decision.id")
        _units(self)
        times = (self.information_at, self.decision_at, self.earliest_fill_at, self.deadline)
        for name, timestamp in zip(
            ("information_at", "decision_at", "earliest_fill_at", "deadline"), times
        ):
            _timestamp(timestamp, name)
        if list(times) != sorted(times):
            raise ValueError(f"decision {self.id}: timestamp order is invalid")
        # Validate signed shares without allowing bool, NaN or unknown values.
        value = self.shares
        if isinstance(value, bool) or not isinstance(value, (int, float, str, Fraction)):
            raise ValueError("shares must be finite and nonzero")
        try:
            shares = Fraction(str(value)) if isinstance(value, float) else Fraction(value)
        except (ValueError, TypeError, ZeroDivisionError, OverflowError) as exc:
            raise ValueError("shares must be finite and nonzero") from exc
        if not shares:
            raise ValueError("shares must be nonzero")
        object.__setattr__(self, "shares", shares)


@dataclass(frozen=True)
class MarketBucket:
    at: datetime
    mark: Fraction
    bid: Fraction
    ask: Fraction
    volume: Fraction
    source: str = "synthetic"
    instrument: str = "SYNTH"
    currency: str = "USD"

    def __post_init__(self) -> None:
        _timestamp(self.at, "bucket.at")
        _identifier(self.source, "source")
        _units(self)
        for name in ("mark", "bid", "ask", "volume"):
            object.__setattr__(
                self, name, _number(getattr(self, name), name, positive=name != "volume")
            )
        if self.bid > self.ask:
            raise ValueError("bid must not exceed ask")


@dataclass(frozen=True)
class Fill:
    decision_id: str | None
    at: datetime
    shares: Fraction
    price: Fraction
    commission: Fraction
    terminal: bool


@dataclass(frozen=True)
class Attempt:
    decision_id: str
    at: datetime
    shares: Fraction
    remaining: Fraction
    reason: str


@dataclass(frozen=True)
class LedgerRow:
    at: datetime
    phase: str
    mark: Fraction
    cash: Fraction
    inventory: Fraction
    equity: Fraction
    market_pnl: Fraction
    execution_pnl: Fraction
    commission: Fraction
    equity_change: Fraction
    capacity: Fraction
    executed_volume: Fraction


@dataclass(frozen=True)
class OrderOutcome:
    decision_id: str
    requested: Fraction
    filled: Fraction
    remaining: Fraction
    status: str


@dataclass(frozen=True)
class TerminalOutcome:
    policy: str
    cash: Fraction
    inventory: Fraction
    marked_equity: Fraction
    completed: bool


@dataclass(frozen=True)
class LedgerResult:
    initial: InitialState
    decisions: tuple[Decision, ...]
    buckets: tuple[MarketBucket, ...]
    terminal_bucket: MarketBucket | None
    participation: Fraction
    fixed_fee: Fraction
    fee_rate: Fraction
    fills: tuple[Fill, ...]
    attempts: tuple[Attempt, ...]
    rows: tuple[LedgerRow, ...]
    orders: tuple[OrderOutcome, ...]
    terminal: TerminalOutcome
    decisions_completed: bool


def run_ledger(
    initial: InitialState,
    decisions: Iterable[Decision],
    buckets: Iterable[MarketBucket],
    *,
    participation: object = 1,
    fixed_fee: object = 0,
    fee_rate: object = 0,
    terminal_policy: str = "mark_only",
    terminal_bucket: MarketBucket | None = None,
) -> LedgerResult:
    """Execute FIFO with shared gross volume; reject cash borrowing and shorts.

    Eligible quotes are strictly after decision_at, at/after earliest_fill_at,
    and at/before deadline. Unknown event ordering at equal timestamps cannot
    produce a fill. A terminal bucket is separate and strictly after the horizon.
    Unfilled orders are cancelled at deadline or regular horizon, never carried
    into terminal liquidation. Zero capacity creates attempts, never zero fills.
    """
    p = _number(participation, "participation")
    fee = _number(fixed_fee, "fixed_fee")
    rate = _number(fee_rate, "fee_rate")
    if p > 1:
        raise ValueError("participation must be at most one")
    if not isinstance(initial, InitialState):
        raise ValueError("initial must be an InitialState")
    orders = tuple(decisions)
    quotes = tuple(buckets)
    if any(not isinstance(d, Decision) for d in orders):
        raise ValueError("decisions must contain Decision records")
    if any(not isinstance(q, MarketBucket) for q in quotes):
        raise ValueError("buckets must contain MarketBucket records")
    orders = tuple(sorted(orders, key=lambda d: (d.decision_at, d.id)))
    if len({d.id for d in orders}) != len(orders):
        raise ValueError("duplicate decision IDs")
    if any(d.decision_at < initial.at for d in orders):
        raise ValueError("decision timestamp precedes initial state")
    if terminal_policy not in ("mark_only", "liquidate_at_declared_bid"):
        raise ValueError("unknown terminal_policy")
    if terminal_policy == "mark_only" and terminal_bucket is not None:
        raise ValueError("mark_only must not supply terminal_bucket")
    if terminal_policy == "liquidate_at_declared_bid" and not isinstance(
        terminal_bucket, MarketBucket
    ):
        raise ValueError("liquidation requires terminal_bucket")
    previous = initial.at
    for quote in quotes:
        if quote.at <= previous:
            raise ValueError("bucket timestamps must be strictly increasing after initial.at")
        previous = quote.at
    if terminal_bucket is not None and terminal_bucket.at <= previous:
        raise ValueError("terminal_bucket must be strictly after the regular horizon")
    records = (*orders, *quotes, *((terminal_bucket,) if terminal_bucket is not None else ()))
    for record in records:
        for unit in ("instrument", "currency"):
            if getattr(record, unit) != getattr(initial, unit):
                raise ValueError(f"inconsistent {unit}")

    cash, inventory, mark = initial.cash, initial.inventory, initial.mark
    remaining = {d.id: d.shares for d in orders}
    fills: list[Fill] = []
    attempts: list[Attempt] = []
    rows: list[LedgerRow] = []
    zero = Fraction(0)

    def transition(quote: MarketBucket, terminal: bool) -> None:
        nonlocal cash, inventory, mark
        old_equity = cash + inventory * mark
        market_pnl = inventory * (quote.mark - mark)
        capacity = p * quote.volume
        left, execution_pnl, commissions = capacity, zero, zero

        def execute(qty: Fraction, decision_id: str | None) -> None:
            nonlocal cash, inventory, left, execution_pnl, commissions
            if not qty:
                return
            price = quote.ask if qty > 0 else quote.bid
            charge = fee + abs(qty) * price * rate
            next_cash, next_inventory = cash - qty * price - charge, inventory + qty
            if next_cash < 0:
                raise ValueError(f"cash would become negative for {decision_id or 'terminal sale'}")
            if next_inventory < 0:
                raise ValueError(f"inventory would become negative for {decision_id}")
            cash, inventory = next_cash, next_inventory
            left -= abs(qty)
            execution_pnl += qty * (quote.mark - price)
            commissions += charge
            fills.append(Fill(decision_id, quote.at, qty, price, charge, terminal))

        if terminal:
            execute(-min(inventory, left), None)
        else:
            for d in orders:
                rem = remaining[d.id]
                if not rem or not (
                    d.decision_at < quote.at and d.earliest_fill_at <= quote.at <= d.deadline
                ):
                    continue
                qty = min(abs(rem), left) * (1 if rem > 0 else -1)
                execute(qty, d.id)
                remaining[d.id] -= qty
                reason = (
                    "filled" if not remaining[d.id] else "partial_volume" if qty else "no_volume"
                )
                attempts.append(Attempt(d.id, quote.at, qty, remaining[d.id], reason))
        equity = cash + inventory * quote.mark
        change = equity - old_equity
        if change != market_pnl + execution_pnl - commissions:
            raise ArithmeticError("exact equity reconciliation failed")
        rows.append(
            LedgerRow(
                quote.at,
                "terminal" if terminal else "regular",
                quote.mark,
                cash,
                inventory,
                equity,
                market_pnl,
                execution_pnl,
                commissions,
                change,
                capacity,
                capacity - left,
            )
        )
        mark = quote.mark

    for quote in quotes:
        transition(quote, False)
    horizon = quotes[-1].at if quotes else initial.at
    outcomes = tuple(
        OrderOutcome(
            d.id,
            d.shares,
            d.shares - remaining[d.id],
            remaining[d.id],
            "filled"
            if not remaining[d.id]
            else "cancelled_at_deadline"
            if d.deadline <= horizon
            else "cancelled_at_horizon",
        )
        for d in orders
    )
    if terminal_bucket is not None:
        transition(terminal_bucket, True)
    terminal = TerminalOutcome(
        terminal_policy,
        cash,
        inventory,
        cash + inventory * mark,
        inventory == 0 if terminal_bucket is not None else True,
    )
    return LedgerResult(
        initial,
        orders,
        quotes,
        terminal_bucket,
        p,
        fee,
        rate,
        tuple(fills),
        tuple(attempts),
        tuple(rows),
        outcomes,
        terminal,
        all(not o.remaining for o in outcomes),
    )

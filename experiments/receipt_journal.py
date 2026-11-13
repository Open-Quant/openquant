"""Offline single-account receipt-journal experiment, not a broker adapter.

SQLite commits commands before returning. Every operation rebuilds state inside
a write transaction, deliberately trading throughput for a small audit surface.
One trusted source assigns gap-free sequence numbers starting at one. Its terminal
event asserts completeness for that order. Neither assertion is authenticated here.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from fractions import Fraction as F
from pathlib import Path
from typing import Any, TypedDict

from experiments.cash_budget_lab import _encode
from experiments.cash_reservations import CashReservations
from experiments.fill_ledger import Fill, _identifier, _number


class _InitialConfiguration(TypedDict):
    version: int
    cash: str
    cash_floor: str


@dataclass(frozen=True)
class JournalSnapshot:
    cash: F
    reserved: F
    available: F
    watermark: int
    pending: int
    quarantined: bool
    reason: str | None


class _State:
    def __init__(self, cash: str, floor: str):
        self.book = CashReservations(cash, cash_floor=floor)
        self.events: dict[str, dict[str, Any]] = {}
        self.sequences: dict[int, dict[str, Any]] = {}
        self.reservations: dict[str, dict[str, Any]] = {}
        self.watermark = 0
        self.reason: str | None = None

    def snapshot(self) -> JournalSnapshot:
        s = self.book.snapshot()
        pending = sum(seq > self.watermark for seq in self.sequences)
        return JournalSnapshot(
            s.cash,
            s.reserved,
            F(0) if pending or self.reason else s.available,
            self.watermark,
            pending,
            self.reason is not None,
            self.reason,
        )

    def apply(self, command: dict[str, Any]) -> None:
        if command["op"] == "reserve":
            identity = command["id"]
            if identity in self.reservations:
                if command != self.reservations[identity]:
                    raise ValueError("conflicting reservation command")
                return
            state = self.snapshot()
            if state.quarantined:
                raise ValueError("account quarantined; reconciliation required")
            if state.pending:
                raise ValueError("pending sequence gap; reservation blocked")
            b = dict(command["bounds"])
            for name in ("information_at", "decision_at"):
                b[name] = datetime.fromisoformat(b[name])
            self.book.reserve(command["order_id"], **b)
            self.reservations[identity] = command
            return
        if command["op"] != "receive":
            raise ValueError("unknown journal operation")
        event = command["event"]
        old_id = self.events.get(event["id"])
        old_seq = self.sequences.get(event["sequence"])
        if old_id == event and old_seq == event:
            return
        if old_id is not None or old_seq is not None:
            self.reason = self.reason or "conflicting event ID or source sequence"
            return
        self.events[event["id"]] = event
        self.sequences[event["sequence"]] = event
        # Keep evidence after quarantine, but never publish more buying power.
        while not self.reason and self.watermark + 1 in self.sequences:
            next_event = self.sequences[self.watermark + 1]
            try:
                if next_event["kind"] == "terminal":
                    self.book.finalize(next_event["order_id"])
                else:
                    payload = next_event["fill"]
                    fill = Fill(
                        next_event["order_id"],
                        datetime.fromisoformat(payload["at"]),
                        F(payload["shares"]),
                        F(payload["price"]),
                        F(payload["commission"]),
                        payload["terminal"],
                    )
                    self.book.fill(next_event["id"], fill)
            except (ValueError, TypeError, KeyError) as exc:
                self.reason = f"source event requires reconciliation: {exc}"
            else:
                self.watermark += 1


def _json(value: Any) -> str:
    return json.dumps(_encode(value), sort_keys=True, separators=(",", ":"), allow_nan=False)


class ReceiptJournal:
    """File-backed synthetic ledger; all cash users must share this local database.

    Source gaps withhold all available funds. Breaches/conflicts are persisted,
    freeze new reservations, and require out-of-band reconciliation (not supplied).
    Cash in quarantine is the last valid prefix, NOT a claimed actual balance.
    No credentials, orders, network calls, selling, settlement or market model.
    """

    def __init__(self, path: str | Path, *, cash: object, cash_floor: object = 0):
        initial: _InitialConfiguration = {
            "version": 1,
            "cash": str(_number(cash, "cash")),
            "cash_floor": str(_number(cash_floor, "cash_floor")),
        }
        CashReservations(initial["cash"], cash_floor=initial["cash_floor"])
        self._initial = initial
        self._db = sqlite3.connect(path, isolation_level=None, timeout=10)
        try:
            self._db.execute("PRAGMA journal_mode=DELETE")
            self._db.execute("PRAGMA synchronous=FULL")
            self._db.execute("BEGIN IMMEDIATE")
            tables = {
                row[0]
                for row in self._db.execute("SELECT name FROM sqlite_master WHERE type='table'")
            }
            existing = bool(tables)
            if existing and not {"metadata", "commands"} <= tables:
                raise ValueError("incomplete journal schema")
            self._db.execute(
                "CREATE TABLE IF NOT EXISTS metadata (id INTEGER PRIMARY KEY "
                "CHECK(id=1), value TEXT NOT NULL)"
            )
            self._db.execute(
                "CREATE TABLE IF NOT EXISTS commands (seq INTEGER PRIMARY KEY, "
                "payload TEXT NOT NULL)"
            )
            row = self._db.execute("SELECT value FROM metadata WHERE id=1").fetchone()
            if row is None:
                if existing:
                    raise ValueError("incomplete journal metadata")
                self._db.execute("INSERT INTO metadata VALUES (1, ?)", (_json(initial),))
            elif row[0] != _json(initial):
                raise ValueError("initial account configuration differs from journal")
            self._replay()
            self._db.commit()
        except BaseException:
            self._db.rollback()
            self._db.close()
            raise

    def _replay(self) -> _State:
        state = _State(self._initial["cash"], self._initial["cash_floor"])
        for expected, (seq, payload) in enumerate(
            self._db.execute("SELECT seq,payload FROM commands ORDER BY seq"), start=1
        ):
            if seq != expected:
                raise ValueError("journal sequence corrupted")
            state.apply(json.loads(payload))
        return state

    def _append(self, command: dict[str, Any]) -> JournalSnapshot:
        payload = _json(command)
        # Replay encoded values as persisted, not mutable caller objects.
        self._db.execute("BEGIN IMMEDIATE")
        try:
            state = self._replay()
            state.apply(json.loads(payload))
            duplicate = self._db.execute(
                "SELECT 1 FROM commands WHERE payload=? LIMIT 1", (payload,)
            ).fetchone()
            if duplicate is None:
                self._db.execute("INSERT INTO commands(payload) VALUES (?)", (payload,))
            self._db.commit()
            return state.snapshot()
        except BaseException:
            self._db.rollback()
            raise

    def reserve(self, command_id: str, order_id: str, **bounds: Any) -> JournalSnapshot:
        """Durably size from current free cash; exact command retries are harmless."""
        _identifier(command_id, "command_id")
        _identifier(order_id, "order_id")
        return self._append(
            {"op": "reserve", "id": command_id, "order_id": order_id, "bounds": bounds}
        )

    def receive(
        self, event_id: str, *, sequence: int, kind: str, order_id: str, fill: Fill | None = None
    ) -> JournalSnapshot:
        """Retain a source receipt before acknowledging; process contiguous prefix.

        A terminal receipt is a trusted completeness assertion, not cancel-request
        acknowledgment. Syntax errors are rejected; semantic conflicts retained.
        """
        _identifier(event_id, "event_id")
        _identifier(order_id, "order_id")
        if type(sequence) is not int or sequence < 1:
            raise ValueError("sequence must be a positive integer")
        if kind not in {"fill", "terminal"}:
            raise ValueError("kind must be fill or terminal")
        if kind == "terminal" and fill is not None:
            raise ValueError("terminal must not contain fill")
        event = {"id": event_id, "sequence": sequence, "kind": kind, "order_id": order_id}
        if kind == "fill":
            if not isinstance(fill, Fill) or fill.decision_id != order_id:
                raise ValueError("fill must match order_id")
            if any(type(v) is not F for v in (fill.shares, fill.price, fill.commission)):
                raise ValueError("fill amounts must be exact Fractions")
            if not isinstance(fill.at, datetime) or type(fill.terminal) is not bool:
                raise ValueError("fill timestamp/terminal fields malformed")
            event["fill"] = {
                "at": fill.at,
                "shares": fill.shares,
                "price": fill.price,
                "commission": fill.commission,
                "terminal": fill.terminal,
            }
        return self._append({"op": "receive", "event": event})

    def snapshot(self) -> JournalSnapshot:
        """Last valid prefix; quarantine/gaps force available funds to zero."""
        return self._replay().snapshot()

    def command_count(self) -> int:
        return self._db.execute("SELECT count(*) FROM commands").fetchone()[0]

    def close(self) -> None:
        self._db.close()

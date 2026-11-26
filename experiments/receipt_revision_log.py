"""Content-bound synthetic corrections and busts; never a venue adapter."""

import hashlib
import re
import struct
from collections.abc import Iterable
from dataclasses import dataclass
from fractions import Fraction as Q
from itertools import islice

from experiments.receipt_completion_witness import MAX_DELIVERIES, MAX_RECEIPTS
from experiments.receipt_content_commitment import (
    RequestScope,
    _execution_fields,
    _integer,
    _text,
)
from experiments.receipt_order_conformance import Execution, Key

SCHEMA = "openquant_synthetic_receipt_revision_log_v2"
_DOMAIN = b"openquant:synthetic-receipt-revisions\x00v2\x00"
Balance = tuple[Q, Q, Q]
_ZERO: Balance = (Q(0), Q(0), Q(0))


def _identity(value: str) -> bytes:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("event identity must be nonempty")
    return _text(value)


def _event_header(sequence: int, identity: str) -> None:
    if type(sequence) is not int or not 1 <= sequence <= MAX_RECEIPTS:
        raise ValueError("event sequence must be a bounded positive integer")
    _identity(identity)


@dataclass(frozen=True)
class Fill:
    sequence: int
    identity: str
    execution: Execution

    def __post_init__(self) -> None:
        _event_header(self.sequence, self.identity)
        _execution_fields(self.execution)


@dataclass(frozen=True)
class Correct:
    sequence: int
    identity: str
    target: str
    execution: Execution

    def __post_init__(self) -> None:
        _event_header(self.sequence, self.identity)
        _identity(self.target)
        _execution_fields(self.execution)


@dataclass(frozen=True)
class Bust:
    sequence: int
    identity: str
    target: str

    def __post_init__(self) -> None:
        _event_header(self.sequence, self.identity)
        _identity(self.target)


Event = Fill | Correct | Bust


@dataclass(frozen=True)
class RevisionPin:
    """Externally frozen knowledge-prefix identity; not a signed receipt."""

    scope: RequestScope
    sha256: str
    schema: str = SCHEMA

    def __post_init__(self) -> None:
        if not isinstance(self.scope, RequestScope):
            raise ValueError("revision request scope required")
        if self.schema != SCHEMA:
            raise ValueError("unsupported revision schema")
        if not isinstance(self.sha256, str) or re.fullmatch("[0-9a-f]{64}", self.sha256) is None:
            raise ValueError("revision SHA-256 must have 64 lowercase hexadecimal digits")


def _checked_rows(scope: RequestScope, delivered: Iterable[Event]) -> tuple[tuple[Event, ...], int]:
    if not isinstance(scope, RequestScope):
        raise ValueError("revision request scope required")
    by_sequence: dict[int, Event] = {}
    ids: set[str] = set()
    retries = 0
    for i, event in enumerate(islice(delivered, MAX_DELIVERIES + 1)):
        if i == MAX_DELIVERIES:
            raise ValueError("delivery limit exceeded")
        if not isinstance(event, (Fill, Correct, Bust)):
            raise ValueError("typed revision event required")
        if by_sequence.get(event.sequence) == event:
            retries += 1
            continue
        if event.sequence in by_sequence or event.identity in ids:
            raise ValueError("conflicting revision identity or source sequence")
        by_sequence[event.sequence] = event
        ids.add(event.identity)
    count = scope.manifest.expected_count
    if sorted(by_sequence) != list(range(1, count + 1)):
        raise ValueError("revision log requires the full contiguous source sequence")
    return tuple(by_sequence[i] for i in range(1, count + 1)), retries


def _digest(scope: RequestScope, rows: tuple[Event, ...]) -> str:
    sha = hashlib.sha256(_DOMAIN)

    def frame(*fields: bytes) -> None:
        for field in fields:
            sha.update(struct.pack(">I", len(field)))
            sha.update(field)

    frame(
        _text(scope.manifest.request_id),
        _text(scope.generation),
        _integer(scope.manifest.expected_count),
    )
    for event in rows:
        seq, identity = _integer(event.sequence), _identity(event.identity)
        if isinstance(event, Fill):
            frame(b"fill", seq, identity, *_execution_fields(event.execution))
        elif isinstance(event, Correct):
            frame(
                b"correct",
                seq,
                identity,
                _identity(event.target),
                *_execution_fields(event.execution),
            )
        else:
            frame(b"bust", seq, identity, _identity(event.target))
    return sha.hexdigest()


def _execution_delta(row: Execution) -> Balance:
    return row.shares, -row.shares * row.price - row.fee, row.fee


def _plus(left: Balance, right: Balance) -> Balance:
    return left[0] + right[0], left[1] + right[1], left[2] + right[2]


def _minus(left: Balance, right: Balance) -> Balance:
    return left[0] - right[0], left[1] - right[1], left[2] - right[2]


def _revisions(rows: tuple[Event, ...]) -> Iterable[tuple[Execution | None, Execution | None]]:
    active: dict[str, Execution] = {}
    seen_keys: set[Key] = set()
    for event in rows:
        if isinstance(event, Fill):
            if event.execution.key in seen_keys:
                raise ValueError("reused original execution key")
            seen_keys.add(event.execution.key)
            active[event.identity] = event.execution
            yield None, event.execution
            continue
        if (old := active.pop(event.target, None)) is None:
            raise ValueError("revision target is unknown, stale or no longer active")
        if isinstance(event, Bust):
            yield old, None
            continue
        new = event.execution
        if new.key != old.key or new.at != old.at or (new.shares > 0) != (old.shares > 0):
            raise ValueError("correction must keep original execution key, effective time and side")
        active[event.identity] = new
        yield old, new


@dataclass(frozen=True)
class Tape:
    """Replay of one pinned synthetic prefix; all monetary values are exact."""

    pin: RevisionPin
    events: tuple[Event, ...]
    redeliveries: int
    _prefix: tuple[Balance, ...]
    _deltas: tuple[tuple[int, Balance], ...]

    @property
    def terminal_authority(self) -> bool:
        return False

    @property
    def external_engine_validated(self) -> bool:
        return False

    def extends(self, earlier: "Tape") -> bool:
        """Check append-only continuity, including empty or equal-length prefixes."""
        scope, prior = self.pin.scope, earlier.pin.scope
        return (
            scope.manifest.request_id == prior.manifest.request_id
            and scope.generation == prior.generation
            and self.events[: len(earlier.events)] == earlier.events
        )

    def as_of(self, known_through: int, *, effective_through: int | None = None) -> Balance:
        """Project fills effective by time using only revisions known by sequence."""
        if type(known_through) is not int or not 0 <= known_through <= len(self.events):
            raise ValueError("knowledge cutoff outside verified prefix")
        if effective_through is None:
            return self._prefix[known_through]
        if type(effective_through) is not int or effective_through < 0:
            raise ValueError("effective cutoff must be a nonnegative integer")
        result = _ZERO
        for at, delta in islice(self._deltas, known_through):
            if at <= effective_through:
                result = _plus(result, delta)
        return result

    def delta_at(self, sequence: int) -> Balance:
        if type(sequence) is not int or not 1 <= sequence <= len(self.events):
            raise ValueError("delta sequence outside verified prefix")
        return self._deltas[sequence - 1][1]


def freeze(scope: RequestScope, events: Iterable[Event]) -> RevisionPin:
    """Produce an independent pin before passing the tape to a consumer."""
    rows, _ = _checked_rows(scope, events)
    for _ in _revisions(rows):
        pass
    return RevisionPin(scope, _digest(scope, rows))


def replay(scope: RequestScope, events: Iterable[Event], expected: RevisionPin) -> Tape | None:
    """Replay only a trusted exact pin; None means no agreement, not no trades."""
    if not isinstance(scope, RequestScope) or not isinstance(expected, RevisionPin):
        raise ValueError("revision request scope and independent pin required")
    if scope != expected.scope:
        return None
    rows, retries = _checked_rows(scope, events)
    if _digest(scope, rows) != expected.sha256:
        return None
    deltas: list[tuple[int, Balance]] = []
    prefix = [_ZERO]
    for old, new in _revisions(rows):
        effective = new if new is not None else old
        assert effective is not None
        delta = _minus(
            _execution_delta(new) if new is not None else _ZERO,
            _execution_delta(old) if old is not None else _ZERO,
        )
        deltas.append((effective.at, delta))
        prefix.append(_plus(prefix[-1], delta))
    return Tape(expected, rows, retries, tuple(prefix), tuple(deltas))

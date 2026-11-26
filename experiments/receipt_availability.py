"""Synthetic observer-time checkpoints for separately pinned revision tapes."""

import hashlib
import re
import struct
from collections.abc import Iterable
from dataclasses import dataclass
from itertools import islice

from experiments.receipt_completion_witness import MAX_DELIVERIES
from experiments.receipt_content_commitment import _integer
from experiments.receipt_revision_log import Balance, RevisionPin, Tape, _identity

_DOMAIN = b"openquant:synthetic-receipt-availability\x00v1\x00"


def _tick(value: int) -> bytes:
    if type(value) is not int or value < 0:
        raise ValueError("observation tick must be a nonnegative integer")
    return _integer(value)


@dataclass(frozen=True)
class Observation:
    """A fixture's first observation of an event, on one named clock."""

    event_id: str
    first_seen: int

    def __post_init__(self) -> None:
        _identity(self.event_id)
        _tick(self.first_seen)


@dataclass(frozen=True)
class ObserverPin:
    revision_pin: RevisionPin
    observer: str
    start: int
    through: int
    sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.revision_pin, RevisionPin):
            raise ValueError("revision pin required")
        _identity(self.observer)
        _tick(self.start)
        _tick(self.through)
        if self.start > self.through:
            raise ValueError("observation coverage interval is reversed")
        if not isinstance(self.sha256, str) or re.fullmatch("[0-9a-f]{64}", self.sha256) is None:
            raise ValueError("observer SHA-256 must have 64 lowercase hexadecimal digits")


def _observations(
    tape: Tape, start: int, through: int, delivered: Iterable[Observation]
) -> tuple[Observation, ...]:
    _tick(start)
    _tick(through)
    if start > through:
        raise ValueError("observation coverage interval is reversed")
    by_id: dict[str, Observation] = {}
    for i, row in enumerate(islice(delivered, MAX_DELIVERIES + 1)):
        if i == MAX_DELIVERIES:
            raise ValueError("observation delivery limit exceeded")
        if not isinstance(row, Observation) or not start <= row.first_seen <= through:
            raise ValueError("typed observation inside coverage required")
        prior = by_id.setdefault(row.event_id, row)
        if row != prior:
            raise ValueError("conflicting first observation")
    if by_id.keys() != {event.identity for event in tape.events}:
        raise ValueError("exactly one first observation per source event required")
    return tuple(by_id[event.identity] for event in tape.events)


def _digest(
    tape: Tape, observer: str, start: int, through: int, rows: tuple[Observation, ...]
) -> str:
    sha = hashlib.sha256(_DOMAIN)
    fields = (
        _identity(tape.pin.schema),
        bytes.fromhex(tape.pin.sha256),
        _identity(observer),
        _tick(start),
        _tick(through),
    )
    for field in (
        *fields,
        *(v for row in rows for v in (_identity(row.event_id), _tick(row.first_seen))),
    ):
        sha.update(struct.pack(">I", len(field)))
        sha.update(field)
    return sha.hexdigest()


@dataclass(frozen=True)
class Checkpoint:
    """Qualified synthetic observation interval; not timestamp attestation."""

    pin: ObserverPin
    tape: Tape
    observations: tuple[Observation, ...]

    @property
    def terminal_authority(self) -> bool:
        return False

    def extends(self, earlier: "Checkpoint") -> bool:
        """Preserve the earlier cut even if both snapshots have valid standalone pins."""
        n = len(earlier.observations)
        return (
            self.pin.observer == earlier.pin.observer
            and self.pin.start == earlier.pin.start
            and self.pin.through >= earlier.pin.through
            and self.tape.extends(earlier.tape)
            and self.observations[:n] == earlier.observations
            and all(row.first_seen > earlier.pin.through for row in self.observations[n:])
        )

    def at(self, tick: int, *, effective_through: int | None = None) -> Balance:
        """Project only when observations available by tick form a complete source prefix."""
        _tick(tick)
        if not self.pin.start <= tick <= self.pin.through:
            raise ValueError("observation cutoff outside verified coverage")
        known = 0
        for index, row in enumerate(self.observations):
            if row.first_seen <= tick:
                if known != index:
                    raise ValueError("observed source sequence has a gap at the cutoff")
                known += 1
        return self.tape.as_of(known, effective_through=effective_through)


def freeze_observer(
    tape: Tape, observer: str, start: int, through: int, observations: Iterable[Observation]
) -> ObserverPin:
    """Producer-side fixture: preserve the pin separately from consumer delivery."""
    rows = _observations(tape, start, through, observations)
    return ObserverPin(
        tape.pin, observer, start, through, _digest(tape, observer, start, through, rows)
    )


def qualify(
    tape: Tape, observations: Iterable[Observation], expected: ObserverPin
) -> Checkpoint | None:
    """Require the expected source and observer pins; None is not no activity."""
    if not isinstance(expected, ObserverPin):
        raise ValueError("independent observer pin required")
    if tape.pin != expected.revision_pin:
        return None
    rows = _observations(tape, expected.start, expected.through, observations)
    if _digest(tape, expected.observer, expected.start, expected.through, rows) != expected.sha256:
        return None
    return Checkpoint(expected, tape, rows)

"""Bind exact selected numeric values to declared revisions across row reordering.

This is an association/consistency guard, NOT provider authentication. A caller can
backdate metadata or swap identities and construct a new self-consistent bundle;
SHA256 cannot detect that without an external trusted anchor. Selection is latest
only relative to supplied history: omitted candidates cannot be discovered. A
bundle does not attest that selection policy executed or history was complete;
a caller may directly bundle an eligible older revision. Identical payloads with
forged/swapped physical identities cannot recover discarded provenance. Python frozen
objects are convenience invariants, not a security boundary.

Order: retain stable sample/source IDs and raw revisions; select at the decision
BEFORE cleaning drops nulls or resolves duplicate events; retain bundles beside
cleaned rows; rejoin by IDs and compare exact payloads. This strict API rejects
all unknown clocks/nonfinite or null payloads globally, including future records.
It never drops an invalid latest revision to resurrect an older value. A later
valid publication is excluded at earlier decisions by the explicit as-of policy.

Payloads are flat named builtin integers/floats. Types and signed zero matter:
integer 1 differs from float 1.0. Cleaner coercion, rounding, renamed fields and
learned transformations fail exact binding. If needed, explicitly normalize
values with precision-loss checks BEFORE constructing revisions; this module
performs no numeric coercion. Mapping inputs cannot reveal duplicate fields
already collapsed upstream; use pairs to reject duplicates before collapse.
Normal package import still initializes existing native/package dependencies.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import NamedTuple, cast

Key = tuple[str, str]
Payload = tuple[tuple[str, int | float], ...]
_SCHEMA = "openquant.selected-numeric-revision.v1"


def _name(value: object) -> str:
    if type(value) is not str or not value:
        raise ValueError("IDs and field names must be nonempty builtin strings")
    return value


def _key(value: object) -> Key:
    if type(value) is not tuple or len(value) != 2:
        raise ValueError("identity must be a (row_id, source_id) tuple")
    return _name(value[0]), _name(value[1])


def _time(value: object) -> int:
    if type(value) is not int or not -(2**63) < value < 2**63:
        raise ValueError("clock must be exact builtin int64 UTC Unix ns, excluding NaT")
    return value


def _payload(value: object) -> Payload:
    if isinstance(value, Mapping):
        value = value.items()
    if isinstance(value, (str, bytes)) or not isinstance(value, Iterable):
        raise ValueError("payload must be a mapping or iterable of named numeric pairs")
    result: dict[str, int | float] = {}
    for pair in value:
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            raise ValueError("payload entries must be name/value pairs")
        name, number = _name(pair[0]), pair[1]
        if name in result:
            raise ValueError("duplicate payload field")
        if type(number) not in (int, float) or (
            type(number) is float and not math.isfinite(number)
        ):
            raise ValueError("payload values must be finite builtin int or float, never null/bool")
        result[name] = number
    if not result:
        raise ValueError("payload must contain at least one field")
    return tuple(sorted(result.items()))


def _encoded_payload(payload: Payload) -> list[list[str]]:
    return [
        [
            name,
            "int" if type(value) is int else "float",
            hex(value) if type(value) is int else cast(float, value).hex(),
        ]
        for name, value in payload
    ]


@dataclass(frozen=True, eq=False)
class Revision:
    """Candidate for a stable sample row and source stream; revision IDs scoped to both.

    Multiple event times may belong to a sample/source group. The caller owns the
    group definition; this is not inference of identity from timestamps or OHLCV.
    event_ns means an observed input event, not a forecast target/validity time;
    availability must not precede that observation. Integers use canonical hex
    encoding, so no decimal conversion limit or binary64 rounding is imposed.
    Use ``make_revision`` to accept mappings or pairs. Direct construction also
    validates all fields. Payload field order is normalized, numeric values are not.
    """

    row_id: str
    source_id: str
    revision_id: str
    event_ns: int
    available_ns: int
    payload: Payload

    def __post_init__(self):
        for name in (self.row_id, self.source_id, self.revision_id):
            _name(name)
        _time(self.event_ns)
        _time(self.available_ns)
        if self.available_ns < self.event_ns:
            raise ValueError("availability cannot precede the declared observation event")
        object.__setattr__(self, "payload", _payload(self.payload))

    def _equality_key(self) -> tuple:
        # Native numeric equality conflates int/float and signed zero. Reuse the
        # digest's exact typed tokens so sets/dicts cannot erase conflict evidence.
        return (
            self.row_id,
            self.source_id,
            self.revision_id,
            self.event_ns,
            self.available_ns,
            tuple(tuple(field) for field in _encoded_payload(self.payload)),
        )

    def __eq__(self, other: object) -> bool:
        if type(self) is not type(other):
            return NotImplemented
        return self._equality_key() == cast(Revision, other)._equality_key()

    def __hash__(self) -> int:
        return hash(self._equality_key())

    @property
    def key(self) -> Key:
        """The `(row_id, source_id)` identity the revision belongs to."""
        return self.row_id, self.source_id


def make_revision(
    row_id: str, source_id: str, revision_id: str, event_ns: int, available_ns: int, payload: object
) -> Revision:
    """Construct an immutable, validated candidate without coercing numeric values."""
    return Revision(row_id, source_id, revision_id, event_ns, available_ns, cast(Payload, payload))


def _record(revision: Revision) -> list:
    return [
        revision.row_id,
        revision.source_id,
        revision.revision_id,
        str(revision.event_ns),
        str(revision.available_ns),
        _encoded_payload(revision.payload),
    ]


def _serialize(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), allow_nan=False).encode(
        "ascii"
    )


@dataclass(frozen=True)
class SelectedRevision:
    """Schema-tagged digest binds decision, IDs, clocks and exact typed payload.

    Digest is consistency evidence, not a signature or external attestation.
    Reconstructing a bundle with fabricated but consistent inputs recomputes it.
    """

    revision: Revision
    decision_ns: int
    digest: str = field(init=False)

    def __post_init__(self):
        if type(self.revision) is not Revision:
            raise ValueError("bundle requires a validated Revision")
        _time(self.decision_ns)
        if max(self.revision.event_ns, self.revision.available_ns) > self.decision_ns:
            raise ValueError("selected revision is not yet eligible at its decision")
        encoded = [_SCHEMA, str(self.decision_ns), _record(self.revision)]
        object.__setattr__(self, "digest", hashlib.sha256(_serialize(encoded)).hexdigest())


@dataclass(frozen=True)
class Selection:
    """Bundles and missing groups, each in original decisions mapping order."""

    bundles: tuple[SelectedRevision, ...]
    missing_keys: tuple[Key, ...]

    def __post_init__(self):
        object.__setattr__(self, "bundles", tuple(self.bundles))
        object.__setattr__(self, "missing_keys", tuple(_key(k) for k in self.missing_keys))
        if any(type(bundle) is not SelectedRevision for bundle in self.bundles):
            raise ValueError("selection requires validated bundles")
        keys = [bundle.revision.key for bundle in self.bundles] + list(self.missing_keys)
        if len(keys) != len(set(keys)):
            raise ValueError("selection has duplicate or both selected and missing identities")


def select_revisions(revisions: Iterable[Revision], decisions: Mapping[Key, int]) -> Selection:
    """Select latest eligible (event, availability), before canonical row cleaning.

    All supplied records and decisions validate. Same full revision identities
    with conflicting content reject globally, even if future or unrequested.
    Exact duplicates are idempotent. Distinct winning revision IDs at equal clocks
    reject even if their values agree. No lexical revision-ID tiebreak is implied.
    Unknown decision groups return missing, never a prior row or positional match.
    """
    if not isinstance(decisions, Mapping):
        raise ValueError("decisions must map identity tuples to exact UTC-ns clocks")
    cutoffs = [(_key(key), _time(value)) for key, value in decisions.items()]
    groups: dict[Key, list[Revision]] = {}
    seen: dict[tuple[str, str, str], bytes] = {}
    for revision in revisions:
        if type(revision) is not Revision:
            raise ValueError("all candidates must be validated Revision objects")
        identity = (*revision.key, revision.revision_id)
        encoded = _serialize(_record(revision))
        if identity in seen:
            if seen[identity] != encoded:
                raise ValueError("conflicting content for one revision identity")
            continue
        seen[identity] = encoded
        groups.setdefault(revision.key, []).append(revision)
    bundles, missing = [], []
    for key, cutoff in cutoffs:
        eligible = [
            r for r in groups.get(key, []) if r.event_ns <= cutoff and r.available_ns <= cutoff
        ]
        if not eligible:
            missing.append(key)
            continue
        winning = max((r.event_ns, r.available_ns) for r in eligible)
        tied = [r for r in eligible if (r.event_ns, r.available_ns) == winning]
        if len(tied) != 1:
            raise ValueError("ambiguous winning revision clocks")
        bundles.append(SelectedRevision(tied[0], cutoff))
    return Selection(tuple(bundles), tuple(missing))


class RejoinResult(NamedTuple):
    """Bundles in incoming row order; their original selection positions and drops.

    Dropped keys are in original selected order. Missing selection groups remain
    missing; they cannot be reintroduced as cleaned rows.
    """

    bundles: tuple[SelectedRevision, ...]
    selection_indices: tuple[int, ...]
    dropped_keys: tuple[Key, ...]


def rejoin_selected(
    selection: Selection, rows: Iterable[tuple[str, str, int, object]], *, require_all: bool = True
) -> RejoinResult:
    """Bind reordered/subset cleaned rows to exactly matching selected payloads.

    Every row is (row_id, source_id, event_ns, numeric payload). Duplicate/unknown keys and
    any typed-value mismatch fail closed. No positional fallback, rounding
    tolerance, field dropping or revision reselection. Empty rows are permitted
    only with explicit require_all=False and dropped-key output.
    """
    if type(selection) is not Selection or type(require_all) is not bool:
        raise ValueError("require a Selection and boolean require_all")
    # Revalidate at the boundary: catches stale/malformed transferred state, not
    # coordinated forgery. No deserializer or external trust is implied here.
    Selection(selection.bundles, selection.missing_keys)
    for bundle in selection.bundles:
        r = bundle.revision
        checked = Revision(
            r.row_id, r.source_id, r.revision_id, r.event_ns, r.available_ns, r.payload
        )
        if SelectedRevision(checked, bundle.decision_ns).digest != bundle.digest:
            raise ValueError("selected bundle digest is inconsistent with its current content")
    lookup = {b.revision.key: (i, b) for i, b in enumerate(selection.bundles)}
    seen: set[Key] = set()
    bundles, indices = [], []
    for row in rows:
        if not isinstance(row, (tuple, list)) or len(row) != 4:
            raise ValueError("rows must contain row_id, source_id, event_ns and payload")
        key = _key((row[0], row[1]))
        if key in seen or key not in lookup:
            raise ValueError("duplicate or unknown cleaned row identity")
        index, bundle = lookup[key]
        if _time(row[2]) != bundle.revision.event_ns:
            raise ValueError("cleaned event differs from bound revision")
        if _encoded_payload(_payload(row[3])) != _encoded_payload(bundle.revision.payload):
            raise ValueError("cleaned payload differs from bound revision")
        seen.add(key)
        bundles.append(bundle)
        indices.append(index)
    dropped = tuple(
        bundle.revision.key for bundle in selection.bundles if bundle.revision.key not in seen
    )
    if require_all and dropped:
        raise ValueError("cleaning dropped selected identities")
    return RejoinResult(tuple(bundles), tuple(indices), dropped)

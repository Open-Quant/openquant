"""Project complete, ordered numeric feature rows from selected source bundles.

Clocks are exact builtin int64 UTC Unix nanoseconds, excluding NaT. Values keep
builtin numeric types and signed zero; no transforms, imputation or coercion run.
Aliases may project the same source field into distinct named columns; this does
not imply independence. Each contributing source bundle is retained once.

Every supplied bundle is revalidated, including unused records. Valid unused
sources do not change projection. Required sources for requested rows must have
exactly the requested decision, even when a required field is missing. Missing
inputs exclude rows explicitly; availability is aggregated only for complete
rows. Coverage does not establish representativeness. The declared specification
cannot prove it includes every real dependency, and provenance is an association
guard, not authentication. Mapping keys collapsed upstream cannot be recovered.
Normal package imports still initialize the existing native dependencies.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

from .provenance import SelectedRevision, Selection, rejoin_selected

FeatureSpec = tuple[str, str, str]


def _name(value: object) -> str:
    if type(value) is not str or not value:
        raise ValueError("identifiers must be nonempty builtin strings")
    return value


def _clock(value: object) -> int:
    if type(value) is not int or not -(2**63) < value < 2**63:
        raise ValueError("decision must be exact builtin int64 UTC Unix ns, excluding NaT")
    return value


@dataclass(frozen=True)
class FeatureRow:
    """Output record; assemble_features establishes lineage and clock invariants.

    Direct construction is not a validation API. Values use typed equality.
    """

    row_id: str
    decision_ns: int
    values: tuple[int | float, ...] = field(compare=False)
    bundles: tuple[SelectedRevision, ...]
    max_available_ns: int
    _value_tokens: tuple[tuple[str, str], ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        values = tuple(self.values)
        tokens = []
        for value in values:
            if type(value) is int:
                tokens.append(("int", hex(value)))
            elif type(value) is float and math.isfinite(value):
                tokens.append(("float", value.hex()))
            else:
                raise ValueError("feature values must be finite builtin integers/floats")
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "bundles", tuple(self.bundles))
        object.__setattr__(self, "_value_tokens", tuple(tokens))


@dataclass(frozen=True)
class MissingFeature:
    """One declared feature an excluded row lacks.

    `reason` is `"missing_source"` (no bundle for the row and source) or
    `"missing_field"` (the bundle's payload has no such field).
    """

    name: str
    source_id: str
    payload_field: str
    reason: str


@dataclass(frozen=True)
class ExcludedRow:
    """A requested row left out of the output, with every feature it lacks."""

    row_id: str
    decision_ns: int
    missing: tuple[MissingFeature, ...]


@dataclass(frozen=True)
class FeatureCoverage:
    """Per declared feature, over all requested rows: present, missing source, missing field."""

    name: str
    source_id: str
    payload_field: str
    present: int
    missing_source: int
    missing_field: int


@dataclass(frozen=True)
class AssemblyResult:
    """What `assemble_features` returns: complete rows, excluded rows and coverage."""

    feature_spec: tuple[FeatureSpec, ...]
    requested_rows: tuple[tuple[str, int], ...]
    rows: tuple[FeatureRow, ...]
    excluded: tuple[ExcludedRow, ...]
    coverage: tuple[FeatureCoverage, ...]

    @property
    def requested_count(self) -> int:
        """Number of requested rows."""
        return len(self.requested_rows)

    @property
    def complete_count(self) -> int:
        """Number of complete output rows."""
        return len(self.rows)

    @property
    def excluded_count(self) -> int:
        """Number of requested rows excluded for a missing source or field."""
        return len(self.excluded)


def assemble_features(
    feature_spec: Iterable[FeatureSpec],
    decisions: Mapping[str, int],
    selection: Selection,
) -> AssemblyResult:
    """Return complete rows in request order, values in declared feature order.

    Empty requests are valid; an empty specification is not. Missing source and
    field counts are per feature over all requests, including excluded rows.
    No empty or partial row receives an aggregate availability timestamp.
    """
    if isinstance(feature_spec, (str, bytes)) or not isinstance(feature_spec, Iterable):
        raise ValueError("feature specification must be an ordered iterable of triples")
    if isinstance(feature_spec, (set, frozenset, Mapping)):
        raise ValueError("feature specification must have explicit sequence order")
    specs = []
    names = set()
    for entry in feature_spec:
        if not isinstance(entry, (tuple, list)) or len(entry) != 3:
            raise ValueError("each feature specification must be a name/source/field triple")
        name, source, payload_field = (_name(item) for item in entry)
        if name in names:
            raise ValueError("duplicate output feature name")
        names.add(name)
        specs.append((name, source, payload_field))
    if not specs:
        raise ValueError("feature specification must not be empty")
    if not isinstance(decisions, Mapping):
        raise ValueError("decisions must map row IDs to decision clocks")
    requests = []
    seen = set()
    for row_id, decision in decisions.items():
        row_id, decision = _name(row_id), _clock(decision)
        if row_id in seen:
            raise ValueError("duplicate requested row ID")
        seen.add(row_id)
        requests.append((row_id, decision))

    # Public API reconstructs every bundle and checks its bound digest.
    rejoin_selected(selection, (), require_all=False)
    lookup = {bundle.revision.key: bundle for bundle in selection.bundles}
    counts = [[0, 0, 0] for _ in specs]
    rows, excluded = [], []
    for row_id, decision in requests:
        values, missing = [], []
        contributing: dict[str, SelectedRevision] = {}
        payloads = {}
        for index, (name, source, payload_field) in enumerate(specs):
            bundle = lookup.get((row_id, source))
            if bundle is None:
                counts[index][1] += 1
                missing.append(MissingFeature(name, source, payload_field, "missing_source"))
                continue
            if bundle.decision_ns != decision:
                raise ValueError("required source bundle decision differs from requested decision")
            if source not in payloads:
                payloads[source] = dict(bundle.revision.payload)
            payload = payloads[source]
            if payload_field not in payload:
                counts[index][2] += 1
                missing.append(MissingFeature(name, source, payload_field, "missing_field"))
                continue
            counts[index][0] += 1
            values.append(payload[payload_field])
            contributing.setdefault(source, bundle)
        if missing:
            excluded.append(ExcludedRow(row_id, decision, tuple(missing)))
        else:
            bundles = tuple(contributing.values())
            rows.append(
                FeatureRow(
                    row_id,
                    decision,
                    tuple(values),
                    bundles,
                    max(bundle.revision.available_ns for bundle in bundles),
                )
            )
    return AssemblyResult(
        tuple(specs),
        tuple(requests),
        tuple(rows),
        tuple(excluded),
        tuple(FeatureCoverage(*spec, *count) for spec, count in zip(specs, counts)),
    )

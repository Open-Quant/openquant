"""Offline point-in-time illustration; synthetic data, not a provider/product API.

Dates use YYYY-MM-DD and availability timestamps use UTC YYYY-MM-DDTHH:MM:SSZ.
Revision date orders eligible versions; tied latest revision dates are ambiguous.
Hashes identify exact fixture content, not authenticity or economic correctness.
"""

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from datetime import UTC, date, datetime
from decimal import Decimal


@dataclass(frozen=True)
class Record:
    provider: str
    series: str
    observation_date: str
    revision_date: str
    available_at: str
    value: str
    version: str


def utc_timestamp(value: str) -> datetime:
    parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)
    if parsed.strftime("%Y-%m-%dT%H:%M:%SZ") != value:
        raise ValueError("availability/cutoff must use canonical UTC timestamps")
    return parsed


def validate(record: Record) -> None:
    for value in (record.observation_date, record.revision_date):
        if date.fromisoformat(value).isoformat() != value:
            raise ValueError("dates must use YYYY-MM-DD")
    utc_timestamp(record.available_at)
    if not all((record.provider, record.series, record.version)):
        raise ValueError("provider, series and version must be nonempty")
    if not isinstance(record.value, str) or not Decimal(record.value).is_finite():
        raise ValueError("value must be a finite decimal string")


def canonical_json(value: object) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    )


def digest(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def archive_identity(records: tuple[Record, ...]) -> str:
    """Content identity of every raw fixture record, independent of input row order."""
    for record in records:
        validate(record)
    rows = sorted((asdict(record) for record in records), key=canonical_json)
    return digest({"schema": "synthetic-archive-v1", "records": rows})


def select(records: tuple[Record, ...], cutoff: str | None) -> tuple[Record, ...]:
    """None means latest within this fixture, never latest from a live provider."""
    boundary = None if cutoff is None else utc_timestamp(cutoff)
    groups: dict[tuple[str, str, str], list[Record]] = {}
    for record in records:
        validate(record)
        if boundary is not None and utc_timestamp(record.available_at) > boundary:
            continue
        key = (record.provider, record.series, record.observation_date)
        groups.setdefault(key, []).append(record)
    chosen = []
    for key in sorted(groups):
        newest = max(record.revision_date for record in groups[key])
        candidates = [r for r in groups[key] if r.revision_date == newest]
        if len(candidates) != 1:
            raise ValueError(f"ambiguous latest eligible revision for {key}")
        chosen.append(candidates[0])
    return tuple(chosen)


def selected_identity(records: tuple[Record, ...], cutoff: str | None) -> str:
    """Bind values, provenance and selection policy; exclude unrelated raw rows."""
    return digest(
        {
            "schema": "synthetic-selection-v1",
            "policy": "latest-revision-date; reject-ties; available-at-inclusive-v1",
            "mode": "latest-within-fixture" if cutoff is None else "as-of",
            "cutoff": cutoff,
            "records": [asdict(record) for record in select(records, cutoff)],
        }
    )


def main() -> None:
    initial = Record(
        "synthetic",
        "activity",
        "2024-01-01",
        "2024-02-01",
        "2024-02-01T12:00:00Z",
        "100.0",
        "release-1",
    )
    revised = replace(
        initial,
        revision_date="2024-03-01",
        available_at="2024-03-01T12:00:00Z",
        value="102.0",
        version="release-2",
    )
    next_observation = replace(
        initial,
        observation_date="2024-02-01",
        revision_date="2024-03-02",
        available_at="2024-03-02T12:00:00Z",
        value="103.0",
        version="release-3",
    )
    records = (revised, next_observation, initial)  # Deliberately not chronological.
    cutoff = "2024-02-15T00:00:00Z"
    assert select(records, cutoff) == (initial,)  # Future revision and release excluded.
    assert select(records, "2024-02-01T11:59:59Z") == ()
    assert select(records, initial.available_at) == (initial,)  # Inclusive boundary.
    assert select(records, revised.available_at) == (revised,)
    assert select(records, None) == (revised, next_observation)
    selected_hash = selected_identity(records, cutoff)
    assert selected_hash != selected_identity(records, None)
    assert selected_hash == selected_identity(tuple(reversed(records)), cutoff)
    assert archive_identity(records) == archive_identity(tuple(reversed(records)))
    # Changed selected values, versions or provider provenance change selected identity.
    for changed in (
        replace(initial, value="100.1"),
        replace(initial, version="corrected"),
        replace(initial, provider="other-synthetic"),
    ):
        assert selected_hash != selected_identity((revised, next_observation, changed), cutoff)
    assert selected_hash != selected_identity(records, "2024-02-16T00:00:00Z")
    future = replace(
        next_observation,
        observation_date="2024-03-01",
        revision_date="2024-04-01",
        available_at="2024-04-01T12:00:00Z",
        version="release-4",
    )
    expanded = records + (future,)
    assert archive_identity(records) != archive_identity(expanded)
    assert selected_hash == selected_identity(expanded, cutoff)
    # Even equal values do not resolve conflicting version identities at a revision date.
    try:
        select(records + (replace(initial, version="conflicting-version"),), cutoff)
    except ValueError as error:
        assert "ambiguous" in str(error)
    else:
        raise AssertionError("ambiguous eligible revisions must fail closed")
    print("as-of value: 100.0; latest fixture value: 102.0")
    print("release boundaries, ambiguity and content identities verified")


if __name__ == "__main__":
    main()

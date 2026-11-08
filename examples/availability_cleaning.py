"""Exercise the actual cleaner with revision sidecars; synthetic integration only.

Run with --source-tree to load data.py and availability.py directly. This uses
real Polars and the real cleaner, not a mock, but does not validate native package
initialization. The adapter is deliberately restricted to exact microsecond event
keys, unique revision record IDs and one decision per event key. It is an example, not a provider adapter.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

EPOCH = datetime(1970, 1, 1, tzinfo=UTC)


def to_ns(value):
    delta = value - EPOCH
    return ((delta.days * 86400 + delta.seconds) * 1_000_000 + delta.microseconds) * 1000


def load_modules(source_tree):
    if not source_tree:
        from openquant import availability, data

        return availability, data
    modules = []
    for name in ("availability", "data"):
        path = Path(__file__).resolve().parents[1] / "python/openquant" / f"{name}.py"
        spec = importlib.util.spec_from_file_location(f"cleaning_example_{name}", path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        modules.append(module)
    return tuple(modules)


def historical_records(records):
    """Select explicit historical vintages before canonicalization, failing on ties."""
    groups = {}
    ids = set()
    for record in records:
        if record["record_id"] in ids:
            raise ValueError("revision record IDs must be unique")
        ids.add(record["record_id"])
        event = record["event_ns"]
        if type(event) is not int or event % 1000:
            raise ValueError("this cleaner adapter requires exact microsecond event keys")
        key = record["symbol"], event
        groups.setdefault(key, []).append(record)
    selected = []
    for group in groups.values():
        decisions = {record["decision_ns"] for record in group}
        if len(decisions) != 1:
            raise ValueError("revisions of one event must share its historical decision")
        eligible = [
            record
            for record in group
            if record["available_ns"] is not None
            and record["event_ns"] <= record["decision_ns"]
            and record["available_ns"] <= record["decision_ns"]
        ]
        if not eligible:
            continue
        latest = max(record["available_ns"] for record in eligible)
        winners = [record for record in eligible if record["available_ns"] == latest]
        if len(winners) != 1:
            raise ValueError("ambiguous latest revision publication")
        selected.append(winners[0])
    return selected


def frame(records):
    import polars as pl

    rows = []
    for record in records:
        rows.append(
            dict(
                ts=EPOCH + timedelta(microseconds=record["event_ns"] // 1000),
                symbol=record["symbol"],
                open=1.0,
                high=200.0,
                low=0.0,
                close=record["value"],
                volume=1.0,
                record_id=record["record_id"],
            )
        )
    return pl.DataFrame(rows)


def align_cleaned(cleaned, selected):
    """Rejoin by an injective canonical key, never the pre-cleaning row position."""
    sidecar = {(r["symbol"], r["event_ns"]): r for r in selected}
    if len(sidecar) != len(selected):
        raise ValueError("sidecar key must identify exactly one selected revision")
    aligned = []
    for row in cleaned.iter_rows(named=True):
        # Actual cleaner documents UTC-naive microseconds. This explicit attachment
        # is valid only for that known output contract, not arbitrary naive input.
        key = row["symbol"], to_ns(row["ts"].replace(tzinfo=UTC))
        if key not in sidecar:
            raise ValueError("cleaned output has no exact source identity")
        aligned.append(sidecar[key])
    return aligned


def fixture():
    def record(day, suffix, value, release_day=None):
        event = datetime(2020, 1, day, 9, tzinfo=UTC)
        release = datetime(2020, 1, release_day or day, 9, 10, tzinfo=UTC)
        return dict(
            record_id=f"{day}-{suffix}",
            symbol="S",
            event_ns=to_ns(event),
            decision_ns=to_ns(event + timedelta(hours=1)),
            available_ns=to_ns(release),
            value=value,
        )

    return [
        record(2, "initial", 20.0),
        record(1, "initial", 10.0),
        record(1, "revision", 100.0, 3),
        record(3, "missing", None),
    ]


def run_example(availability, data):
    records = fixture()
    chosen = historical_records(records)
    safe, quality = data.clean_ohlcv(frame(chosen), return_report=True)
    aligned = align_cleaned(safe, chosen)
    fit = to_ns(datetime(2020, 1, 5, tzinfo=UTC))

    def audit(metadata):
        decisions = [r["decision_ns"] for r in metadata]
        return availability.select_available_training(
            decisions,
            [t + 3600 * 10**9 for t in decisions],
            [t + 7200 * 10**9 for t in decisions],
            [r["available_ns"] for r in metadata],
            fit_time=fit,
            clock="unix_ns",
        )

    archive = data.clean_ohlcv(frame(records))
    # Independent fixture truth: values uniquely identify each revision within
    # its event key. This is NOT a general way to recover discarded provenance.
    actual_archive = []
    for row in archive.iter_rows(named=True):
        event = to_ns(row["ts"].replace(tzinfo=UTC))
        matches = [
            r
            for r in records
            if r["symbol"] == row["symbol"]
            and r["event_ns"] == event
            and r["value"] == row["close"]
        ]
        if len(matches) != 1:
            raise ValueError("fixture truth requires uniquely identifiable values")
        actual_archive.append(matches[0])
    # False provenance: attaching an INITIAL release to a FINAL revision's value.
    false_sidecar = align_cleaned(archive, chosen)
    safe_audit, false_audit = audit(aligned), audit(false_sidecar)
    positional = chosen[: safe.height]
    positional_audit = audit(positional)
    return dict(
        synthetic_only=True,
        safe=dict(
            record_ids=[r["record_id"] for r in aligned],
            values=safe["close"].to_list(),
            eligible_indices=list(safe_audit.eligible_indices),
            dropped_null_rows=quality["null_counts"]["close"],
        ),
        false_revision_metadata=dict(
            values=archive["close"].to_list(),
            eligible_indices=list(false_audit.eligible_indices),
            actual_late_revision_rows=[
                i for i, r in enumerate(actual_archive) if r["available_ns"] > r["decision_ns"]
            ],
        ),
        positional_metadata=dict(
            assigned_ids=[r["record_id"] for r in positional],
            true_ids=[r["record_id"] for r in aligned],
            eligible_indices=list(positional_audit.eligible_indices),
            misaligned_rows=sum(
                a["record_id"] != b["record_id"] for a, b in zip(positional, aligned, strict=True)
            ),
        ),
        conclusions=[
            "Select historical revisions before cleaning; preserve exact source IDs.",
            "Passing supplied clocks cannot authenticate their association with values.",
            "Canonical key joins require uniqueness and no timestamp precision loss.",
        ],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-tree", action="store_true")
    args = parser.parse_args()
    availability, data = load_modules(args.source_tree)
    result = run_example(availability, data)
    result["import_mode"] = "isolated_real_modules" if args.source_tree else "installed_package"
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

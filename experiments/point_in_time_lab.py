"""Availability-aware revision lookup and synthetic delayed-publication controls."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import time
from datetime import UTC, datetime, timedelta
from numbers import Integral, Real
from pathlib import Path

import numpy as np

FIELDS = {"entity", "feature", "event_time", "available_time", "value", "revision_id"}
FAMILIES = ("on_time", "delayed_release", "late_revision", "signal_then_revision")
MODES = ("point_in_time", "event_only_final_revision")


def utc_timestamp(value):
    """Require an aware datetime/ISO timestamp; normalize offsets without guessing."""
    if isinstance(value, str):
        grammar = r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?(?:Z|[+-]\d{2}:\d{2})"
        if not re.fullmatch(grammar, value, flags=re.ASCII) or value.endswith("-00:00"):
            raise ValueError(
                "timestamps require seconds, at most microseconds, and Z or a known ±HH:MM offset"
            )
        value = datetime.fromisoformat(value)
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamps must be timezone-aware datetimes or ISO strings")
    return value.astimezone(UTC)


def _name(value, field):
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a nonempty string")
    return value


class PointInTimeIndex:
    """Immutable row snapshots; query ambiguities are checked only at selected keys.

    This simple reference scans one entity/feature group per query. It is not a
    production database, an availability timestamp validator or a global data audit.
    """

    def __init__(self, records):
        grouped = {}
        for record in records:
            if not isinstance(record, dict) or set(record) != FIELDS:
                raise ValueError("records require exactly the six declared fields")
            entity, feature = _name(record["entity"], "entity"), _name(record["feature"], "feature")
            revision = _name(record["revision_id"], "revision_id")
            event = utc_timestamp(record["event_time"])
            available = utc_timestamp(record["available_time"])
            value = record["value"]
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
                raise ValueError("value must be a finite real number, not a boolean")
            original = value
            try:
                value = float(value)
            except (OverflowError, ValueError) as exc:
                raise ValueError("value must be representable as finite binary64") from exc
            if not math.isfinite(value):
                raise ValueError("value must be finite")
            if isinstance(original, Integral):
                if int(original) != int(value):
                    raise ValueError("integer value loses precision in binary64")
            elif original != value:
                raise ValueError("value loses precision in binary64")
            value = 0.0 if value == 0 else value
            revisions = grouped.setdefault((entity, feature), {}).setdefault((event, available), {})
            revisions.setdefault(value, set()).add(revision)
        self._groups = {
            key: tuple(
                (
                    event,
                    available,
                    tuple((value, tuple(sorted(ids))) for value, ids in sorted(values.items())),
                )
                for (event, available), values in sorted(rows.items(), reverse=True)
            )
            for key, rows in grouped.items()
        }

    def lookup(self, entity, feature, decision_time, *, mode="point_in_time"):
        entity, feature = _name(entity, "entity"), _name(feature, "feature")
        decision = utc_timestamp(decision_time)
        if mode not in MODES:
            raise ValueError("unsupported lookup mode")
        result = {
            "entity": entity,
            "feature": feature,
            "decision_time": decision.isoformat(),
            "status": "missing",
            "selected": None,
        }
        for event, available, values in self._groups.get((entity, feature), ()):
            if event > decision or (mode == "point_in_time" and available > decision):
                continue
            if len(values) != 1:
                raise ValueError(
                    "conflicting values share the selected event and availability timestamps"
                )
            value, revisions = values[0]
            result.update(
                status="selected",
                selected={
                    "event_time": event.isoformat(),
                    "available_time": available.isoformat(),
                    "value": value,
                    "revision_ids": list(revisions),
                },
            )
            break
        return result


def _integer(value, name, minimum):
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or value < minimum
    ):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def synthetic_records(family, labels, errors, start):
    """Latent labels are revealed next day; proxies precede revelation, revisions follow it."""
    rows = []
    for day, (label, error) in enumerate(zip(labels, errors, strict=True)):
        event = start + timedelta(days=day)
        initial = int(label) ^ int(error)
        available = event + timedelta(hours=18 if family == "delayed_release" else 6)
        rows.append(
            dict(
                entity="A",
                feature="proxy",
                event_time=event.isoformat(),
                available_time=available.isoformat(),
                value=2 * initial - 1,
                revision_id=f"initial-{day}",
            )
        )
        if family in ("late_revision", "signal_then_revision"):
            rows.append(
                dict(
                    entity="A",
                    feature="proxy",
                    event_time=event.isoformat(),
                    available_time=(event + timedelta(days=2)).isoformat(),
                    value=2 * int(label) - 1,
                    revision_id=f"revised-{day}",
                )
            )
    return rows


def run_lab(*, repetitions=200, days=64, seed=97, families=FAMILIES):
    started = time.perf_counter()
    repetitions, days, seed = (
        _integer(repetitions, "repetitions", 1),
        _integer(days, "days", 2),
        _integer(seed, "seed", 0),
    )
    families = tuple(families)
    if (
        not families
        or len(set(families)) != len(families)
        or any(f not in FAMILIES for f in families)
    ):
        raise ValueError("families must be distinct supported names")
    start = datetime(2020, 1, 1, tzinfo=UTC)
    output = {}
    for family in families:
        records, audit = [], None
        for rep in range(repetitions):
            rng = np.random.default_rng(np.random.SeedSequence([seed, FAMILIES.index(family), rep]))
            labels = rng.integers(0, 2, size=days)
            errors = rng.random(days) < (0.5 if family == "late_revision" else 0.25)
            rows = synthetic_records(family, labels, errors, start)
            index = PointInTimeIndex(rows)
            methods = {}
            decisions = []
            for mode in MODES:
                selected, correct, future, stale = 0, 0, 0, 0
                for day, label in enumerate(labels):
                    event = start + timedelta(days=day)
                    decision = event + timedelta(hours=12)
                    choice = index.lookup("A", "proxy", decision, mode=mode)
                    chosen = choice["selected"]
                    prediction = None if chosen is None else int(chosen["value"] > 0)
                    if chosen is not None:
                        selected += 1
                        correct += prediction == int(label)
                        future += utc_timestamp(chosen["available_time"]) > decision
                        stale += utc_timestamp(chosen["event_time"]) < event
                    if rep == 0:
                        decisions.append(
                            dict(
                                mode=mode,
                                label=int(label),
                                label_available_time=(event + timedelta(days=1)).isoformat(),
                                prediction=prediction,
                                **choice,
                            )
                        )
                methods[mode] = {
                    "selected": selected,
                    "missing": days - selected,
                    "correct": int(correct),
                    "correct_fraction_all_decisions": correct / days,
                    "accuracy_given_selected": correct / selected if selected else None,
                    "future_publications_used": int(future),
                    "older_events_used": int(stale),
                }
            payload = {"records": rows, "labels": labels.tolist()}
            records.append(
                {
                    "replication": rep,
                    "input_sha256": hashlib.sha256(
                        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
                    ).hexdigest(),
                    "methods": methods,
                    "constant_positive_correct": int(labels.sum()),
                }
            )
            if rep == 0:
                audit = {"source_records": rows, "decisions": decisions}
        summary = {}
        for mode in MODES:
            totals = {
                k: sum(r["methods"][mode][k] for r in records)
                for k in (
                    "selected",
                    "missing",
                    "correct",
                    "future_publications_used",
                    "older_events_used",
                )
            }
            total = repetitions * days
            summary[mode] = {
                **totals,
                "total_decisions": total,
                "selected_fraction": totals["selected"] / total,
                "correct_fraction_all_decisions": totals["correct"] / total,
                "accuracy_given_selected": totals["correct"] / totals["selected"]
                if totals["selected"]
                else None,
            }
        for mode in MODES:
            fractions = [r["methods"][mode]["correct_fraction_all_decisions"] for r in records]
            summary[mode]["correct_fraction_mcse_across_independent_replications"] = (
                float(np.std(fractions, ddof=1) / math.sqrt(repetitions))
                if repetitions > 1
                else None
            )
        output[family] = {
            "constant_positive_baseline": {
                "correct": sum(r["constant_positive_correct"] for r in records),
                "total_decisions": repetitions * days,
                "accuracy": sum(r["constant_positive_correct"] for r in records)
                / (repetitions * days),
                "population_accuracy": 0.5,
            },
            "records_per_replication": days
            * (2 if family in ("late_revision", "signal_then_revision") else 1),
            "lookups_per_replication": len(MODES) * days,
            "summary": summary,
            "replications": records,
            "representative_first_replication": audit,
        }
    return {
        "lab": "point_in_time",
        "schema_version": 1,
        "synthetic_only": True,
        "config": {
            "repetitions": repetitions,
            "days": days,
            "seed": seed,
            "families": list(families),
        },
        "provenance": {
            "numpy": np.__version__,
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
        "performance": {"wall_seconds": time.perf_counter() - started},
        "families": output,
        "caveats": [
            "Synthetic labels and publication clocks, not investment performance.",
            "Fixed sign prediction rule; no parameters fitted using future labels.",
            "Publication timestamps are assumed correct; lookup cannot establish actual availability.",
            "Missing decisions are retained; conditional accuracy is descriptive.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = run_lab()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()

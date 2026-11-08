"""Strict selected-revision association through the actual OHLCV cleaner.

Synthetic, explicit Float64 normalization before binding; no provider truth claim.
The adapter rejects nulls, lossy integer casts, submicrosecond event keys and
nonunique canonical keys. It does not silently repair those inputs. Stable IDs
include symbol and event; a payload guard then checks every numeric output value.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

EPOCH = datetime(1970, 1, 1, tzinfo=UTC)
FIELDS = ("open", "high", "low", "close", "volume", "adj_close")


def modules(source_tree):
    if not source_tree:
        from openquant import data, provenance

        return data, provenance
    loaded = []
    for name in ("data", "provenance"):
        spec = importlib.util.spec_from_file_location(
            f"provenance_cleaning_{name}",
            Path(__file__).resolve().parents[1] / "python/openquant" / f"{name}.py",
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        loaded.append(module)
    return tuple(loaded)


def float64_payload(close):
    """Declared normalization, never implicit loss of a large integer or null."""
    if type(close) not in (int, float):
        raise ValueError("finite builtin numeric close required; nulls are not skipped")
    try:
        number = float(close)
    except OverflowError as exc:
        raise ValueError("close cannot fit Float64") from exc
    if not math.isfinite(number) or (type(close) is int and int(number) != close):
        raise ValueError("nonfinite or lossy Float64 normalization")
    return dict(open=1.0, high=200.0, low=0.0, close=number, volume=1.0, adj_close=number)


def bind_inputs(provenance, rows):
    revisions, decisions, keys = [], {}, {}
    for row in rows:
        event = row["event_ns"]
        if type(event) is not int or event % 1000:
            raise ValueError("adapter requires exact microsecond event keys")
        if type(row["symbol"]) is not str or not row["symbol"]:
            raise ValueError("explicit nonempty symbol required")
        row_id = json.dumps([row["symbol"], str(event)], separators=(",", ":"))
        key = row_id, row["source_id"]
        if key in decisions and decisions[key] != row["decision_ns"]:
            raise ValueError("one historical decision required per source/event")
        decisions[key] = row["decision_ns"]
        keys[key] = row["symbol"], event
        revisions.append(
            provenance.make_revision(
                row_id,
                row["source_id"],
                row["revision_id"],
                event,
                row["available_ns"],
                float64_payload(row["close"]),
            )
        )
    selection = provenance.select_revisions(revisions, decisions)
    selected_keys = [keys[b.revision.key] for b in selection.bundles]
    if len(selected_keys) != len(set(selected_keys)):
        raise ValueError("canonical cleaner key merges distinct source identities")
    return selection, keys


def to_frame(selection, keys):
    import polars as pl

    rows = [
        dict(
            ts=EPOCH + timedelta(microseconds=b.revision.event_ns // 1000),
            symbol=keys[b.revision.key][0],
            **dict(b.revision.payload),
        )
        for b in selection.bundles
    ]
    if not rows:
        raise ValueError("no eligible observations; no cleaner fallback")
    return pl.DataFrame(rows)


def rejoin_frame(provenance, selection, keys, cleaned, *, require_all=True):
    inverse = {keys[b.revision.key]: b.revision.key for b in selection.bundles}
    if len(inverse) != len(selection.bundles):
        raise ValueError("nonunique selected canonical keys")
    records = []
    for row in cleaned.iter_rows(named=True):
        # Only actual clean_ohlcv output has this verified UTC-naive contract.
        delta = row["ts"].replace(tzinfo=UTC) - EPOCH
        event = ((delta.days * 86400 + delta.seconds) * 1_000_000 + delta.microseconds) * 1000
        key = row["symbol"], event
        if key not in inverse:
            raise ValueError("unknown cleaned event identity")
        row_id, source_id = inverse[key]
        records.append((row_id, source_id, event, {field: row[field] for field in FIELDS}))
    return provenance.rejoin_selected(selection, records, require_all=require_all)


def fixture():
    day = 86_400_000_000_000
    event = 1_577_836_800_000_000_000
    base = dict(
        symbol="DEMO", source_id="synthetic-source", event_ns=event, decision_ns=event + day
    )
    return [
        dict(base, revision_id="initial", available_ns=event + day // 4, close=10),
        dict(base, revision_id="revised", available_ns=event + 2 * day, close=100),
    ]


def run(data, provenance):
    import polars as pl

    rows = fixture()
    selected, keys = bind_inputs(provenance, rows)
    cleaned = data.clean_ohlcv(to_frame(selected, keys))
    good = rejoin_frame(provenance, selected, keys, cleaned)
    tampered = cleaned.with_columns(pl.lit(100.0).alias("close"), pl.lit(100.0).alias("adj_close"))
    try:
        rejoin_frame(provenance, selected, keys, tampered)
    except ValueError as exc:
        mismatch = str(exc)
    else:
        raise AssertionError("changed payload was accepted")
    # A new, internally consistent lie cannot be authenticated by local hashing.
    forged_rows = [dict(rows[1], available_ns=rows[0]["available_ns"])]
    forged, forged_keys = bind_inputs(provenance, forged_rows)
    forged_cleaned = data.clean_ohlcv(to_frame(forged, forged_keys))
    forged_join = rejoin_frame(provenance, forged, forged_keys, forged_cleaned)
    dropped = rejoin_frame(provenance, selected, keys, cleaned.head(0), require_all=False)
    return dict(
        synthetic_only=True,
        initial_value=10.0,
        revised_value=100.0,
        key_only_baseline_accepts_changed_value=tampered.select("ts", "symbol").equals(
            cleaned.select("ts", "symbol")
        ),
        unchanged_selected_value=cleaned["close"].item(),
        changed_value_rejected=mismatch,
        selected_digest=good.bundles[0].digest,
        self_consistent_backdate_passes=bool(forged_join.bundles),
        false_value=forged_cleaned["close"].item(),
        forged_digest=forged_join.bundles[0].digest,
        explicit_dropped_keys=dropped.dropped_keys,
        limits="Exact declared association, not provider authenticity or complete revision history.",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-tree", action="store_true")
    args = parser.parse_args()
    result = run(*modules(args.source_tree))
    result["import_mode"] = "source_modules" if args.source_tree else "installed_package"
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

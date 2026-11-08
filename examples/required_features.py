"""Required-source projection, explicit coverage and downstream split intersection.

No imputation, fitting, performance estimate or provider adapter. Source-tree mode
loads an isolated package namespace; installed mode exercises normal imports.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import sys
import types
from pathlib import Path

BASE = 1_577_836_800_000_000_000
HOUR = 3_600_000_000_000
SPEC = (("rate", "macro", "rate"), ("price", "market", "close"))


def modules(source_tree):
    if not source_tree:
        from openquant import availability, feature_assembly, provenance

        return availability, feature_assembly, provenance
    name = "_required_features_source"
    root = Path(__file__).resolve().parents[1] / "python/openquant"
    package = types.ModuleType(name)
    package.__path__ = [str(root)]
    sys.modules[name] = package
    return tuple(
        importlib.import_module(f"{name}.{module}")
        for module in ("availability", "feature_assembly", "provenance")
    )


def fixture(provenance):
    chronological = {row: BASE + (i + 1) * HOUR for i, row in enumerate("ABCD")}
    decisions = {row: chronological[row] for row in ("D", "B", "A", "C")}
    revisions = []
    for i, (row, decision) in enumerate(chronological.items()):
        revisions.append(
            provenance.make_revision(
                row,
                "market",
                "initial",
                decision - HOUR,
                decision - HOUR // 2,
                {"close": 10 * (i + 1)},
            )
        )
        if row != "B":
            field = "rate_estimate" if row == "C" else "rate"
            revisions.append(
                provenance.make_revision(
                    row,
                    "macro",
                    "initial",
                    decision - HOUR,
                    decision - HOUR // 4,
                    {field: 1.0 if row == "A" else 3.0},
                )
            )
    source_decisions = {
        (row, source): decision
        for row, decision in decisions.items()
        for source in ("market", "macro")
    }
    selection = provenance.select_revisions(reversed(revisions), source_decisions)
    return decisions, selection


def informative_coverage(assembly, provenance):
    decisions = {f"row-{i}": BASE + HOUR for i in range(16)}
    revisions = [
        provenance.make_revision(row, "market", "one", BASE, BASE, {"close": i})
        for i, row in enumerate(decisions)
    ]
    revisions += [
        provenance.make_revision(f"row-{i}", "macro", "one", BASE, BASE, {"rate": 1.0})
        for i in range(10)
    ]
    source_decisions = {
        (row, source): decision
        for row, decision in decisions.items()
        for source in ("market", "macro")
    }
    result = assembly.assemble_features(
        SPEC, decisions, provenance.select_revisions(revisions, source_decisions)
    )
    targets = {f"row-{i}": 1 if i < 8 else -1 for i in range(16)}
    return dict(
        requested=result.requested_count,
        complete=result.complete_count,
        selected_target_mean=sum(targets[r.row_id] for r in result.rows) / len(result.rows),
        full_target_mean=sum(targets.values()) / len(targets),
        limitation="Deliberately informative completeness; no representativeness guarantee.",
    )


def run(availability, assembly, provenance):
    decisions, selection = fixture(provenance)
    result = assembly.assemble_features(SPEC, decisions, selection)
    # Weak baseline checks only timestamps of sources that happen to be present.
    # It neither verifies that all required inputs exist nor creates valid features.
    baseline = []
    for row, decision in decisions.items():
        known = [b.revision.available_ns for b in selection.bundles if b.revision.row_id == row]
        if known and max(known) <= decision:
            baseline.append(row)
    by_id = {row.row_id: row for row in result.rows}
    ordered_ids = list(decisions)
    # A was withheld by the pre-existing split, despite having complete features.
    candidates = [i for i, row in enumerate(ordered_ids) if row != "A"]
    train = availability.select_available_training(
        list(decisions.values()),
        [d + HOUR // 4 for d in decisions.values()],
        [d + HOUR // 2 for d in decisions.values()],
        [by_id[row].max_available_ns if row in by_id else None for row in ordered_ids],
        fit_time=BASE + 6 * HOUR,
        clock="unix_ns",
        candidate_indices=candidates,
    )
    return dict(
        feature_order=[s[0] for s in result.feature_spec],
        requested_ids=ordered_ids,
        complete_ids=[r.row_id for r in result.rows],
        values=[list(r.values) for r in result.rows],
        source_ids=[[b.revision.source_id for b in row.bundles] for row in result.rows],
        excluded={
            r.row_id: [dict(name=m.name, reason=m.reason) for m in r.missing]
            for r in result.excluded
        },
        coverage=[
            dict(
                name=c.name,
                present=c.present,
                missing_source=c.missing_source,
                missing_field=c.missing_field,
            )
            for c in result.coverage
        ],
        clock_only_baseline_ids=baseline,
        downstream_train_ids=[ordered_ids[i] for i in train.eligible_indices],
        withheld_complete_row_restored="A" in [ordered_ids[i] for i in train.eligible_indices],
        informative_coverage=informative_coverage(assembly, provenance),
        limits="Complete declared inputs only; no authentication, dependency discovery, imputation or model claim.",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-tree", action="store_true")
    args = parser.parse_args()
    report = run(*modules(args.source_tree))
    report["import_mode"] = "isolated_source_namespace" if args.source_tree else "installed_package"
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

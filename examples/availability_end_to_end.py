"""Synthetic availability workflow, not an investment or generalization estimate.

Frozen protocol: seeds 7,19,43; 96 rows; fits at positions 48,64,80; next eight
rows scored. No tuning. Initial releases can miss decisions; labels resolve two
hours later and some publish 36 hours after resolution. Revisions encode y/2 on
purpose, making final-vintage leakage an outcome-encoded positive control.
Four training arms cross PIT/final features and published/resolved labels. Each
model is scored on BOTH PIT and final test features, on identical common IDs,
separating changes in training data from test-feature leakage. Zero uses those
same IDs. All arms retain the pre-existing candidate exclusions.

Run: python examples/availability_end_to_end.py --source-tree
Source-tree mode tests the isolated stdlib helper, not native package import.
Clocks are supplied claims: this example cannot authenticate provider vintages,
serving latency, or establish independence/purging. Stable IDs survive ordering;
real cleaning pipelines must also preserve and verify their identity mapping.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
from pathlib import Path

HOUR = 3_600_000_000_000
EPOCH = 1_577_836_800_000_000_000
PROTOCOL = {"seeds": [7, 19, 43], "rows": 96, "fit_positions": [48, 64, 80], "test_rows": 8}


def load_selector(source_tree=False):
    if not source_tree:
        from openquant.availability import select_available_training

        return select_available_training
    path = Path(__file__).resolve().parents[1] / "python/openquant/availability.py"
    spec = importlib.util.spec_from_file_location("availability_e2e_source", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.select_available_training


def generate(seed):
    rng = random.Random(seed)
    rows, revisions = [], []
    decision = EPOCH
    for position in range(PROTOCOL["rows"]):
        decision += rng.randint(3, 9) * HOUR
        latent = rng.gauss(0, 1)
        label = 2 * latent + rng.gauss(0, 0.4)
        end = decision + 2 * HOUR
        row_id = f"row-{position:03d}"
        rows.append(
            dict(
                row_id=row_id,
                position=position,
                decision=decision,
                end=end,
                publication=end + (36 * HOUR if position % 5 == 0 else 0),
                y=label,
            )
        )
        revisions.append(
            dict(
                row_id=row_id,
                revision="initial",
                event=decision - HOUR,
                available=decision + (2 * HOUR if position % 7 == 0 else -HOUR),
                value=latent + rng.gauss(0, 0.3),
            )
        )
        revisions.append(
            dict(
                row_id=row_id,
                revision="final",
                event=decision - HOUR,
                available=end + 48 * HOUR,
                value=label / 2,
            )
        )
    # Deliberately emulate a reorder. Identity joins, never positional metadata joins.
    return rows[::2] + rows[1::2], list(reversed(revisions))


def snapshot(revisions, row_id, decision, *, final=False):
    eligible = [
        r
        for r in revisions
        if r["row_id"] == row_id
        and (final or (r["event"] <= decision and r["available"] <= decision))
    ]
    if not eligible:
        return None
    key = max((r["event"], r["available"]) for r in eligible)
    tied = [r for r in eligible if (r["event"], r["available"]) == key]
    if len({r["value"] for r in tied}) != 1:
        raise ValueError("conflicting revisions at selected timestamp")
    return dict(min(tied, key=lambda r: r["revision"]))


def fit_ols(samples):
    if len(samples) < 2:
        raise ValueError("at least two training rows required; no fallback to withheld rows")
    center_x = sum(s["x"] for s in samples) / len(samples)
    center_y = sum(s["y"] for s in samples) / len(samples)
    variance = sum((s["x"] - center_x) ** 2 for s in samples)
    if variance == 0:
        raise ValueError("constant training feature")
    slope = sum((s["x"] - center_x) * (s["y"] - center_y) for s in samples) / variance
    return dict(center_x=center_x, center_y=center_y, slope=slope)


def predict(model, x):
    return model["center_y"] + model["slope"] * (x - model["center_x"])


def run_fold(selector, rows, revisions, fit_position):
    chronological = sorted(rows, key=lambda r: r["position"])
    if len({r["row_id"] for r in rows}) != len(rows):
        raise ValueError("row identity must be unique")
    fit_time = chronological[fit_position]["decision"]
    candidates = [
        i for i, row in enumerate(rows) if row["decision"] < fit_time and row["position"] % 11 != 0
    ]
    pit = [snapshot(revisions, r["row_id"], r["decision"]) for r in rows]
    final = [snapshot(revisions, r["row_id"], r["decision"], final=True) for r in rows]
    safe = selector(
        [r["decision"] for r in rows],
        [r["end"] for r in rows],
        [r["publication"] for r in rows],
        [None if s is None else s["available"] for s in pit],
        fit_time=fit_time,
        clock="unix_ns",
        candidate_indices=candidates,
    )
    tests = chronological[fit_position : fit_position + PROTOCOL["test_rows"]]
    test_pit = {r["row_id"]: snapshot(revisions, r["row_id"], r["decision"]) for r in tests}
    common = [r for r in tests if test_pit[r["row_id"]] is not None]
    if not common:
        raise ValueError("empty common test cohort; no score or imputation")
    arms = {}
    for vintage in ("pit", "final"):
        for maturity in ("published", "resolved"):
            sources = pit if vintage == "pit" else final
            selected = [
                i
                for i in candidates
                if sources[i] is not None
                and rows[i]["end"] < fit_time
                and (maturity == "resolved" or rows[i]["publication"] <= fit_time)
            ]
            if vintage == "pit" and maturity == "published":
                assert tuple(selected) == safe.eligible_indices
            samples = [
                dict(
                    row_id=rows[i]["row_id"],
                    x=sources[i]["value"],
                    y=rows[i]["y"],
                    revision=sources[i]["revision"],
                    source_available=sources[i]["available"],
                    decision=rows[i]["decision"],
                    label_publication=rows[i]["publication"],
                )
                for i in selected
            ]
            # Canonical order keeps floating summation stable across reordered input rows.
            samples.sort(key=lambda s: s["row_id"])
            model = fit_ols(samples)
            scores = {}
            for test_vintage in ("pit", "final"):
                predictions = []
                for row in common:
                    source = snapshot(
                        revisions, row["row_id"], row["decision"], final=test_vintage == "final"
                    )
                    predictions.append(
                        dict(
                            row_id=row["row_id"],
                            decision=row["decision"],
                            source=source,
                            target=row["y"],
                            prediction=predict(model, source["value"]),
                        )
                    )
                scores[test_vintage] = dict(
                    mse=sum((p["prediction"] - p["target"]) ** 2 for p in predictions)
                    / len(predictions),
                    future_source_count=sum(
                        p["source"]["available"] > p["decision"] for p in predictions
                    ),
                    predictions=predictions,
                )
            arms[f"{vintage}_{maturity}"] = dict(
                model=model,
                training=samples,
                training_count=len(samples),
                future_training_source_count=sum(
                    s["source_available"] > s["decision"] for s in samples
                ),
                training_source_after_fit_count=sum(
                    s["source_available"] > fit_time for s in samples
                ),
                unpublished_training_label_count=sum(
                    s["label_publication"] > fit_time for s in samples
                ),
                scores=scores,
            )
    return dict(
        fit_time=fit_time,
        fit_position=fit_position,
        candidate_ids=sorted(rows[i]["row_id"] for i in candidates),
        safe_exclusions={rows[i]["row_id"]: reasons for i, reasons in safe.exclusions},
        test_ids=[r["row_id"] for r in tests],
        common_ids=[r["row_id"] for r in common],
        test_count=len(tests),
        test_available_counts={"pit": len(common), "final": len(tests), "zero": len(tests)},
        missing_pit_test_count=len(tests) - len(common),
        zero_mse=sum(r["y"] ** 2 for r in common) / len(common),
        arms=arms,
    )


def false_clock_control(selector):
    # Actual revision arrives at 12; false old metadata claims 1. The helper only
    # sees the claim and cannot detect replacement of a value without provenance.
    claimed = selector([2], [3], [4], [1], fit_time=10, clock="unix_ns")
    truthful = selector([2], [3], [4], [12], fit_time=10, clock="unix_ns")
    return dict(
        claimed_source_time=1,
        true_source_time=12,
        decision=2,
        claimed_eligible=list(claimed.eligible_indices),
        truthful_eligible=list(truthful.eligible_indices),
        failure="overwritten final revision paired with original source clock",
    )


def run_workflow(selector):
    return dict(
        protocol=PROTOCOL,
        runs=[
            dict(
                seed=seed,
                folds=[
                    run_fold(selector, *generate(seed), pos) for pos in PROTOCOL["fit_positions"]
                ],
            )
            for seed in PROTOCOL["seeds"]
        ],
        false_clock_control=false_clock_control(selector),
        limits="Synthetic outcome-encoded controls, supplied-clock trust, no investment or generalization claim. "
        "Resolved-label arms intentionally ignore publication. Final-vintage arms intentionally bypass historical-source audit. "
        "Final training revisions known at fit can be valid for other target contracts; here they violate the historical-vintage contract. "
        "Each score uses the same common test IDs; missing PIT rows are not imputed. No purging guarantee.",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-tree", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run_workflow(load_selector(args.source_tree)), indent=2))


if __name__ == "__main__":
    main()

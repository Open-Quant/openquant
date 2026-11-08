"""Synthetic repeated-holdout research; no production inference or trading rule.

Training fixes a candidate order. Validation feedback adaptively constructs blends.
A separately simulated audit segment is never passed to the search procedure.
See ADAPTIVE_HOLDOUT_LAB.md for the experimental contract and its limits.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments import validation_lab as lab  # noqa: E402


def _score(values: np.ndarray, weights: np.ndarray) -> float | None:
    """Return undefined for constant blends, retaining that outcome in the report."""
    active = weights != 0
    used = values[:, active]
    scale = np.max(np.abs(used)) if used.size else 0
    series = (used / scale) @ weights[active] if scale else np.zeros(len(values))
    if np.ptp(series) == 0:
        return None
    return float(lab.score_candidates(series[:, None])[0])


def search_validation(training: Any, validation: Any, *, rounds: int) -> dict[str, Any]:
    """Search using training and validation only; audit data is not an argument.

    At each round, retain the incumbent or take a 50:50 positive/negative blend
    with the next training-ranked candidate. Ties retain the first proposal.
    Exact constant proposals cannot win and are counted, never treated as zero.
    """
    train, valid = lab._matrix(training), lab._matrix(validation)
    if train.shape[1] != valid.shape[1]:
        raise ValueError("training and validation must have the same candidate columns")
    rounds = lab._integer(rounds, "rounds", 0)
    if rounds >= train.shape[1]:
        raise ValueError("rounds must be smaller than the number of candidates")
    # Training fixes the order. Score validation columns only as they are queried.
    train_scores = lab.score_candidates(train)
    order = np.argsort(-train_scores, kind="stable")
    valid_scores = {int(order[0]): float(lab.score_candidates(valid[:, [order[0]]])[0])}
    incumbent = np.zeros(train.shape[1])
    incumbent[order[0]] = 1.0
    snapshots = []
    undefined = 0
    best_single = int(order[0])
    for step in range(rounds + 1):
        choice = "initial" if step == 0 else "retain"
        if step:
            candidate = int(order[step])
            valid_scores[candidate] = float(lab.score_candidates(valid[:, [candidate]])[0])
            basis = np.zeros_like(incumbent)
            basis[candidate] = 1.0
            proposals = [incumbent, 0.5 * (incumbent + basis), 0.5 * (incumbent - basis)]
            scores = [_score(valid, weights) for weights in proposals]
            undefined += sum(score is None for score in scores[1:])
            winner = max(range(3), key=lambda i: -np.inf if scores[i] is None else scores[i])
            incumbent = proposals[winner]
            choice = ("retain", "positive_blend", "negative_blend")[winner]
            if valid_scores[candidate] > valid_scores[best_single]:
                best_single = candidate
        snapshots.append(
            {
                "round": step,
                "proposal_choice": choice,
                "adaptive_weights": incumbent.tolist(),
                "adaptive_validation_sharpe": _score(valid, incumbent),
                "single_candidate": best_single,
                "single_validation_sharpe": float(valid_scores[best_single]),
                "undefined_proposals_cumulative": undefined,
                "distinct_candidate_columns_queried": step + 1,
                "adaptive_new_proposal_scores": 1 + 2 * step,
                "single_candidate_scores_used": step + 1,
            }
        )
    return {"training_order": order.tolist(), "snapshots": snapshots}


def evaluate_search(training: Any, validation: Any, audit: Any, *, rounds: int) -> dict[str, Any]:
    """Freeze the complete search path before scoring a separately supplied audit."""
    result = search_validation(training, validation, rounds=rounds)
    # Audit validation happens only after all choices have been frozen.
    audit_values = lab._matrix(audit)
    if audit_values.shape[1] != len(result["training_order"]):
        raise ValueError("audit must have the same candidate columns")
    audit_scores = lab.score_candidates(audit_values)
    frozen = result["training_order"][0]
    result["frozen_training_candidate"] = frozen
    result["frozen_audit_sharpe"] = float(audit_scores[frozen])
    for snapshot in result["snapshots"]:
        score = _score(audit_values, np.array(snapshot["adaptive_weights"]))
        snapshot["adaptive_audit_sharpe"] = score
        snapshot["adaptive_validation_minus_audit"] = (
            snapshot["adaptive_validation_sharpe"] - score if score is not None else None
        )
        snapshot["single_audit_sharpe"] = float(audit_scores[snapshot["single_candidate"]])
    return result


def _summary(values: list[float | None]) -> dict[str, Any]:
    available = [v for v in values if v is not None]
    return {
        "available": len(available),
        "unavailable": len(values) - len(available),
        **(
            lab.summarize_values(available)
            if len(available) >= 2
            else {
                "mean": available[0] if available else None,
                "monte_carlo_standard_error": None,
            }
        ),
    }


def run_lab(
    *,
    repetitions: int = 200,
    sample_size: int = 252,
    n_candidates: int = 32,
    rounds: int = 16,
    seed: int = 7,
    families: tuple[str, ...] = lab.FAMILIES,
) -> dict[str, Any]:
    """Retain paired records and source/input identities for every independent run."""
    repetitions = lab._integer(repetitions, "repetitions", 2)
    sample_size = lab._integer(sample_size, "sample_size", 4)
    n_candidates = lab._integer(n_candidates, "n_candidates", 1)
    rounds = lab._integer(rounds, "rounds", 0)
    seed = lab._integer(seed, "seed", 0)
    if rounds >= n_candidates:
        raise ValueError("rounds must be smaller than n_candidates")
    if (
        not families
        or len(set(families)) != len(families)
        or any(family not in lab.FAMILIES for family in families)
    ):
        raise ValueError("families must be a nonempty unique selection of known families")
    start = time.perf_counter()
    reports = {}
    for family in families:
        records, digest = [], hashlib.sha256()
        family_id = lab.FAMILIES.index(family)
        for repetition in range(repetitions):
            segments = []
            for segment in range(3):
                values = lab.simulate_family(
                    family,
                    sample_size,
                    n_candidates,
                    seed=np.random.SeedSequence([seed, family_id, repetition, segment, 8401]),
                )
                digest.update(np.asarray(values.shape, dtype="<u8").tobytes())
                digest.update(np.asarray(values, dtype="<f8").tobytes())
                segments.append(values)
            record = evaluate_search(*segments, rounds=rounds)
            record["replication"] = repetition
            records.append(record)
        summaries = []
        for step in range(rounds + 1):
            snapshots = [record["snapshots"][step] for record in records]
            row = {"round": step}
            for metric in (
                "adaptive_validation_sharpe",
                "adaptive_audit_sharpe",
                "adaptive_validation_minus_audit",
                "single_validation_sharpe",
                "single_audit_sharpe",
            ):
                row[metric] = _summary([snapshot[metric] for snapshot in snapshots])
            # Paired differences across independent replications, not unpaired SEs.
            row["adaptive_minus_frozen_audit"] = _summary(
                [
                    snapshot["adaptive_audit_sharpe"] - record["frozen_audit_sharpe"]
                    if snapshot["adaptive_audit_sharpe"] is not None
                    else None
                    for snapshot, record in zip(snapshots, records, strict=True)
                ]
            )
            row["adaptive_minus_single_audit"] = _summary(
                [
                    snapshot["adaptive_audit_sharpe"] - snapshot["single_audit_sharpe"]
                    if snapshot["adaptive_audit_sharpe"] is not None
                    else None
                    for snapshot in snapshots
                ]
            )
            summaries.append(row)
        reports[family] = {
            "input_sha256": digest.hexdigest(),
            "records": records,
            "by_round": summaries,
            "frozen_audit_sharpe": _summary([r["frozen_audit_sharpe"] for r in records]),
        }
    try:
        git_base = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.SubprocessError):
        git_base = None
    return {
        "lab": "adaptive_holdout",
        "synthetic_only": True,
        "schema_version": 1,
        "config": {
            "repetitions": repetitions,
            "sample_size": sample_size,
            "n_candidates": n_candidates,
            "rounds": rounds,
            "seed": seed,
            "families": list(families),
        },
        "provenance": {
            "git_base": git_base,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "dependency_sha256": {
                "validation_lab.py": hashlib.sha256(Path(lab.__file__).read_bytes()).hexdigest()
            },
        },
        "performance": {"wall_seconds": time.perf_counter() - start},
        "segment_protocol": "independently simulated training, reused validation, final audit",
        "families": reports,
        "caveats": [
            "All validation scores after feedback are contaminated selection scores.",
            "Audit is independent by simulation construction, not a chronological gap guarantee.",
            "The AR family resets stationary state between independent segments.",
            "An audit reused to choose rounds or methods becomes another validation set.",
            "Monte Carlo standard errors quantify simulation precision, not investment skill.",
            "Undefined mixtures are recorded; summaries condition on availability when needed.",
            "This does not implement a reusable-holdout guarantee, RAS, PBO, PSR or DSR.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--families", nargs="+", choices=lab.FAMILIES, default=lab.FAMILIES)
    for name, default in (
        ("repetitions", 200),
        ("sample-size", 252),
        ("candidates", 32),
        ("rounds", 16),
        ("seed", 7),
    ):
        parser.add_argument("--" + name, type=int, default=default)
    args = parser.parse_args()
    try:
        result = run_lab(
            repetitions=args.repetitions,
            sample_size=args.sample_size,
            n_candidates=args.candidates,
            rounds=args.rounds,
            seed=args.seed,
            families=tuple(args.families),
        )
    except (ValueError, TypeError) as exc:
        parser.error(str(exc))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()

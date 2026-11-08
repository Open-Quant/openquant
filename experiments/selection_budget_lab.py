"""Finite-sum IID sign model: selection versus audit under equal row budgets."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import scipy
from scipy.stats import binom

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments import selection_interval_lab as intervals  # noqa: E402


@lru_cache(maxsize=128)
def _interval_table(n, family_size, alpha):
    bounds = [
        intervals.clopper_pearson(k, n, family_size=family_size, alpha=alpha) for k in range(n + 1)
    ]
    lower = np.array([b["lower"] for b in bounds])
    upper = np.array([b["upper"] for b in bounds])
    lower.flags.writeable = upper.flags.writeable = False
    return lower, upper, tuple(b["status"] for b in bounds)


def winner_masses(n, n_candidates, signal_probability):
    """Joint masses for designated-candidate wins and other-candidate wins.

    Candidate zero wins ties. Its probability is p; all others have probability
    1/2. Candidates and rows are independent, with no zero returns.
    """
    n = intervals._integer(n, "n", 1)
    n_candidates = intervals._integer(n_candidates, "n_candidates", 1)
    if n > intervals.MAX_QUALIFIED_COUNT or n_candidates > 1_000_000:
        raise ValueError("finite-sum supported n and n_candidates are at most 1,000,000")
    if (
        isinstance(signal_probability, (bool, np.bool_, str, bytes))
        or np.iscomplexobj(signal_probability)
        or not np.isscalar(signal_probability)
    ):
        raise TypeError("signal_probability must be a real scalar")
    p = float(signal_probability)
    if not 0 <= p <= 1:
        raise ValueError("signal_probability must be in [0,1]")
    counts = np.arange(n + 1)
    designated = binom.pmf(counts, n, p) * binom.cdf(counts, n, 0.5) ** (n_candidates - 1)
    if n_candidates == 1:
        other = np.zeros(n + 1)
    else:
        cdf = binom.cdf(counts, n, 0.5) ** (n_candidates - 1)
        other = np.diff(np.r_[0.0, cdf]) * binom.cdf(counts - 1, n, p)
    mass = float(designated.sum() + other.sum())
    if abs(mass - 1) > 1e-10 or min(designated.min(), other.min()) < 0:
        raise ArithmeticError("numerical winner masses failed conservation/nonnegativity check")
    return designated, other


def _measure(n, masses, family_size, p, alpha):
    lower, upper, statuses = _interval_table(n, family_size, alpha)
    metrics = dict(
        mean_width=0.0,
        miscoverage=0.0,
        positive_claim=0.0,
        true_positive_claim=0.0,
        null_positive_claim=0.0,
        numerical_vacuity_probability=0.0,
    )
    for target, weights in zip((p, 0.5), masses, strict=True):
        claim = lower > 0.5
        metrics["mean_width"] += float(weights @ (upper - lower))
        metrics["miscoverage"] += float(weights @ ((lower > target) | (upper < target)))
        probability = float(weights @ claim)
        metrics["positive_claim"] += probability
        if target > 0.5:
            metrics["true_positive_claim"] += probability
        elif target <= 0.5:
            metrics["null_positive_claim"] += probability
        metrics["numerical_vacuity_probability"] += float(
            weights @ np.array([s != "ok" for s in statuses])
        )
    return metrics


def evaluate_budget(total_rows, n_candidates, *, signal_probability=0.65, alpha=0.05):
    total_rows = intervals._integer(total_rows, "total_rows", 2)
    if total_rows % 2:
        raise ValueError("total_rows must be even for an equal split")
    n_candidates = intervals._integer(n_candidates, "n_candidates", 1)
    alpha = intervals._probability(alpha)
    full_masses = winner_masses(total_rows, n_candidates, signal_probability)
    p = float(signal_probability)
    half = total_rows // 2
    split_masses = winner_masses(half, n_candidates, p)
    full_selected = float(full_masses[0].sum())
    split_selected = float(split_masses[0].sum())
    counts = np.arange(half + 1)
    audit_masses = (
        split_selected * binom.pmf(counts, half, p),
        float(split_masses[1].sum()) * binom.pmf(counts, half, 0.5),
    )
    full = _measure(total_rows, full_masses, n_candidates, p, alpha)
    split = _measure(half, audit_masses, 1, p, alpha)
    for record, selected in ((full, full_selected), (split, split_selected)):
        record["select_designated_candidate"] = selected
        record["expected_selected_sign_probability"] = 0.5 + (p - 0.5) * selected
    full.update(
        discovery_rows=total_rows, audit_rows=0, candidate_evaluations=total_rows * n_candidates
    )
    split.update(
        discovery_rows=half, audit_rows=half, candidate_evaluations=half * n_candidates + half
    )
    return {
        "total_rows": total_rows,
        "n_candidates": n_candidates,
        "signal_probability": p,
        "full_sample_bonferroni": full,
        "split_holdout": split,
        "numerical_mass_residual": float(full_masses[0].sum() + full_masses[1].sum() - 1),
    }


def run_lab(
    *,
    row_budgets=(32, 64, 128, 256, 512, 1024),
    candidate_counts=(1, 8, 32, 128),
    probabilities=(0.5, 0.55, 0.65, 0.8),
    alpha=0.05,
):
    start = time.perf_counter()
    row_budgets, candidate_counts, probabilities = (
        tuple(row_budgets),
        tuple(candidate_counts),
        tuple(probabilities),
    )
    if not row_budgets or not candidate_counts or not probabilities:
        raise ValueError("all grid axes must be nonempty")
    alpha = intervals._probability(alpha)
    records = [
        evaluate_budget(n, k, signal_probability=p, alpha=alpha)
        for n in row_budgets
        for k in candidate_counts
        for p in probabilities
    ]
    config = {
        "row_budgets": [int(n) for n in row_budgets],
        "candidate_counts": [int(k) for k in candidate_counts],
        "probabilities": [float(p) for p in probabilities],
        "alpha": alpha,
    }
    return {
        "lab": "selection_budget",
        "schema_version": 1,
        "synthetic_only": True,
        "config": config,
        "provenance": {
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "dependency_sha256": {
                "selection_interval_lab.py": hashlib.sha256(
                    Path(intervals.__file__).read_bytes()
                ).hexdigest()
            },
        },
        "performance": {"wall_seconds": time.perf_counter() - start},
        "records": records,
        "caveats": [
            "Finite exact-law sums evaluated numerically, not Monte Carlo or certified arithmetic.",
            "Fixed independent candidates and iid Bernoulli rows; no ties in returns.",
            "Equal observation rows, not equal candidate evaluations or financial costs.",
            "Sign probabilities are not expected returns or investment skill.",
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

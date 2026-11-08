"""Joint candidate and time multiplicity: global discovery versus attribution."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from fractions import Fraction
from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments import sequential_sign_lab as sequential  # noqa: E402

FAMILIES = ("iid_null", "duplicated_null", "correlated_null", "one_signal")
MAX_HORIZON = 256


def candidate_budget(alpha, n_candidates):
    alpha = sequential._probability(alpha, "alpha")
    n_candidates = sequential._integer(n_candidates, "n_candidates")
    exact = Fraction.from_float(alpha) / n_candidates
    budget = float(exact)
    if Fraction.from_float(budget) > exact:
        budget = float(np.nextafter(budget, 0.0))
    return budget


def capital_table(horizon, qs=(0.55, 0.65, 0.8)):
    """Exact integer numerator table for a common dyadic mixture denominator."""
    horizon = sequential._integer(horizon, "horizon")
    if horizon > MAX_HORIZON:
        raise ValueError("global-capital table horizon exceeds 256")
    qs = sequential._mixture(qs)
    return _capital_table(horizon, qs)


@lru_cache(maxsize=4)
def _capital_table(horizon, qs):
    ratios = [q.as_integer_ratio() for q in qs]
    exponent = max(d.bit_length() - 2 for _, d in ratios)
    rows = []
    for t in range(horizon + 1):
        row = []
        for s in range(t + 1):
            row.append(
                sum(
                    (pow(a, s) * pow(d - a, t - s)) << ((exponent - (d.bit_length() - 2)) * t)
                    for a, d in ratios
                )
            )
        rows.append(tuple(row))
    return tuple(rows), exponent, len(qs)


def independent_any_crossing(single_probability, n_candidates):
    p = sequential._probability(single_probability, "single_probability", closed=True)
    n_candidates = sequential._integer(n_candidates, "n_candidates")
    return 1.0 if p == 1 else -math.expm1(n_candidates * math.log1p(-p))


def evaluate_paths(paths, *, alpha=0.05, qs=(0.55, 0.65, 0.8)):
    paths = np.asarray(paths)
    if (
        paths.ndim != 3
        or not all(paths.shape)
        or paths.dtype.kind not in "iub"
        or np.any((paths != 0) & (paths != 1))
    ):
        raise ValueError("paths must be binary repetitions-by-time-by-candidates")
    repetitions, horizon, k = paths.shape
    if horizon > MAX_HORIZON:
        raise ValueError("global-capital table horizon exceeds 256")
    alpha = sequential._probability(alpha, "alpha")
    qs = sequential._mixture(qs)
    budget = candidate_budget(alpha, k)
    uncorrected = sequential.crossing_boundaries(horizon, alpha=alpha, qs=qs)["fixed_mixture"]
    corrected = (
        sequential.crossing_boundaries(horizon, alpha=budget, qs=qs)["fixed_mixture"]
        if budget
        else np.arange(horizon + 1) + 1
    )
    table, exponent, components = capital_table(horizon, qs)
    numerator, denominator = alpha.as_integer_ratio()
    counts = np.zeros((repetitions, k), dtype=np.int64)
    first_naive = np.zeros_like(counts)
    first_corrected = np.zeros_like(counts)
    global_first = np.zeros(repetitions, dtype=np.int64)
    leader = np.full(repetitions, -1, dtype=np.int64)
    for t in range(1, horizon + 1):
        counts += paths[:, t - 1, :]
        first_naive[(first_naive == 0) & (counts >= uncorrected[t])] = t
        first_corrected[(first_corrected == 0) & (counts >= corrected[t])] = t
        # Average cannot exceed maximum: this exact count gate skips only
        # mathematically impossible crossings, never approximate log decisions.
        eligible = np.flatnonzero((global_first == 0) & (counts.max(axis=1) >= uncorrected[t]))
        rhs = denominator * k * components * (1 << (exponent * t))
        cutoff = (rhs + numerator - 1) // numerator
        for r in eligible:
            if sum(table[t][int(s)] for s in counts[r]) >= cutoff:
                global_first[r] = t
                leader[r] = int(np.argmax(counts[r]))
    return {
        "uncorrected_first": first_naive,
        "bonferroni_first": first_corrected,
        "global_first": global_first,
        "global_leader": leader,
        "candidate_alpha": budget,
        "candidate_budget_underflow": budget == 0,
        "individual_boundary": uncorrected,
        "corrected_boundary": corrected,
    }


def _event(values):
    values = np.asarray(values, dtype=bool)
    rate = float(values.mean())
    return {
        "rate": rate,
        "mcse": math.sqrt(rate * (1 - rate) / len(values)),
        "count": int(values.sum()),
    }


def run_lab(
    *,
    horizon=64,
    n_candidates=8,
    repetitions=1000,
    alpha=0.05,
    seed=19,
    qs=(0.55, 0.65, 0.8),
    families=FAMILIES,
):
    start = time.perf_counter()
    horizon = sequential._integer(horizon, "horizon")
    if horizon > MAX_HORIZON:
        raise ValueError("horizon exceeds 256")
    n_candidates = sequential._integer(n_candidates, "n_candidates")
    repetitions = sequential._integer(repetitions, "repetitions")
    seed = sequential._integer(seed, "seed", 0)
    alpha = sequential._probability(alpha, "alpha")
    qs = sequential._mixture(qs)
    families = tuple(families)
    if (
        not families
        or len(set(families)) != len(families)
        or any(f not in FAMILIES for f in families)
    ):
        raise ValueError("families must be unique supported names")
    reports = {}
    for family in families:
        rng = np.random.default_rng(
            np.random.SeedSequence([seed, FAMILIES.index(family), horizon, n_candidates])
        )
        p = np.full(n_candidates, 0.5)
        if family == "one_signal":
            p[0] = 0.65
        paths = (rng.random((repetitions, horizon, n_candidates)) < p).astype(np.uint8)
        if family == "duplicated_null":
            paths[:] = paths[:, :, :1]
        elif family == "correlated_null":
            common = rng.random((repetitions, horizon, 1)) < 0.5
            paths = np.where(rng.random(paths.shape) < 0.75, common, paths).astype(np.uint8)
        result = evaluate_paths(paths, alpha=alpha, qs=qs)
        true_null = p <= 0.5
        methods = {}
        for label, key in [
            ("uncorrected", "uncorrected_first"),
            ("bonferroni", "bonferroni_first"),
        ]:
            crossings = result[key] > 0
            methods[label] = {
                "any_candidate_claim": _event(crossings.any(axis=1)),
                "any_false_candidate_claim": _event(crossings[:, true_null].any(axis=1)),
                "any_true_candidate_claim": _event(crossings[:, ~true_null].any(axis=1)),
                "first_crossing_rows": result[key].tolist(),
            }
        global_cross = result["global_first"] > 0
        null_leader = global_cross & (true_null[np.maximum(result["global_leader"], 0)])
        methods["global_average"] = {
            "global_discovery": _event(global_cross),
            "crossed_with_true_null_leader_descriptive": _event(null_leader),
            "null_leader_fraction_given_crossing_descriptive": float(
                null_leader.sum() / global_cross.sum()
            )
            if global_cross.any()
            else None,
            "first_crossing_rows": result["global_first"].tolist(),
            "leader_indices_descriptive": result["global_leader"].tolist(),
        }
        oracle = {}
        if family in ("iid_null", "duplicated_null"):
            effective = n_candidates if family == "iid_null" else 1
            for method, boundary in [
                ("uncorrected", result["individual_boundary"]),
                ("bonferroni", result["corrected_boundary"]),
            ]:
                single = sequential.crossing_probability(boundary)["crossing_probability"]
                oracle[method] = {
                    "single_stream_probability": single,
                    "any_probability": independent_any_crossing(single, effective),
                    "independent_unique_streams": effective,
                }
        reports[family] = {
            "global_null": bool(true_null.all()),
            "null_candidate_joint_filtration_premise": True,
            "sample_sha256": hashlib.sha256(
                json.dumps({"shape": list(paths.shape), "dtype": "uint8"}, sort_keys=True).encode()
                + paths.tobytes()
            ).hexdigest(),
            "candidate_alpha": result["candidate_alpha"],
            "candidate_budget_underflow": result["candidate_budget_underflow"],
            "methods": methods,
            "independent_reference": oracle,
        }
    return {
        "lab": "multiple_sequential",
        "schema_version": 1,
        "synthetic_only": True,
        "config": {
            "horizon": horizon,
            "n_candidates": n_candidates,
            "repetitions": repetitions,
            "alpha": alpha,
            "seed": seed,
            "qs": list(qs),
            "families": list(families),
        },
        "provenance": {
            "numpy": np.__version__,
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "dependency_sha256": {
                "sequential_sign_lab.py": hashlib.sha256(
                    Path(sequential.__file__).read_bytes()
                ).hexdigest()
            },
        },
        "performance": {"wall_seconds": time.perf_counter() - start},
        "families": reports,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--horizon", type=int, default=64)
    parser.add_argument("--n-candidates", type=int, default=8)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = run_lab(horizon=args.horizon, n_candidates=args.n_candidates)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()

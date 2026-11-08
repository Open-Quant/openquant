"""Oracle weighting and nonidentification under outcome-dependent missingness."""

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np

MODELS = (
    ("mcar", 0.5, 0.5, 0.5),
    ("informative_A", 0.5, 1.0, 0.25),
    ("equivalent_B", 0.625, 0.8, 1 / 3),
    ("near_zero", 0.5, 1.0, 0.01),
    ("zero_propensity", 0.5, 1.0, 0.0),
    ("all_missing", 0.5, 0.0, 0.0),
)


def probability(value):
    if (
        isinstance(value, (bool, np.bool_, str, bytes))
        or np.iscomplexobj(value)
        or not np.isscalar(value)
    ):
        raise TypeError("probability must be real")
    value = float(value)
    if not 0 <= value <= 1:
        raise ValueError("probability outside[0,1]")
    return value


def reference(p, splus, sminus, n):
    p, splus, sminus = map(probability, (p, splus, sminus))
    if isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)) or n < 1:
        raise ValueError("n must be positive integer")
    n = int(n)
    if n > 1_000_000 or any(0 < s < 1e-12 for s in (splus, sminus)):
        raise ValueError("qualified n<=1e6 and positive propensities>=1e-12")
    a = p * splus
    b = (1 - p) * sminus
    mu = 2 * p - 1
    positive = splus > 0 and sminus > 0
    return {
        "true_mean": mu,
        "observable_probabilities": [a, b, 1 - a - b],
        "observed_target": (a - b) / (a + b) if a + b else None,
        "no_observation_probability": (1 - a - b) ** n,
        "ht_available": positive,
        "ht_expectation": mu if positive else None,
        "ht_variance": (p / splus + (1 - p) / sminus - mu * mu) / n if positive else None,
        "full_mean_variance": 4 * p * (1 - p) / n,
    }


def estimate_counts(positive, negative, n, splus, sminus):
    for value in (positive, negative, n):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise TypeError("counts must be integers")
    positive, negative, n = int(positive), int(negative), int(n)
    if n < 1 or min(positive, negative) < 0 or positive + negative > n:
        raise ValueError("invalid counts")
    splus, sminus = map(probability, (splus, sminus))
    if n > 1_000_000 or any(0 < s < 1e-12 for s in (splus, sminus)):
        raise ValueError("qualified n<=1e6 and positive propensities>=1e-12")
    observed = positive + negative
    if (splus == 0 and positive) or (sminus == 0 and negative):
        raise ValueError("observed impossible category")
    naive = (positive - negative) / observed if observed else None
    if splus == 0 or sminus == 0:
        return {"observed_only": naive, "ht": None, "hajek": None}
    up = positive / splus
    down = negative / sminus
    if not math.isfinite(up + down):
        raise ValueError("weights outside finite range")
    return {
        "observed_only": naive,
        "ht": (up - down) / n,
        "hajek": (up - down) / (up + down) if up + down else None,
    }


def run_lab(*, sample_sizes=(32, 128, 512), repetitions=5000, seed=79):
    start = time.perf_counter()
    sample_sizes = tuple(sample_sizes)
    for val in (repetitions, seed, *sample_sizes):
        if isinstance(val, (bool, np.bool_)) or not isinstance(val, (int, np.integer)) or val < 0:
            raise ValueError("integer parameters required")
    if repetitions < 2 or not sample_sizes or min(sample_sizes) < 1:
        raise ValueError("nonempty positive grid and repetitions>=2 required")
    repetitions, seed = int(repetitions), int(seed)
    sample_sizes = tuple(int(n) for n in sample_sizes)
    records = []
    for n in sample_sizes:
        for index, (name, p, sp, sm) in enumerate(MODELS):
            rng = np.random.default_rng(np.random.SeedSequence([seed, n, index]))
            truth = reference(p, sp, sm, n)
            # Four categories preserve missing outcomes for the full-data control only.
            probs = [p * sp, (1 - p) * sm, p * (1 - sp), (1 - p) * (1 - sm)]
            counts = rng.multinomial(n, probs, size=repetitions)
            rows = []
            for positive, negative, missing_positive, missing_negative in counts:
                estimates = estimate_counts(int(positive), int(negative), n, sp, sm)
                estimates["full_data"] = (
                    positive + missing_positive - negative - missing_negative
                ) / n
                rows.append(
                    {
                        "observed_positive": int(positive),
                        "observed_negative": int(negative),
                        "missing": int(missing_positive + missing_negative),
                        "estimates": estimates,
                    }
                )
            summary = {}
            for method in ["observed_only", "ht", "hajek", "full_data"]:
                values = np.array(
                    [r["estimates"][method] for r in rows if r["estimates"][method] is not None]
                )
                size = len(values)
                summary[method] = {
                    "available": size,
                    "unavailable": repetitions - size,
                    "conditional_on_available": size != repetitions,
                    "mean": float(values.mean()) if size else None,
                    "bias_vs_population_mean": float(values.mean() - truth["true_mean"])
                    if size
                    else None,
                    "variance": float(values.var(ddof=1)) if size > 1 else None,
                    "mean_mcse": float(values.std(ddof=1) / math.sqrt(size)) if size > 1 else None,
                    "outside_outcome_range_frequency": float(np.mean(abs(values) > 1))
                    if size
                    else None,
                }
            records.append(
                {
                    "n": int(n),
                    "model": name,
                    "p": p,
                    "splus": sp,
                    "sminus": sm,
                    "reference": truth,
                    "counts_sha256": hashlib.sha256(
                        np.asarray(counts, dtype="<i8").tobytes()
                    ).hexdigest(),
                    "summary": summary,
                    "rows": rows,
                }
            )
    return {
        "lab": "informative_missingness",
        "schema_version": 1,
        "synthetic_only": True,
        "config": {"sample_sizes": list(sample_sizes), "repetitions": repetitions, "seed": seed},
        "provenance": {
            "numpy": np.__version__,
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
        "performance": {"wall_seconds": time.perf_counter() - start},
        "records": records,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    r = run_lab()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(r, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()

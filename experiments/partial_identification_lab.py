"""Sharp binary-outcome missing-data bounds and conservative outer confidence sets."""

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import scipy

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments import selection_interval_lab as cp  # noqa: E402

MODELS = (
    ("equivalent_A", 0.5, 0.125, 0.0),
    ("equivalent_B", 0.5, 0.125, 0.125),
    ("mcar", 0.25, 0.25, 0.25),
    ("fully_observed", 0.5, 0.5, 0.0),
    ("all_missing", 0.0, 0.0, 0.5),
)


def compatible_model(a, b, missing_positive):
    vals = []
    for x in (a, b, missing_positive):
        if isinstance(x, (bool, np.bool_, str, bytes)) or np.iscomplexobj(x) or not np.isscalar(x):
            raise TypeError("masses must be real")
        x = float(x)
        if not math.isfinite(x) or x < 0:
            raise ValueError("masses must be finite nonnegative")
        vals.append(x)
    a, b, z = vals
    c = 1 - a - b
    if c < 0 or z > c:
        raise ValueError("incompatible probability masses")
    p = a + z
    return {
        "observable": [a, b, c],
        "missing_positive": z,
        "population_positive": p,
        "mean": 2 * p - 1,
        "sharp_bounds": [2 * a - 1, 1 - 2 * b],
        "structural_width": 2 * c,
        "splus": a / p if p else None,
        "sminus": b / (1 - p) if p < 1 else None,
    }


def outer_interval(positive, negative, n, *, alpha=0.05):
    n = cp._integer(n, "n", 1)
    positive = cp._integer(positive, "positive")
    negative = cp._integer(negative, "negative")
    if positive + negative > n:
        raise ValueError("observed counts exceed full n")
    alpha = cp._probability(alpha)
    left = cp.clopper_pearson(positive, n, alpha=alpha)
    right = cp.clopper_pearson(negative, n, alpha=alpha)
    return {
        "plugin_bounds": [2 * positive / n - 1, 1 - 2 * negative / n],
        "outer_bounds": [
            max(-1.0, math.nextafter(2 * left["lower"] - 1, -math.inf)),
            min(1.0, math.nextafter(1 - 2 * right["lower"], math.inf)),
        ],
        "observed_mean": (positive - negative) / (positive + negative)
        if positive + negative
        else None,
        "cp_statuses": [left["status"], right["status"]],
    }


def run_lab(*, sample_sizes=(32, 128, 512, 4096), repetitions=2000, alpha=0.05, seed=113):
    start = time.perf_counter()
    sample_sizes = tuple(cp._integer(n, "n", 1) for n in sample_sizes)
    repetitions = cp._integer(repetitions, "repetitions", 2)
    seed = cp._integer(seed, "seed")
    alpha = cp._probability(alpha)
    if not sample_sizes or max(sample_sizes) > 100000:
        raise ValueError("nonempty grid n<=100000 required")
    reports = []
    for n in sample_sizes:
        for index, (name, a, b, z) in enumerate(MODELS):
            truth = compatible_model(a, b, z)
            # Equivalent A/B intentionally use exactly the same observed counts.
            stream = 0 if name.startswith("equivalent") else index
            rng = np.random.default_rng(np.random.SeedSequence([seed, n, stream]))
            counts = rng.multinomial(n, truth["observable"], size=repetitions)
            cache = {}
            records = []
            for kp, km, missing in counts:
                key = (int(kp), int(km))
                if key not in cache:
                    cache[key] = outer_interval(*key, n, alpha=alpha)
                records.append(
                    {
                        "observed_positive": int(kp),
                        "observed_negative": int(km),
                        "missing": int(missing),
                        **cache[key],
                    }
                )
            summary = {}
            for method in ["plugin_bounds", "outer_bounds"]:
                bounds = np.array([r[method] for r in records])
                cover = (bounds[:, 0] <= truth["mean"]) & (truth["mean"] <= bounds[:, 1])
                setcover = (bounds[:, 0] <= truth["sharp_bounds"][0]) & (
                    truth["sharp_bounds"][1] <= bounds[:, 1]
                )
                summary[method] = {
                    "mean_width": float(np.mean(bounds[:, 1] - bounds[:, 0])),
                    "mean_lower": float(bounds[:, 0].mean()),
                    "mean_upper": float(bounds[:, 1].mean()),
                    "population_mean_coverage": float(cover.mean()),
                    "identified_set_coverage": float(setcover.mean()),
                    "identified_set_coverage_mcse": float(
                        setcover.std(ddof=1) / math.sqrt(repetitions)
                    ),
                }
            observed = [r["observed_mean"] for r in records if r["observed_mean"] is not None]
            reports.append(
                {
                    "n": n,
                    "model": name,
                    "truth": truth,
                    "observed_mean_summary": {
                        "available": len(observed),
                        "mean_conditional_on_available": float(np.mean(observed))
                        if observed
                        else None,
                    },
                    "counts_sha256": hashlib.sha256(
                        np.asarray(counts, dtype="<i8").tobytes()
                    ).hexdigest(),
                    "summary": summary,
                    "records": records,
                }
            )
    return {
        "lab": "partial_identification",
        "schema_version": 1,
        "synthetic_only": True,
        "config": {
            "sample_sizes": list(sample_sizes),
            "repetitions": repetitions,
            "alpha": alpha,
            "seed": seed,
        },
        "provenance": {
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "dependency_sha256": {
                "selection_interval_lab.py": hashlib.sha256(
                    Path(cp.__file__).read_bytes()
                ).hexdigest()
            },
        },
        "performance": {"wall_seconds": time.perf_counter() - start},
        "reports": reports,
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

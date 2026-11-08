"""Finite Bernoulli-law uncertainty when an empirical sample misses rare events."""

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import scipy
from scipy.stats import binom, norm, t

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments import selection_interval_lab as cp  # noqa: E402


def evaluate(n, p, *, alpha=0.05):
    n = cp._integer(n, "n", 2)
    p = cp._probability(p)
    alpha = cp._probability(alpha)
    if n > 10000 or not 1e-8 <= p <= 1 - 1e-8 or not 1e-8 <= alpha <= 0.5:
        raise ValueError("qualified range: n<=10000,p in[1e-8,1-1e-8],alpha in[1e-8,.5]")
    a = 1 / p
    counts = np.arange(n + 1)
    phat = counts / n
    means = a * phat
    masses = binom.pmf(counts, n, p)
    if abs(float(masses.sum()) - 1) > 1e-10:
        raise ArithmeticError("mass conservation failed")
    se = a * np.sqrt(phat * (1 - phat) / (n - 1))
    known = math.sqrt(a * a * p * (1 - p) / n / alpha)
    intervals = {
        "known_variance_chebyshev": np.column_stack((means - known, means + known)),
        "plugin_chebyshev": np.column_stack(
            (means - se / math.sqrt(alpha), means + se / math.sqrt(alpha))
        ),
        "gaussian_reference": np.column_stack(
            (means - norm.isf(alpha / 2) * se, means + norm.isf(alpha / 2) * se)
        ),
        "student_reference": np.column_stack(
            (means - t.isf(alpha / 2, n - 1) * se, means + t.isf(alpha / 2, n - 1) * se)
        ),
        "percentile_bootstrap": a
        / n
        * np.column_stack((binom.ppf(alpha / 2, n, phat), binom.ppf(1 - alpha / 2, n, phat))),
    }
    bounds = [cp.clopper_pearson(int(k), n, alpha=alpha) for k in counts]
    intervals["known_support_cp"] = a * np.array([[b["lower"], b["upper"]] for b in bounds])
    summaries = {}
    for method, limits in intervals.items():
        widths = limits[:, 1] - limits[:, 0]
        if not np.isfinite(limits).all() or np.any(widths < 0):
            raise ArithmeticError("invalid intervals")
        covered = (limits[:, 0] <= 1) & (1 <= limits[:, 1])
        summaries[method] = {
            "coverage": float(masses @ covered),
            "expected_width": float(masses @ widths),
            "zero_width_probability": float(masses @ (widths == 0)),
            "zero_event_interval": limits[0].tolist(),
        }
    return {
        "n": n,
        "p": p,
        "amplitude": a,
        "true_mean": 1.0,
        "true_variance": a * a * p * (1 - p),
        "no_event_probability": float(binom.pmf(0, n, p)),
        "log_no_event_probability": n * math.log1p(-p),
        "no_event_probability_underflow": bool(binom.pmf(0, n, p) == 0),
        "summaries": summaries,
        "counts": counts.tolist(),
        "count_probabilities": masses.tolist(),
        "intervals": {m: v.tolist() for m, v in intervals.items()},
        "cp_statuses": [b["status"] for b in bounds],
    }


def run_lab(
    *, sample_sizes=(32, 128, 512, 2048), probabilities=(0.001, 0.01, 0.05, 0.5), alpha=0.05
):
    start = time.perf_counter()
    sample_sizes = tuple(sample_sizes)
    probabilities = tuple(probabilities)
    if not sample_sizes or not probabilities:
        raise ValueError("grid axes must be nonempty")
    records = [evaluate(n, p, alpha=alpha) for n in sample_sizes for p in probabilities]
    return {
        "lab": "rare_event_uncertainty",
        "schema_version": 1,
        "synthetic_only": True,
        "config": {
            "sample_sizes": list(sample_sizes),
            "probabilities": list(probabilities),
            "alpha": alpha,
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

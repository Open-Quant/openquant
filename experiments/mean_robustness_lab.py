"""Finite-variance mean intervals with explicit variance and IID assumptions."""

import argparse
import hashlib
import json
import math
import time
from functools import lru_cache
from pathlib import Path
from statistics import NormalDist

import numpy as np

FAMILIES = ("gaussian", "student_t3", "rare_skew", "repeated_rows", "wrong_variance")


@lru_cache(maxsize=64)
def block_count(alpha):
    if (
        isinstance(alpha, (bool, np.bool_, str, bytes))
        or not np.isscalar(alpha)
        or np.iscomplexobj(alpha)
    ):
        raise TypeError("alpha must be real")
    alpha = float(alpha)
    if not 1e-12 <= alpha < 1:
        raise ValueError("supported alpha is [1e-12,1)")
    num, den = alpha.as_integer_ratio()
    for b in range(1, 10001, 2):
        tail = sum(math.comb(b, j) * 3 ** (b - j) for j in range((b + 1) // 2, b + 1))
        if tail * den <= num * 4**b:
            return b, tail, 4**b
    raise ValueError("alpha outside supported block allocation")


def intervals(values, *, alpha=0.05, variance_bound=1.0):
    x = np.asarray(values)
    if x.ndim != 1 or len(x) < 2 or x.dtype.kind not in "iuf" or not np.isfinite(x).all():
        raise ValueError("values must be a finite real vector with at least two rows")
    if isinstance(variance_bound, (bool, np.bool_, str, bytes)) or np.iscomplexobj(variance_bound):
        raise TypeError("variance_bound must be real")
    v = float(variance_bound)
    if not math.isfinite(v) or not 1e-200 <= v <= 1e200:
        raise ValueError("supported variance_bound is [1e-200,1e200]")
    b, tail, denominator = block_count(alpha)
    alpha = float(alpha)
    n = len(x)
    mean = float(np.mean(x))
    m = n // b

    def pair(center, radius):
        if not math.isfinite(center) or not math.isfinite(radius):
            raise ValueError("interval arithmetic outside finite range")
        bounds = [center - radius, center + radius]
        if not all(math.isfinite(value) for value in bounds):
            raise ValueError("interval endpoints outside finite range")
        return bounds

    return {
        "chebyshev": pair(mean, math.sqrt(v / n) / math.sqrt(alpha)),
        "gaussian_reference": pair(mean, -NormalDist().inv_cdf(alpha / 2) * math.sqrt(v / n)),
        "median_of_means": pair(
            float(np.median(x[: b * m].reshape(b, m).mean(axis=1))), 2 * math.sqrt(v / m)
        )
        if m
        else None,
        "clipped_mean": pair(
            float(np.clip(x, -2, 2).mean()), 4 * math.sqrt(math.log(2 / alpha) / (2 * n))
        ),
        "blocks": b,
        "block_size": m,
        "unused_rows": n - b * m,
        "block_failure_bound": tail / denominator,
    }


def run_lab(*, sample_size=256, repetitions=5000, alpha=0.05, seed=41, families=FAMILIES):
    start = time.perf_counter()
    for name, value, minimum in [
        ("sample_size", sample_size, 2),
        ("repetitions", repetitions, 2),
        ("seed", seed, 0),
    ]:
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or value < minimum
        ):
            raise ValueError(name + " must be an integer in range")
    families = tuple(families)
    if (
        not families
        or len(set(families)) != len(families)
        or any(f not in FAMILIES for f in families)
    ):
        raise ValueError("families must be unique supported names")
    block_count(alpha)
    reports = {}
    for family in families:
        rng = np.random.default_rng(
            np.random.SeedSequence([seed, FAMILIES.index(family), sample_size])
        )
        shape = (repetitions, sample_size)
        if family == "student_t3":
            x = rng.standard_t(3, size=shape) / math.sqrt(3)
        elif family == "rare_skew":
            x = (rng.binomial(1, 0.01, size=shape) - 0.01) / math.sqrt(0.01 * 0.99)
        elif family == "repeated_rows":
            x = np.repeat(rng.normal(size=(repetitions, 1)), sample_size, axis=1)
        else:
            x = rng.normal(size=shape) * (2 if family == "wrong_variance" else 1)
        clipped_target = (
            0.01 * 2 - 0.99 * 0.01 / math.sqrt(0.01 * 0.99) if family == "rare_skew" else 0.0
        )
        records = [intervals(row, alpha=alpha) for row in x]
        summary = {}
        for method in ["chebyshev", "gaussian_reference", "median_of_means", "clipped_mean"]:
            bounds = [r[method] for r in records if r[method] is not None]

            def coverage(target):
                if not bounds:
                    return None
                rate = float(np.mean([lo <= target <= hi for lo, hi in bounds]))
                return {"rate": rate, "mcse": math.sqrt(rate * (1 - rate) / len(bounds))}

            summary[method] = {
                "available": len(bounds),
                "coverage_original_mean": coverage(0.0),
                "coverage_own_target": coverage(
                    clipped_target if method == "clipped_mean" else 0.0
                ),
                "mean_width": float(np.mean([hi - lo for lo, hi in bounds])) if bounds else None,
            }
        reports[family] = {
            "iid": family != "repeated_rows",
            "variance_bound_valid": family != "wrong_variance",
            "true_variance": 4.0 if family == "wrong_variance" else 1.0,
            "true_mean": 0.0,
            "clipped_target": clipped_target,
            "input_sha256": hashlib.sha256(np.asarray(x, dtype="<f8").tobytes()).hexdigest(),
            "summary": summary,
            "records": records,
        }
    return {
        "lab": "mean_robustness",
        "schema_version": 1,
        "synthetic_only": True,
        "config": {
            "sample_size": int(sample_size),
            "repetitions": int(repetitions),
            "alpha": float(alpha),
            "seed": int(seed),
            "families": list(families),
        },
        "provenance": {
            "numpy": np.__version__,
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
        "performance": {"wall_seconds": time.perf_counter() - start},
        "families": reports,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sample-size", type=int, default=256)
    p.add_argument("--repetitions", type=int, default=5000)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    report = run_lab(sample_size=a.sample_size, repetitions=a.repetitions, alpha=a.alpha)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()

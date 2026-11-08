"""Numerically qualified origin membership versus declared analytic probabilities."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import scipy
from scipy.linalg import helmert
from scipy.optimize import linprog


def wendel(p, d):
    if (
        isinstance(p, bool)
        or isinstance(d, bool)
        or not isinstance(p, int)
        or not isinstance(d, int)
        or p < 1
        or d < 1
    ):
        raise ValueError("positive integer dimensions required")
    return 1 - sum(math.comb(p - 1, k) for k in range(min(d, p))) / 2 ** (p - 1)


def checked_lp(size, *args, **kwargs):
    """Malformed, failed or nonfinite solver results are unavailable."""
    try:
        result = linprog(*args, **kwargs)
        value = np.asarray(getattr(result, "x", None))
        if (
            getattr(result, "success", False)
            and value.shape == (size,)
            and value.dtype.kind in "fiu"
            and np.isfinite(value).all()
        ):
            return value.astype(float)
    except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError):
        pass
    return None


def classify(columns):
    """Return checked primal membership or checked strict separation, else unresolved."""
    raw = np.asarray(columns)
    if (
        raw.ndim != 2
        or min(raw.shape) < 1
        or raw.dtype.kind not in "fiu"
        or not np.isfinite(raw).all()
    ):
        raise ValueError("finite real nonempty matrix required")
    a = raw.astype(float)
    scales = np.max(np.abs(a), axis=0)
    a = a / np.where(scales > 0, scales, 1)
    d, p = a.shape
    options = {"primal_feasibility_tolerance": 1e-9, "dual_feasibility_tolerance": 1e-9}
    primal = checked_lp(
        p,
        np.zeros(p),
        A_eq=np.vstack([np.ones(p), a]),
        b_eq=np.r_[1.0, np.zeros(d)],
        bounds=(0, None),
        method="highs",
        options=options,
    )
    if primal is not None:
        w = primal
        residual = max(abs(w.sum() - 1), float(np.max(np.abs(a @ w))), max(0.0, -float(w.min())))
        if residual <= 1e-8:
            return {
                "status": "member",
                "residual": float(residual),
                "weights": w.tolist(),
                "separator": None,
                "margin": None,
            }
    # Maximize margin m with ||v||_infinity <=1 and a_j'v >=m.
    separation = checked_lp(
        d + 1,
        np.r_[np.zeros(d), -1.0],
        A_ub=np.column_stack([-a.T, np.ones(p)]),
        b_ub=np.zeros(p),
        bounds=[(-1, 1)] * d + [(None, None)],
        method="highs",
        options=options,
    )
    if separation is not None:
        v = separation[:d]
        margin = float(np.min(a.T @ v))
        if np.max(np.abs(v)) <= 1 + 1e-8 and margin > 1e-8:
            return {
                "status": "separated",
                "residual": None,
                "weights": None,
                "separator": v.tolist(),
                "margin": margin,
            }
    return {
        "status": "unresolved",
        "residual": None,
        "weights": None,
        "separator": None,
        "margin": None,
    }


def run_lab(*, repetitions=2000, seed=733):
    if isinstance(repetitions, bool) or not isinstance(repetitions, int) or repetitions < 1:
        raise ValueError("positive integer repetitions required")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("nonnegative integer seed required")
    started = time.perf_counter()
    cells = []
    designs = [
        (dist, t, 8, 0.0)
        for dist in ("gaussian", "independent_t5", "common_row_t5")
        for t in (2, 4, 8, 9)
    ] + [("correlated_gaussian", 2, 2, rho) for rho in (-0.75, 0.0, 0.75)]
    for index, (dist, t, p, rho) in enumerate(designs):
        rng = np.random.default_rng(np.random.SeedSequence([seed, index]))
        records = []
        for _ in range(repetitions):
            x = rng.normal(size=(t, p))
            if dist == "independent_t5":
                x *= np.sqrt(3 / rng.chisquare(5, size=(t, p)))
            elif dist == "common_row_t5":
                x *= np.sqrt(3 / rng.chisquare(5, size=(t, 1)))
            elif dist == "correlated_gaussian":
                x = x @ np.linalg.cholesky([[1, rho], [rho, 1]]).T
            a = helmert(t) @ x
            result = classify(a)
            result["columns"] = a.tolist()
            records.append(result)
        counts = {
            status: sum(r["status"] == status for r in records)
            for status in ("member", "separated", "unresolved")
        }
        target = math.acos(rho) / math.pi if dist == "correlated_gaussian" else wendel(p, t - 1)
        cells.append(
            {
                "distribution": dist,
                "train_size": t,
                "assets": p,
                "rho": rho,
                "analytic_probability": target,
                "counts": counts,
                "total": repetitions,
                "membership_bracket": [
                    counts["member"] / repetitions,
                    (counts["member"] + counts["unresolved"]) / repetitions,
                ],
                "records": records,
            }
        )
    return {
        "lab": "zero_fit_probability",
        "schema_version": 1,
        "synthetic_only": True,
        "config": {"repetitions": repetitions, "seed": seed},
        "cells": cells,
        "provenance": {
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "dependency_sha256": {},
        },
        "performance": {"wall_seconds": time.perf_counter() - started},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repetitions", type=int, default=2000)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(run_lab(repetitions=args.repetitions), indent=2, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    main()

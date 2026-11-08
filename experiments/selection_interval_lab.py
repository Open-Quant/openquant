"""Selection-aware exact-binomial intervals; optional synthetic research utility."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from fractions import Fraction
from pathlib import Path

import numpy as np
import scipy
from scipy.special import betainccinv, betaincinv
from scipy.stats import binom

MAX_QUALIFIED_COUNT = 1_000_000

FAMILIES = (
    "iid_null",
    "duplicated",
    "correlated",
    "heterogeneous_ties",
    "signal",
    "repeated_row",
    "hidden_family",
    "all_ties",
)


def _integer(value, name, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    value = int(value)
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


def _probability(alpha):
    if (
        isinstance(alpha, (bool, np.bool_, str, bytes))
        or np.iscomplexobj(alpha)
        or not np.isscalar(alpha)
    ):
        raise TypeError("alpha must be a real scalar")
    alpha = float(alpha)
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0,1)")
    return alpha


def allocated_tail(alpha=0.05, family_size=1):
    """Round the exact binary-float alpha/(2K) downward, including underflow."""
    alpha = _probability(alpha)
    family_size = _integer(family_size, "family_size", 1)
    budget = Fraction.from_float(alpha) / (2 * family_size)
    tail = float(budget)
    if Fraction.from_float(tail) > budget:
        tail = float(np.nextafter(tail, 0.0))
    return tail


def clopper_pearson(positive, n, *, alpha=0.05, family_size=1):
    """Two-sided CP interval with fixed-family Bonferroni allocation.

    This numerically evaluates an exact statistical construction; it is not
    certified interval arithmetic. Failed numerical evaluations fail closed.
    """
    n = _integer(n, "n")
    positive = _integer(positive, "positive")
    if positive > n:
        raise ValueError("positive cannot exceed n")
    tail = allocated_tail(alpha, family_size)
    out = {
        "positive": positive,
        "n_nonzero": n,
        "estimate": positive / n if n else None,
        "lower": 0.0,
        "upper": 1.0,
        "tail_budget": tail,
        "status": "ok",
    }
    if n > MAX_QUALIFIED_COUNT:
        out["status"] = "unsupported_count"
        return out
    if not n or tail == 0:
        out["status"] = "empty" if not n else "tail_underflow"
        return out
    if tail < np.finfo(float).tiny:
        out["status"] = "numerical_fallback"
        return out
    logtail = math.log(tail)
    lower = (
        0.0
        if positive == 0
        else (
            math.exp(logtail / n)
            if positive == n
            else float(betaincinv(positive, n - positive + 1, tail))
        )
    )
    upper = (
        1.0
        if positive == n
        else (
            -math.expm1(logtail / n)
            if positive == 0
            else float(betainccinv(positive + 1, n - positive, tail))
        )
    )
    if not (math.isfinite(lower) and math.isfinite(upper) and 0 <= lower <= upper <= 1):
        out["status"] = "numerical_fallback"
        return out
    out.update(
        lower=max(0.0, float(np.nextafter(lower, 0.0))),
        upper=min(1.0, float(np.nextafter(upper, 1.0))),
    )
    return out


def sign_counts(returns):
    values = np.asarray(returns)
    if values.ndim != 2 or values.shape[1] == 0 or values.dtype.kind not in "iuf":
        raise ValueError("returns must be a real rows-by-candidates array")
    if not np.isfinite(values).all():
        raise ValueError("returns must be finite")
    return (values > 0).sum(axis=0), (values != 0).sum(axis=0)


def select_candidate(positive, nonzero):
    """Maximum conditional win fraction; exact comparison, first index on ties."""
    best = None
    for j, (wins, n) in enumerate(zip(positive, nonzero, strict=True)):
        wins, n = _integer(wins, "positive"), _integer(n, "nonzero")
        if wins > n:
            raise ValueError("positive cannot exceed nonzero")
        if n and (best is None or wins * int(nonzero[best]) > int(positive[best]) * n):
            best = j
    return best


def iid_max_coverage(n, search_size, *, correction_size=1, p=0.5, alpha=0.05):
    """Finite-sum oracle for the largest count among iid Bin(n,p) candidates.

    Evaluates the exact discrete law numerically, not Monte Carlo. No ties in
    returns and identical p across independent candidates are required.
    """
    n = _integer(n, "n", 1)
    search_size = _integer(search_size, "search_size", 1)
    p = _probability(p)
    correction_size = _integer(correction_size, "correction_size", 1)
    previous, coverage, mass = 0.0, 0.0, 0.0
    for k in range(n + 1):
        cumulative = float(binom.cdf(k, n, p)) ** search_size
        weight = max(0.0, cumulative - previous)
        interval = clopper_pearson(k, n, alpha=alpha, family_size=correction_size)
        coverage += weight * (interval["lower"] <= p <= interval["upper"])
        mass += weight
        previous = cumulative
    return {"coverage": coverage, "probability_mass": mass}


def _draw(rng, family, n, k):
    pool = k * 16 if family == "hidden_family" else k
    p = np.full(pool, 0.5)
    if family == "signal":
        p[0] = 0.65
    signs = np.where(rng.random((n, pool)) < p, 1, -1)
    if family == "duplicated":
        signs = signs[:, np.arange(pool) // 2]
    elif family == "correlated":
        common = rng.choice([-1, 1], size=(n, 1))
        signs = np.where(rng.random((n, pool)) < 0.75, common, signs)
    elif family == "heterogeneous_ties":
        signs = np.where(rng.random((n, pool)) < np.linspace(0, 0.95, pool), 0, signs)
    elif family == "repeated_row":
        signs[:] = signs[0]
    elif family == "all_ties":
        signs[:] = 0
    return signs, p


def _summary(records, method):
    rows = [r["intervals"][method] for r in records if method in r["intervals"]]
    if not rows:
        return {"evaluated": 0, "conditional_coverage": None, "unconditional_miscoverage": 0.0}
    outcomes = np.array([r["covered"] for r in rows], dtype=float)
    rate, count = float(outcomes.mean()), len(rows)
    z = 1.959963984540054
    center = (rate + z * z / (2 * count)) / (1 + z * z / count)
    half = (
        z * math.sqrt(rate * (1 - rate) / count + z * z / (4 * count * count)) / (1 + z * z / count)
    )
    missed = sum(not r["covered"] for r in rows)
    unconditional = missed / len(records)
    return {
        "evaluated": count,
        "conditional_coverage": rate,
        "unconditional_miscoverage": unconditional,
        "unconditional_miscoverage_mcse": math.sqrt(
            unconditional * (1 - unconditional) / len(records)
        ),
        "mcse": math.sqrt(rate * (1 - rate) / count),
        "wilson_95": [0.0 if rate == 0 else center - half, 1.0 if rate == 1 else center + half],
        "mean_width": float(np.mean([r["upper"] - r["lower"] for r in rows])),
        "vacuous": sum(r["lower"] == 0 and r["upper"] == 1 for r in rows),
        "numerical_fallback": sum(r["status"] == "numerical_fallback" for r in rows),
        "tail_underflow": sum(r["status"] == "tail_underflow" for r in rows),
        "unsupported_count": sum(r["status"] == "unsupported_count" for r in rows),
    }


def run_lab(
    *, repetitions=1000, sample_size=64, n_candidates=16, alpha=0.05, seed=7, families=FAMILIES
):
    start = time.perf_counter()
    repetitions = _integer(repetitions, "repetitions", 1)
    sample_size = _integer(sample_size, "sample_size", 1)
    n_candidates = _integer(n_candidates, "n_candidates", 1)
    seed = _integer(seed, "seed")
    alpha = _probability(alpha)
    families = tuple(families)
    if (
        not families
        or len(set(families)) != len(families)
        or any(f not in FAMILIES for f in families)
    ):
        raise ValueError("families must be a nonempty unique selection of supported families")
    reports = {}
    for family in families:
        rng = np.random.default_rng(np.random.SeedSequence([seed, FAMILIES.index(family)]))
        records = []
        for rep in range(repetitions):
            discovery, targets = _draw(rng, family, sample_size, n_candidates)
            positive, nonzero = sign_counts(discovery)
            selected = select_candidate(positive, nonzero)
            # Freeze the chosen original column before generating independent audit rows.
            audit, _ = _draw(rng, family, sample_size, n_candidates)
            audit_positive, audit_nonzero = sign_counts(audit)
            record = {
                "rep": rep,
                "selected": selected,
                "pool_size": len(targets),
                "sample_sha256": hashlib.sha256(
                    json.dumps(
                        {
                            "shape": list(discovery.shape),
                            "dtype": "int8",
                            "order": "discovery_then_audit_C",
                        },
                        sort_keys=True,
                    ).encode()
                    + discovery.astype(np.int8).tobytes(order="C")
                    + audit.astype(np.int8).tobytes(order="C")
                ).hexdigest(),
                "intervals": {},
                "reported_ids": sorted(
                    range(len(targets)),
                    key=lambda j: (
                        -Fraction(int(positive[j]), int(nonzero[j])) if nonzero[j] else Fraction(1),
                        j,
                    ),
                )[:n_candidates],
            }
            if selected is not None:
                record["target"] = float(targets[selected])
                record["discovery_ties"] = sample_size - int(nonzero[selected])
                record["audit_ties"] = sample_size - int(audit_nonzero[selected])
                for method, wins, count, correction in (
                    ("naive", positive[selected], nonzero[selected], 1),
                    ("bonferroni", positive[selected], nonzero[selected], n_candidates),
                    ("holdout", audit_positive[selected], audit_nonzero[selected], 1),
                    ("full_pool", positive[selected], nonzero[selected], len(targets)),
                ):
                    interval = clopper_pearson(wins, count, alpha=alpha, family_size=correction)
                    interval["covered"] = (
                        interval["lower"] <= targets[selected] <= interval["upper"]
                    )
                    interval["covered"] = bool(interval["covered"])
                    record["intervals"][method] = interval
            records.append(record)
        reports[family] = {
            "iid_rows": family != "repeated_row",
            "fixed_reported_family": family != "hidden_family",
            "population_target_defined": family != "all_ties",
            "selected_nonzero_histogram": {
                str(n): sum(
                    r["intervals"]["naive"]["n_nonzero"] == n
                    for r in records
                    if r["selected"] is not None
                )
                for n in sorted(
                    {
                        r["intervals"]["naive"]["n_nonzero"]
                        for r in records
                        if r["selected"] is not None
                    }
                )
            },
            "no_selection": sum(r["selected"] is None for r in records),
            "summary": {
                m: _summary(records, m) for m in ("naive", "bonferroni", "holdout", "full_pool")
            },
            "records": records,
        }
    return {
        "lab": "selection_interval",
        "schema_version": 1,
        "synthetic_only": True,
        "config": {
            "repetitions": repetitions,
            "sample_size": sample_size,
            "n_candidates": n_candidates,
            "alpha": alpha,
            "seed": seed,
            "families": list(families),
        },
        "provenance": {
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
        "performance": {"wall_seconds": time.perf_counter() - start},
        "families": reports,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    for name, default in (
        ("repetitions", 1000),
        ("sample-size", 64),
        ("n-candidates", 16),
        ("seed", 7),
    ):
        parser.add_argument("--" + name, type=int, default=default)
    parser.add_argument("--alpha", type=float, default=0.05)
    args = vars(parser.parse_args())
    out = args.pop("out")
    report = run_lab(**args)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()

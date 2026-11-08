"""Optional stopping for binary signs: fixed mixtures and invalid controls."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np

MAX_HORIZON = 4096
MAX_LOG_TIME = 2**53 - 1

METHODS = ("repeated_fixed_test", "fixed_bet", "fixed_mixture", "anticipatory_bet")
FAMILIES = {
    "iid_fair": (0.5, 0.5, 0.5),
    "iid_negative": (0.45, 0.45, 0.45),
    "iid_weak_signal": (0.55, 0.55, 0.55),
    "iid_signal": (0.65, 0.65, 0.65),
    "dependent_conditional_null": (0.3, 0.5, 0.375),
    "persistent_marginally_fair": (0.1, 0.9, 0.5),
}


def _integer(value, name, minimum=1):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return int(value)


def _probability(value, name, *, closed=False):
    if (
        isinstance(value, (bool, np.bool_, str, bytes))
        or np.iscomplexobj(value)
        or not np.isscalar(value)
    ):
        raise TypeError(f"{name} must be real")
    value = float(value)
    if not (0 <= value <= 1 if closed else 0 < value < 1):
        raise ValueError(f"{name} is outside its probability range")
    return value


def _mixture(qs):
    qs = tuple(_probability(q, "q") for q in qs)
    if not qs or any(q <= 0.5 for q in qs):
        raise ValueError("fixed mixture requires at least one q in (0.5,1)")
    return qs


def log_mixture_capital(t, successes, qs=(0.55, 0.65, 0.8)):
    """Equal predetermined mixture of Bernoulli likelihood-ratio capitals."""
    t = _integer(t, "t", 0)
    qs = _mixture(qs)
    if t > MAX_LOG_TIME:
        raise ValueError("t exceeds supported exact floating integer range")
    successes = np.asarray(successes)
    if successes.dtype.kind not in "iu" or np.any(successes < 0) or np.any(successes > t):
        raise ValueError("successes must be integer counts in [0,t]")
    components = np.stack(
        [successes * math.log(2 * q) + (t - successes) * math.log(2 * (1 - q)) for q in qs]
    )
    return np.logaddexp.reduce(components, axis=0) - math.log(len(qs))


def _exact_crossed(t, successes, alpha_ratio, ratios, *, anticipatory=False):
    """Compare exact dyadic-input capital with 1/alpha using integers."""
    terms = []
    for numerator, denominator in ratios:
        exponent = (denominator.bit_length() - 2) * (successes if anticipatory else t)
        capital_numerator = pow(numerator, successes)
        if not anticipatory:
            capital_numerator *= pow(denominator - numerator, t - successes)
        terms.append((capital_numerator, exponent))
    common_exponent = max(exponent for _, exponent in terms)
    total = sum(numerator << (common_exponent - exponent) for numerator, exponent in terms)
    return alpha_ratio[0] * total >= alpha_ratio[1] * (len(terms) << common_exponent)


def crossing_boundaries(horizon, *, alpha=0.05, qs=(0.55, 0.65, 0.8)):
    """Minimum positive count crossing each rule at each time (t+1 if none)."""
    horizon = _integer(horizon, "horizon")
    alpha = _probability(alpha, "alpha")
    qs = _mixture(qs)
    if horizon > MAX_HORIZON:
        raise ValueError("horizon exceeds supported limit 4096")
    threshold = -math.log(alpha)
    boundaries = {m: np.arange(horizon + 1, dtype=np.int64) + 1 for m in METHODS}
    numerator, denominator = alpha.as_integer_ratio()
    exact_ratios = {
        "fixed_bet": ((0.65).as_integer_ratio(),),
        "fixed_mixture": tuple(q.as_integer_ratio() for q in qs),
        "anticipatory_bet": ((0.8).as_integer_ratio(),),
    }
    for t in range(1, horizon + 1):
        s = np.arange(t + 1)
        logs = {
            "fixed_bet": s * math.log(1.3) + (t - s) * math.log(0.7),
            "fixed_mixture": log_mixture_capital(t, s, qs),
            "anticipatory_bet": s * math.log(1.6),
        }
        for method, capital in logs.items():
            crossings = np.flatnonzero(capital >= threshold)
            candidate = int(crossings[0]) if len(crossings) else t + 1

            def crossed(count):
                return _exact_crossed(
                    t,
                    count,
                    (numerator, denominator),
                    exact_ratios[method],
                    anticipatory=method == "anticipatory_bet",
                )

            while candidate <= t and not crossed(candidate):
                candidate += 1
            while candidate > 0 and crossed(candidate - 1):
                candidate -= 1
            boundaries[method][t] = candidate
        tail, coefficient = 0, 1
        for k in range(t, -1, -1):
            tail += coefficient
            if tail * denominator <= numerator * (1 << t):
                boundaries["repeated_fixed_test"][t] = k
            else:
                break
            if k:
                coefficient = coefficient * k // (t - k + 1)
    return boundaries


def crossing_probability(boundary, *, p_after_zero=0.5, p_after_one=0.5, initial_positive=0.5):
    """Finite-state forward probability of first crossing, including Markov rows."""
    boundary = np.asarray(boundary)
    if boundary.ndim != 1 or len(boundary) < 2 or boundary.dtype.kind not in "iu":
        raise ValueError("boundary must be an integer vector including unused time zero")
    horizon = len(boundary) - 1
    if horizon > MAX_HORIZON:
        raise ValueError("horizon exceeds supported limit 4096")
    if np.any(boundary < 0) or np.any(boundary > np.arange(horizon + 1) + 1):
        raise ValueError("boundary counts must be in [0,t+1]")
    p0 = _probability(p_after_zero, "p_after_zero", closed=True)
    p1 = _probability(p_after_one, "p_after_one", closed=True)
    initial = _probability(initial_positive, "initial_positive", closed=True)
    alive = np.zeros((2, horizon + 1))
    alive[0, 0], alive[1, 1] = 1 - initial, initial
    first = []
    for t in range(1, horizon + 1):
        if t > 1:
            next_alive = np.zeros_like(alive)
            next_alive[0] = alive[0] * (1 - p0) + alive[1] * (1 - p1)
            next_alive[1, 1:] = alive[0, :-1] * p0 + alive[1, :-1] * p1
            alive = next_alive
        critical = boundary[t]
        first.append(float(alive[:, critical:].sum()))
        alive[:, critical:] = 0
    total = float(sum(first))
    remaining = float(alive.sum())
    if abs(total + remaining - 1) > 1e-10:
        raise ArithmeticError("forward probability mass did not conserve")
    return {
        "crossing_probability": total,
        "survival_probability": remaining,
        "first_crossing_probabilities": first,
        "expected_observed_rows": float(np.arange(1, horizon + 1) @ first + horizon * remaining),
    }


def evaluate_paths(paths, boundaries):
    paths = np.asarray(paths)
    if (
        paths.ndim != 2
        or paths.shape[1] == 0
        or paths.dtype.kind not in "iub"
        or np.any((paths != 0) & (paths != 1))
    ):
        raise ValueError("paths must be a nonempty-time matrix of binary observations")
    counts = np.cumsum(paths, axis=1)
    result = {}
    for method in METHODS:
        boundary = np.asarray(boundaries[method])
        if (
            boundary.shape != (paths.shape[1] + 1,)
            or boundary.dtype.kind not in "iu"
            or np.any(boundary < 0)
            or np.any(boundary > np.arange(paths.shape[1] + 1) + 1)
        ):
            raise ValueError("boundary and path horizon mismatch")
        crossed = counts >= boundary[1:]
        first = np.where(crossed.any(axis=1), crossed.argmax(axis=1) + 1, 0)
        result[method] = first
    return result


def run_lab(
    *,
    horizon=256,
    repetitions=2000,
    alpha=0.05,
    seed=11,
    qs=(0.55, 0.65, 0.8),
    families=tuple(FAMILIES),
):
    start = time.perf_counter()
    horizon = _integer(horizon, "horizon")
    repetitions = _integer(repetitions, "repetitions")
    seed = _integer(seed, "seed", 0)
    alpha = _probability(alpha, "alpha")
    qs = _mixture(qs)
    families = tuple(families)
    if (
        not families
        or len(set(families)) != len(families)
        or any(f not in FAMILIES for f in families)
    ):
        raise ValueError("families must be unique supported names")
    boundaries = crossing_boundaries(horizon, alpha=alpha, qs=qs)
    reports = {}
    for family in families:
        p0, p1, initial = FAMILIES[family]
        rng = np.random.default_rng(np.random.SeedSequence([seed, list(FAMILIES).index(family)]))
        paths = np.empty((repetitions, horizon), dtype=np.uint8)
        paths[:, 0] = rng.random(repetitions) < initial
        for t in range(1, horizon):
            paths[:, t] = rng.random(repetitions) < np.where(paths[:, t - 1], p1, p0)
        first = evaluate_paths(paths, boundaries)
        summaries = {}
        for method in METHODS:
            exact = crossing_probability(
                boundaries[method], p_after_zero=p0, p_after_one=p1, initial_positive=initial
            )
            observed = float(np.mean(first[method] > 0))
            summaries[method] = {
                "finite_sum": exact,
                "observed_crossing_rate": observed,
                "mcse": math.sqrt(observed * (1 - observed) / repetitions),
                "first_crossing_rows": first[method].tolist(),
            }
        reports[family] = {
            "conditional_null": max(p0, p1, initial) <= 0.5,
            "transition": {"p_after_zero": p0, "p_after_one": p1, "initial_positive": initial},
            "sample_sha256": hashlib.sha256(
                json.dumps({"shape": list(paths.shape), "dtype": "uint8"}, sort_keys=True).encode()
                + paths.tobytes()
            ).hexdigest(),
            "methods": summaries,
        }
    return {
        "lab": "sequential_sign",
        "schema_version": 1,
        "synthetic_only": True,
        "config": {
            "horizon": horizon,
            "repetitions": repetitions,
            "alpha": alpha,
            "seed": seed,
            "qs": list(qs),
            "families": list(families),
        },
        "provenance": {
            "numpy": np.__version__,
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
        "performance": {"wall_seconds": time.perf_counter() - start},
        "boundaries": {m: b.tolist() for m, b in boundaries.items()},
        "families": reports,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--horizon", type=int, default=256)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = run_lab(horizon=args.horizon)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()

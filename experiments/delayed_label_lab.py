"""Integer-clock label maturity with exact Gaussian forecast-error references."""

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np


def integer(value, name, minimum=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(name + " must be an integer clock/count")
    value = int(value)
    if minimum is not None and value < minimum:
        raise ValueError(name + " is below minimum")
    return value


def eligible_training(decisions, available, cutoff, n):
    """Latest n distinct decisions strictly before cutoff and released by cutoff."""
    cutoff = integer(cutoff, "cutoff")
    n = integer(n, "n", 1)
    decisions = [integer(x, "decision") for x in decisions]
    available = [integer(x, "available") for x in available]
    if len(decisions) != len(available) or len(set(decisions)) != len(decisions):
        raise ValueError("equal-length clocks and distinct decisions required")
    if any(a < t for t, a in zip(decisions, available, strict=True)):
        raise ValueError("label availability cannot precede its decision")
    selected = sorted(
        t for t, a in zip(decisions, available, strict=True) if t < cutoff and a <= cutoff
    )
    return selected[-n:] if len(selected) >= n else None


def design(horizon, n, lag, *, cutoff=0):
    h = integer(horizon, "horizon", 1)
    n = integer(n, "n", 1)
    lag = integer(lag, "lag", 0)
    cutoff = integer(cutoff, "cutoff")
    if max(h, n, lag) > 4096:
        raise ValueError("qualified counts at most4096")
    # Include enough history for every safe fit without borrowing a future label.
    decisions = list(range(cutoff - n - h - lag, cutoff))
    available = [t + h + lag for t in decisions]
    safe = eligible_training(decisions, available, cutoff, n)
    naive = decisions[-n:]
    clocks = list(range(decisions[0] + 1, cutoff + h + 1))
    lookup = {t: i for i, t in enumerate(clocks)}
    target = np.zeros(len(clocks))
    target[[lookup[t] for t in range(cutoff + 1, cutoff + h + 1)]] = 1
    methods = {}
    for name, selected in [("zero", []), ("mature_only", safe), ("complete_archive", naive)]:
        weights = np.zeros(len(clocks))
        for t in selected:
            for innovation in range(t + 1, t + h + 1):
                weights[lookup[innovation]] += 1 / n
        error = weights - target
        covariance = float(weights @ target)
        methods[name] = {
            "training_decisions": selected,
            "training_available": [t + h + lag for t in selected],
            "unmatured_count": sum(t + h + lag > cutoff for t in selected),
            "future_content_label_count": sum(t + h > cutoff for t in selected),
            "innovation_weights": weights.tolist(),
            "prediction_variance": float(weights @ weights),
            "target_covariance": covariance,
            "expected_mse": float(error @ error),
            "mse_standard_deviation": math.sqrt(2) * float(error @ error),
        }
    return {
        "horizon": h,
        "n": n,
        "lag": lag,
        "cutoff": cutoff,
        "innovation_clocks": clocks,
        "target_weights": target.tolist(),
        "methods": methods,
    }


def run_lab(
    *, horizons=(1, 4, 16), training_multipliers=(1, 4), lags=(0, 2), repetitions=10000, seed=101
):
    start = time.perf_counter()
    horizons = tuple(horizons)
    training_multipliers = tuple(training_multipliers)
    lags = tuple(lags)
    repetitions = integer(repetitions, "repetitions", 2)
    seed = integer(seed, "seed", 0)
    if not all((horizons, training_multipliers, lags)):
        raise ValueError("nonempty grid required")
    horizons = tuple(integer(h, "horizon", 1) for h in horizons)
    training_multipliers = tuple(integer(m, "training multiplier", 1) for m in training_multipliers)
    lags = tuple(integer(lag, "lag", 0) for lag in lags)
    records = []
    for h in horizons:
        for multiplier in training_multipliers:
            for lag in lags:
                layout = design(h, h * multiplier, lag)
                h, n, lag = layout["horizon"], layout["n"], layout["lag"]
                rng = np.random.default_rng(np.random.SeedSequence([seed, h, n, lag]))
                innovations = rng.normal(size=(repetitions, len(layout["innovation_clocks"])))
                target = innovations @ np.asarray(layout["target_weights"])
                errors = {}
                summaries = {}
                for name, method in layout["methods"].items():
                    prediction = innovations @ np.asarray(method["innovation_weights"])
                    squared = (prediction - target) ** 2
                    errors[name] = squared
                    summaries[name] = {
                        "mean_mse": float(squared.mean()),
                        "path_mcse": float(squared.std(ddof=1) / math.sqrt(repetitions)),
                        "squared_errors": squared.tolist(),
                        "predictions": prediction.tolist(),
                    }
                paired = {}
                for name in ["mature_only", "complete_archive"]:
                    delta = errors[name] - errors["zero"]
                    target_weights = np.asarray(layout["target_weights"])
                    error_weights = (
                        np.asarray(layout["methods"][name]["innovation_weights"]) - target_weights
                    )
                    covariance = float(error_weights @ (-target_weights))
                    paired_variance = max(
                        0.0,
                        2 * layout["methods"][name]["expected_mse"] ** 2
                        + 2 * h**2
                        - 4 * covariance**2,
                    )
                    paired[name] = {
                        "mean_mse_minus_zero": float(delta.mean()),
                        "paired_path_mcse": float(delta.std(ddof=1) / math.sqrt(repetitions)),
                        "analytic_paired_path_standard_error": math.sqrt(
                            paired_variance / repetitions
                        ),
                    }
                records.append(
                    {
                        "design": layout,
                        "input_sha256": hashlib.sha256(
                            np.asarray(innovations, dtype="<f8").tobytes()
                        ).hexdigest(),
                        "targets": target.tolist(),
                        "summary": summaries,
                        "paired_comparison": paired,
                    }
                )
    return {
        "lab": "delayed_label",
        "schema_version": 1,
        "synthetic_only": True,
        "config": {
            "horizons": list(horizons),
            "training_multipliers": list(training_multipliers),
            "lags": list(lags),
            "repetitions": repetitions,
            "seed": seed,
        },
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

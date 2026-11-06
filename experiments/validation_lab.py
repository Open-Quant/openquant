"""Offline, synthetic selection-validation experiment (not a trading strategy).

Select a candidate on a training prefix, freeze that choice, then score the suffix.
The best candidate chosen on the suffix is a deliberately contaminated comparator,
never an estimate of out-of-sample skill. See VALIDATION_LAB.md for assumptions.

Only NumPy is required. No market downloads, compiled extension, fitting, RAS bound,
PSR/DSR confidence claim, promotion gate, or live trading is involved.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np

# Stable family IDs keep each family's random stream independent of invocation order.
FAMILIES = (
    "iid_null",
    "correlated_null",
    "duplicated_null",
    "heavy_tailed_null",
    "serial_null",
    "positive_drift",
)
FAMILY_PARAMETERS = {
    "iid_null": {"distribution": "Gaussian", "marginal_std": 0.01, "mean": 0.0},
    "correlated_null": {
        "distribution": "Gaussian",
        "marginal_std": 0.01,
        "mean": 0.0,
        "cross_candidate_correlation": 0.75,
    },
    "duplicated_null": {
        "distribution": "Gaussian",
        "marginal_std": 0.01,
        "mean": 0.0,
        "copies_per_distinct_candidate": 2,
    },
    "heavy_tailed_null": {
        "distribution": "Student t",
        "degrees_of_freedom": 3,
        "marginal_std": 0.01,
        "mean": 0.0,
    },
    "serial_null": {
        "distribution": "stationary Gaussian AR(1)",
        "ar_coefficient": 0.6,
        "marginal_std": 0.01,
        "mean": 0.0,
    },
    "positive_drift": {
        "distribution": "Gaussian",
        "marginal_std": 0.01,
        "signal_candidate": 0,
        "signal_mean": 0.001,
        "other_candidates_mean": 0.0,
    },
}


def _integer(value: int, name: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return int(value)


def _matrix(values: Any) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 2 or array.shape[0] < 3 or array.shape[1] < 1:
        raise ValueError("returns must have shape (observations >= 3, candidates >= 1)")
    if array.dtype.kind not in "iuf":
        raise TypeError("returns must contain real numbers")
    array = np.asarray(array, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError("returns must contain only finite values")
    return array


def score_candidates(returns: Any) -> np.ndarray:
    """Per-observation mean / sample deviation, one value per candidate column.

    Constant columns are rejected, not silently assigned a zero or infinite score.
    Scale before moments to avoid overflow/underflow; the ratio is scale invariant.
    """
    values = _matrix(returns)
    scale = np.max(np.abs(values), axis=0)
    if np.any(scale == 0):
        raise ValueError("constant returns have undefined Sharpe ratios")
    scaled = values / scale
    deviation = scaled.std(axis=0, ddof=1)
    if np.any(deviation == 0) or np.any(np.ptp(scaled, axis=0) == 0):
        raise ValueError("constant returns have undefined Sharpe ratios")
    result = scaled.mean(axis=0) / deviation
    if not np.isfinite(result).all():
        raise ValueError("returns produce nonfinite Sharpe ratios")
    return result


def select_on_training(training_returns: Any) -> int:
    """Choose by training Sharpe only; exact ties select the first candidate."""
    return int(np.argmax(score_candidates(training_returns)))


def evaluate_split(returns: Any, *, train_size: int, gap: int = 0) -> dict[str, Any]:
    """Freeze training selection before scoring the chronological holdout.

    Ranges are half-open row indices. The optional gap belongs to neither sample.
    The gap is a fixed separation, not event-overlap purging or proof of independence.
    """
    values = _matrix(returns)
    train_size = _integer(train_size, "train_size", 3)
    gap = _integer(gap, "gap", 0)
    holdout_start = train_size + gap
    if len(values) - holdout_start < 3:
        raise ValueError("holdout needs at least 3 observations")
    training = values[:train_size]
    selected = select_on_training(training)
    train_scores = score_candidates(training)
    # No information from this suffix is supplied to the selection function.
    holdout_scores = score_candidates(values[holdout_start:])
    contaminated = int(np.argmax(holdout_scores))
    holdout = float(holdout_scores[selected])
    return {
        "selected_candidate": selected,
        "train_range": [0, train_size],
        "holdout_range": [holdout_start, len(values)],
        "train_sharpe": float(train_scores[selected]),
        "holdout_sharpe": holdout,
        "contaminated_candidate": contaminated,
        "contaminated_holdout_sharpe": float(holdout_scores[contaminated]),
        "selection_optimism": float(train_scores[selected]) - holdout,
    }


def simulate_family(family: str, n_periods: int, n_candidates: int, *, seed: Any) -> np.ndarray:
    """Create original synthetic returns, not fitted real-market returns.

    Columns are candidates and rows chronological observations. The serial family
    starts in its stationary distribution and continues across the split and gap.
    Positive drift is a known data-generating mean, not a fitted predictive model.
    """
    if family not in FAMILIES:
        raise ValueError(f"unknown family: {family}")
    n_periods = _integer(n_periods, "n_periods", 3)
    n_candidates = _integer(n_candidates, "n_candidates", 1)
    rng = np.random.default_rng(seed)
    shape = (n_periods, n_candidates)
    if family == "duplicated_null":
        base = rng.normal(size=(n_periods, (n_candidates + 1) // 2))
        values = np.repeat(base, 2, axis=1)[:, :n_candidates]
    elif family == "correlated_null":
        common = rng.normal(size=(n_periods, 1))
        values = math.sqrt(0.75) * common + math.sqrt(0.25) * rng.normal(size=shape)
    elif family == "heavy_tailed_null":
        # Var(t_3) = 3, but its fourth moment is infinite.
        values = rng.standard_t(df=3, size=shape) / math.sqrt(3)
    else:
        values = rng.normal(size=shape)
        if family == "serial_null":
            phi = 0.6
            for row in range(1, n_periods):
                values[row] = phi * values[row - 1] + math.sqrt(1 - phi**2) * values[row]
    values *= 0.01
    if family == "positive_drift":
        values[:, 0] += 0.001
    return values


def summarize_values(values: Any) -> dict[str, float]:
    """Mean and Monte Carlo standard error across independent replications.

    This is simulation precision, not a strategy confidence interval or a p-value.
    """
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or len(array) < 2 or not np.isfinite(array).all():
        raise ValueError("summary needs at least two finite replication values")
    return {
        "mean": float(np.mean(array)),
        "monte_carlo_standard_error": float(np.std(array, ddof=1) / math.sqrt(len(array))),
    }


def run_lab(
    *,
    repetitions: int = 200,
    train_size: int = 252,
    holdout_size: int = 252,
    n_candidates: int = 64,
    gap: int = 0,
    seed: int = 7,
    families: Any = FAMILIES,
) -> dict[str, Any]:
    """Run six controlled families, retaining all selection results and input hashes."""
    repetitions = _integer(repetitions, "repetitions", 2)
    # At n=3 even Gaussian sample Sharpe has infinite variance (t_2 / sqrt(3)),
    # so a finite-variance Monte Carlo standard error would be misleading.
    train_size = _integer(train_size, "train_size", 4)
    holdout_size = _integer(holdout_size, "holdout_size", 4)
    n_candidates = _integer(n_candidates, "n_candidates", 1)
    gap = _integer(gap, "gap", 0)
    seed = _integer(seed, "seed", 0)
    families = tuple(families)
    if not families or any(f not in FAMILIES for f in families):
        raise ValueError("families must be a nonempty selection of known families")
    if len(set(families)) != len(families):
        raise ValueError("families must not contain duplicates")
    total = train_size + gap + holdout_size
    scenarios = []
    for family in families:
        digest = hashlib.sha256()
        records = []
        for replication in range(repetitions):
            stream = np.random.SeedSequence([seed, FAMILIES.index(family), replication])
            returns = simulate_family(family, total, n_candidates, seed=stream)
            # Hash explicit shape plus canonical little-endian row-major float64 bytes.
            digest.update(np.asarray(returns.shape, dtype="<i8").tobytes())
            digest.update(np.asarray(returns, dtype="<f8").tobytes(order="C"))
            record = evaluate_split(returns, train_size=train_size, gap=gap)
            record["replication"] = replication
            records.append(record)
        summary = {
            metric: summarize_values([row[metric] for row in records])
            for metric in (
                "train_sharpe",
                "holdout_sharpe",
                "contaminated_holdout_sharpe",
                "selection_optimism",
            )
        }
        summary["holdout_nonpositive_fraction"] = (
            sum(row["holdout_sharpe"] <= 0 for row in records) / repetitions
        )
        if family == "positive_drift":
            summary["known_signal_selected_fraction"] = (
                sum(row["selected_candidate"] == 0 for row in records) / repetitions
            )
        scenarios.append(
            {
                "family": family,
                "generator_parameters": dict(FAMILY_PARAMETERS[family]),
                "dataset_sha256": digest.hexdigest(),
                "summary": summary,
                "replications": records,
            }
        )
    return {
        "schema_version": 1,
        "synthetic_only": True,
        "sharpe_units": "per_observation",
        "config": {
            "repetitions": repetitions,
            "train_size": train_size,
            "holdout_size": holdout_size,
            "n_candidates": n_candidates,
            "gap": gap,
            "seed": seed,
            "families": list(families),
        },
        "selection_rule": "maximum training sample Sharpe, first index on exact ties",
        "caveats": [
            "Synthetic candidate returns only; no strategy fitting or real-market evidence.",
            "Contaminated holdout selection is a diagnostic, not deployable performance.",
            "Serial dependence persists across the chronological split; a gap is not proof of independence.",
            "No multiple-testing correction, RAS confidence bound, or promotion decision is produced.",
            "Monte Carlo standard errors describe replicate means, not real-strategy confidence.",
            "Monte Carlo precision can be unstable for small samples or few repetitions.",
        ],
        "scenarios": scenarios,
    }


def _provenance() -> dict[str, str]:
    source = Path(__file__).resolve()
    try:
        sha = subprocess.check_output(
            ["git", "-C", str(source.parent), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        sha = "unavailable"
    return {
        "git_sha": sha,
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "platform": platform.platform(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repetitions", type=int, default=200)
    parser.add_argument("--train-size", type=int, default=252)
    parser.add_argument("--holdout-size", type=int, default=252)
    parser.add_argument("--candidates", type=int, default=64)
    parser.add_argument("--gap", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--families", nargs="+", choices=FAMILIES, default=list(FAMILIES))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    start = time.perf_counter()
    try:
        report = run_lab(
            repetitions=args.repetitions,
            train_size=args.train_size,
            holdout_size=args.holdout_size,
            n_candidates=args.candidates,
            gap=args.gap,
            seed=args.seed,
            families=args.families,
        )
    except (ValueError, TypeError) as exc:
        parser.error(str(exc))
    elapsed = time.perf_counter() - start
    count = (
        len(args.families)
        * args.repetitions
        * args.candidates
        * (args.train_size + args.gap + args.holdout_size)
    )
    report["provenance"] = _provenance()
    report["performance"] = {
        "wall_seconds": elapsed,
        "candidate_observations": count,
        "candidate_observations_per_second": count / elapsed,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Synthetic validation report: {args.out} ({elapsed:.3f}s)")


if __name__ == "__main__":
    main()

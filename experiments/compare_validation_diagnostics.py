"""Compare existing OpenQuant diagnostics on fixed synthetic selection controls.

This optional adapter needs the built OpenQuant extension. The base validation_lab
remains NumPy-only. Scores and threshold exceedance rates are diagnostic outputs,
not calibrated probabilities of skill, trading decisions, or a RAS implementation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import tempfile
import time
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np

# Direct script invocation places experiments/, not the repository root, on sys.path.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments import validation_lab as lab  # noqa: E402 - bootstrap direct script imports

METRICS = (
    "training_psr",
    "training_dsr",
    "holdout_psr",
    "contaminated_holdout_psr",
    "exact_unique_training_dsr",
)


def _evaluation():
    try:
        from openquant import evaluation
    except ImportError as exc:
        raise RuntimeError(
            "Build the OpenQuant extension and run with the project environment; "
            "the standalone validation_lab.py does not need the extension."
        ) from exc
    return evaluation


def _scaled(values: np.ndarray) -> np.ndarray:
    # All diagnostics are invariant under positive column scaling. Normalizing before
    # calling the existing API prevents overflow in its unscaled higher moments.
    lab.score_candidates(values)  # Validate finiteness, shape, and nonconstant columns.
    return values / np.max(np.abs(values), axis=0)


def _unique_indices(values: np.ndarray) -> np.ndarray:
    # Numeric exact equality, not tolerance clustering or an effective-trials estimate.
    _, indices = np.unique(values.T, axis=0, return_index=True)
    return np.sort(indices)


def _probability(value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("diagnostic output must be finite and between zero and one")
    return value


def _dsr(evaluation: Any, values: np.ndarray, scores: np.ndarray) -> float | None:
    if len(scores) < 2:
        return None
    return _probability(evaluation.deflated_sharpe_ratio(values, scores.tolist()))


def compare_candidates(returns: Any, *, train_size: int, gap: int = 0) -> dict[str, Any]:
    """Compare scores for a choice frozen using only the training prefix."""
    choice = lab.evaluate_split(returns, train_size=train_size, gap=gap)
    values = np.asarray(returns, dtype=float)
    training = values[:train_size]
    holdout = values[train_size + gap :]
    normalized_train, normalized_holdout = _scaled(training), _scaled(holdout)
    scores = lab.score_candidates(training)
    selected = choice["selected_candidate"]
    contaminated = choice["contaminated_candidate"]
    unique = _unique_indices(training)
    evaluation = _evaluation()
    return {
        **choice,
        "n_candidates": values.shape[1],
        "n_exact_unique_training_returns": len(unique),
        "trial_sharpe_std": float(np.std(scores, ddof=0)),
        "training_psr": _probability(
            evaluation.probabilistic_sharpe_ratio(normalized_train[:, selected])
        ),
        "training_dsr": _dsr(evaluation, normalized_train[:, selected], scores),
        "dsr_status": "available" if len(scores) >= 2 else "requires_at_least_two_trials",
        "exact_unique_training_dsr": _dsr(
            evaluation, normalized_train[:, selected], scores[unique]
        ),
        "holdout_psr": _probability(
            evaluation.probabilistic_sharpe_ratio(normalized_holdout[:, selected])
        ),
        "contaminated_holdout_psr": _probability(
            evaluation.probabilistic_sharpe_ratio(normalized_holdout[:, contaminated])
        ),
    }


def audit_registry(training_returns: Any, *, path: Path, selected_candidate: int) -> dict[str, Any]:
    """Exercise actual persisted registry semantics, isolated from other experiments.

    Distinct column IDs are distinct configurations even if their returns coincide.
    Reopening and re-recording one config must preserve that nominal count. A caller's
    existing registry is never touched. Returned evidence excludes wall-clock timestamps.
    """
    path = Path(path)
    if path.exists():
        raise ValueError("registry audit requires a new path to avoid mixing trial families")
    values = lab._matrix(training_returns)
    normalized = _scaled(values)
    selected_candidate = lab._integer(selected_candidate, "selected_candidate", 0)
    if selected_candidate >= values.shape[1]:
        raise ValueError("selected_candidate is outside the candidate matrix")
    evaluation = _evaluation()
    registry = evaluation.TrialRegistry(path)
    configs = [
        {"candidate_id": i, "normalization": "positive column maxabs"}
        for i in range(values.shape[1])
    ]
    for index, config in enumerate(configs):
        registry.record(config, normalized[:, index])
    recorded = registry.n_trials
    reopened = evaluation.TrialRegistry(path)
    reopened_count = reopened.n_trials
    reopened.record(configs[selected_candidate], normalized[:, selected_candidate])
    count_after_rerecord = reopened.n_trials
    scores = lab.score_candidates(values)
    differences = [
        abs(trial.sharpe - float(scores[trial.config["candidate_id"]])) for trial in reopened.trials
    ]
    max_difference = max(differences)
    if any(not math.isfinite(difference) or difference > 1e-10 for difference in differences):
        raise RuntimeError("a persisted registry Sharpe differs from the direct score")
    direct = _dsr(evaluation, normalized[:, selected_candidate], scores)
    registry_dsr = (
        None
        if len(scores) < 2
        else _probability(reopened.deflated_sharpe_ratio(normalized[:, selected_candidate]))
    )
    if (recorded, reopened_count, count_after_rerecord) != (len(scores),) * 3:
        raise RuntimeError("trial registry count changed unexpectedly")
    if direct is not None and (
        registry_dsr is None or not math.isclose(direct, registry_dsr, abs_tol=1e-10, rel_tol=0)
    ):
        raise RuntimeError("trial registry DSR differs from direct DSR")
    return {
        "recorded_trials": recorded,
        "reopened_trials": reopened_count,
        "trials_after_rerecord": count_after_rerecord,
        "n_exact_unique_returns": len(_unique_indices(values)),
        "registry_sharpe_max_abs_error": max_difference,
        "registry_dsr": registry_dsr,
        "direct_dsr": direct,
        "candidate_config_hashes": sorted(trial.config_hash for trial in reopened.trials),
    }


def summarize_probabilities(
    values: list[float | None], *, threshold: float = 0.95
) -> dict[str, Any]:
    """Descriptive threshold rate and Wilson 95% interval across replications.

    The interval quantifies binomial simulation uncertainty; it is not an interval
    for profitability or true Sharpe. Unavailable scores are counted, never set to zero.
    """
    threshold = float(threshold)
    if not math.isfinite(threshold) or not 0 < threshold < 1:
        raise ValueError("threshold must be finite and in (0, 1)")
    available = [_probability(value) for value in values if value is not None]
    n = len(available)
    count = sum(value >= threshold for value in available)
    interval = None
    rate = None
    if n:
        rate = count / n
        z = NormalDist().inv_cdf(0.975)
        denominator = 1 + z * z / n
        center = (rate + z * z / (2 * n)) / denominator
        radius = z * math.sqrt(rate * (1 - rate) / n + z * z / (4 * n * n)) / denominator
        interval = [
            0.0 if count == 0 else max(0.0, center - radius),
            1.0 if count == n else min(1.0, center + radius),
        ]
    return {
        "score_mean": float(np.mean(available)) if n else None,
        "score_monte_carlo_standard_error": float(np.std(available, ddof=1) / math.sqrt(n))
        if n >= 2
        else None,
        "n_available": n,
        "n_unavailable": len(values) - n,
        "threshold": threshold,
        "n_at_or_above_threshold": count,
        "threshold_exceedance_fraction": rate,
        "wilson_interval_95": interval,
    }


def run_comparison(
    *,
    repetitions: int = 200,
    train_size: int = 252,
    holdout_size: int = 252,
    n_candidates: int = 64,
    gap: int = 0,
    seed: int = 7,
    families: Any = lab.FAMILIES,
    threshold: float = 0.95,
    registry_repetitions: int = 1,
) -> dict[str, Any]:
    """Replay the base lab's exact samples and compare production diagnostic APIs."""
    summarize_probabilities([], threshold=threshold)
    registry_repetitions = lab._integer(registry_repetitions, "registry_repetitions", 0)
    if registry_repetitions > repetitions:
        raise ValueError("registry_repetitions cannot exceed repetitions")
    _evaluation()  # Fail before simulation if the extension is unavailable.
    report = lab.run_lab(
        repetitions=repetitions,
        train_size=train_size,
        holdout_size=holdout_size,
        n_candidates=n_candidates,
        gap=gap,
        seed=seed,
        families=families,
    )
    total = train_size + gap + holdout_size
    with tempfile.TemporaryDirectory(prefix="openquant-registry-audit-") as directory:
        for scenario in report["scenarios"]:
            family = scenario["family"]
            audits = []
            digest = hashlib.sha256()
            for replication, record in enumerate(scenario["replications"]):
                stream = np.random.SeedSequence([seed, lab.FAMILIES.index(family), replication])
                values = lab.simulate_family(family, total, n_candidates, seed=stream)
                digest.update(np.asarray(values.shape, dtype="<i8").tobytes())
                digest.update(np.asarray(values, dtype="<f8").tobytes(order="C"))
                record["diagnostics"] = compare_candidates(values, train_size=train_size, gap=gap)
                if replication < registry_repetitions:
                    audit = audit_registry(
                        values[:train_size],
                        path=Path(directory) / f"{family}-{replication}.json",
                        selected_candidate=record["selected_candidate"],
                    )
                    audit["replication"] = replication
                    audits.append(audit)
            if digest.hexdigest() != scenario["dataset_sha256"]:
                raise RuntimeError("diagnostic replay changed the base control dataset")
            scenario["registry_audits"] = audits
            scenario["diagnostic_summary"] = {
                metric: summarize_probabilities(
                    [r["diagnostics"][metric] for r in scenario["replications"]],
                    threshold=threshold,
                )
                for metric in METRICS
            }
            scenario["diagnostic_assumption_flags"] = {
                "iid_null": [],
                "correlated_null": [
                    "Candidate returns are cross-correlated; nominal trial count is not an independent-trials count."
                ],
                "duplicated_null": [
                    "Distinct configurations can have exactly identical returns; the registry still counts all configurations."
                ],
                "heavy_tailed_null": [
                    "Student t(3) has infinite fourth moment; finite-moment asymptotic interpretations are not justified."
                ],
                "serial_null": [
                    "Positive serial dependence violates an independent-observations interpretation; no long-run variance correction is applied."
                ],
                "positive_drift": [
                    "Known mean is injected by construction; this is not a null rejection experiment or learned alpha."
                ],
            }[family]
    report["caveats"] = [
        x for x in report["caveats"] if not x.startswith("No multiple-testing correction")
    ]
    report["caveats"] += [
        "PSR/DSR outputs are compared descriptively; no posterior probability of skill or calibration is claimed.",
        "Training and contaminated-holdout diagnostics are post-selection, not independent holdout tests.",
        "Exact-duplicate collapse is a sensitivity comparison, not an estimated effective trial count or general correlation correction.",
        "All registry configurations are recorded within the same training window; holdout returns never enter the registry.",
        "No RAS confidence bound, promotion decision, real trading strategy or real-market result is produced.",
    ]
    report["diagnostics"] = {
        "threshold": threshold,
        "registry_repetitions_per_family": registry_repetitions,
        "scoring_normalization": "positive column maxabs; scale-invariant moments",
        "registry_files": "Temporary isolated files, reopened and re-recorded; summary evidence retained without timestamps.",
        "interval": "Wilson 95% for independent replication threshold events; descriptive, not simultaneous across metrics.",
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for flag, default in [
        ("repetitions", 200),
        ("train-size", 252),
        ("holdout-size", 252),
        ("candidates", 64),
        ("gap", 0),
        ("seed", 7),
        ("registry-repetitions", 1),
    ]:
        parser.add_argument("--" + flag, type=int, default=default)
    parser.add_argument("--families", nargs="+", choices=lab.FAMILIES, default=list(lab.FAMILIES))
    parser.add_argument("--threshold", type=float, default=0.95)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    start = time.perf_counter()
    try:
        report = run_comparison(
            repetitions=args.repetitions,
            train_size=args.train_size,
            holdout_size=args.holdout_size,
            n_candidates=args.candidates,
            gap=args.gap,
            seed=args.seed,
            families=args.families,
            threshold=args.threshold,
            registry_repetitions=args.registry_repetitions,
        )
    except (ValueError, TypeError, RuntimeError) as exc:
        parser.error(str(exc))
    elapsed = time.perf_counter() - start
    evaluation = _evaluation()
    report["provenance"] = lab._provenance()
    with Path(evaluation._core.__file__).open("rb") as native:
        native_digest = hashlib.file_digest(native, "sha256").hexdigest()
    report["diagnostics"]["provenance"] = {
        "adapter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "evaluation_sha256": hashlib.sha256(Path(evaluation.__file__).read_bytes()).hexdigest(),
        "core_version": getattr(evaluation._core, "__version__", "unavailable"),
        "native_extension_sha256": native_digest,
    }
    report["performance"] = {
        "wall_seconds": elapsed,
        "includes": "base lab, replay, diagnostics and temporary registry audits; excludes report serialization and provenance",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Diagnostic comparison: {args.out} ({elapsed:.3f}s)")


if __name__ == "__main__":
    main()

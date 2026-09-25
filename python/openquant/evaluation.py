"""Research evaluation: is a backtest's Sharpe ratio evidence of skill?

A thin layer over the compiled `backtest_stats` and `strategy_risk` modules that works
from a returns series instead of from pre-computed moments:

- `probabilistic_sharpe_ratio`, `deflated_sharpe_ratio`, `minimum_track_record_length`
  (Bailey and López de Prado 2012, 2014; AFML Chapter 14, §14.7);
- `meta_label_metrics`: precision, recall and F1 of a meta-labeling overlay against the
  primary model it filters (AFML Chapter 3, §3.6);
- `strategy_failure_probability` (AFML Chapter 15, §15.4);
- `TrialRegistry`: a JSON file that records every configuration tried, across runs, so the
  deflated Sharpe ratio deflates by the number of trials actually run.

Conventions, shared with `backtest_stats`: every Sharpe ratio here is **per period**
(mean over sample standard deviation, not annualised), skewness and kurtosis are the
population moment ratios, and kurtosis is **raw** (3 for a normal distribution).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from . import _core

_bs = _core.backtest_stats
_sr = _core.strategy_risk

REGISTRY_SCHEMA = 1


@dataclass(frozen=True)
class ReturnMoments:
    """The sample statistics PSR, DSR and MinTRL are computed from."""

    n_obs: int
    mean: float
    std: float
    sharpe: float
    skewness: float
    kurtosis: float


@dataclass(frozen=True)
class Trial:
    """One configuration's record in a `TrialRegistry`."""

    config_hash: str
    timestamp: str
    sharpe: float
    n_obs: int
    skewness: float
    kurtosis: float
    config: Any = None


# --- validation ------------------------------------------------------------------------


def _as_returns(returns: Iterable[float], name: str = "returns") -> list[float]:
    if isinstance(returns, (str, bytes)):
        raise TypeError(f"{name} must be a sequence of numbers, not a string")
    try:
        out = [float(r) for r in returns]
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a sequence of numbers") from exc
    if len(out) < 3:
        raise ValueError(f"{name} needs at least 3 observations, got {len(out)}")
    if not all(math.isfinite(r) for r in out):
        raise ValueError(f"{name} must contain only finite values")
    return out


def _finite(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value}")
    return value


def _check_alpha(alpha: float) -> float:
    alpha = float(alpha)
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    return alpha


# --- Sharpe ratio statistics -------------------------------------------------------------


def return_moments(returns: Iterable[float]) -> ReturnMoments:
    """Per-period Sharpe ratio, skewness and raw kurtosis of a returns series.

    `std` is the sample standard deviation (n - 1); skewness and kurtosis are the
    population ratios m3/m2^1.5 and m4/m2^2, as in Bailey and López de Prado.
    """
    r = _as_returns(returns)
    n = len(r)
    mean = sum(r) / n
    dev = [x - mean for x in r]
    m2 = sum(d * d for d in dev) / n
    if m2 <= 0.0:
        raise ValueError("returns are constant; the Sharpe ratio is undefined")
    m3 = sum(d**3 for d in dev) / n
    m4 = sum(d**4 for d in dev) / n
    return ReturnMoments(
        n_obs=n,
        mean=mean,
        std=math.sqrt(m2 * n / (n - 1)),
        sharpe=_bs.sharpe_ratio(r, 1.0, 0.0),
        skewness=m3 / m2**1.5,
        kurtosis=m4 / m2**2,
    )


def probabilistic_sharpe_ratio(returns: Iterable[float], benchmark_sr: float = 0.0) -> float:
    """PSR: the probability that the true per-period Sharpe ratio exceeds `benchmark_sr`.

    PSR(SR*) = Z[(SR - SR*)·sqrt(T - 1) / sqrt(1 - γ3·SR + (γ4 - 1)/4·SR²)].
    """
    m = return_moments(returns)
    benchmark_sr = _finite(benchmark_sr, "benchmark_sr")
    return _bs.probabilistic_sharpe_ratio(m.sharpe, benchmark_sr, m.n_obs, m.skewness, m.kurtosis)


def expected_max_sharpe(n_trials: int, sharpe_std: float) -> float:
    """SR0: the expected maximum per-period Sharpe ratio of `n_trials` skill-less trials.

    SR0 = σ_SR·[(1 - γ)·Z⁻¹(1 - 1/N) + γ·Z⁻¹(1 - 1/(N·e))], γ the Euler–Mascheroni constant.
    """
    n = _n_trials(n_trials)
    sd = _sharpe_std(sharpe_std)
    # The observed-SR and moment arguments are unused when benchmark_out is true.
    return _bs.deflated_sharpe_ratio(0.0, [sd, float(n)], 2, 0.0, 3.0, True, True)


def deflated_sharpe_ratio(
    returns: Iterable[float],
    trial_sharpes: Sequence[float] | None = None,
    *,
    n_trials: int | None = None,
    sharpe_std: float | None = None,
) -> float:
    """DSR: the PSR measured against SR0, the best Sharpe ratio luck alone would produce.

    Give the trials either as `trial_sharpes` — every configuration's per-period Sharpe
    ratio, including this one — or as `n_trials` and `sharpe_std`, the count and the
    standard deviation of their Sharpe ratios. `TrialRegistry.deflated_sharpe_ratio`
    takes both from a registry.
    """
    if trial_sharpes is not None:
        if n_trials is not None or sharpe_std is not None:
            raise ValueError("pass trial_sharpes or (n_trials, sharpe_std), not both")
        sharpes = _as_trial_sharpes(trial_sharpes)
        n_trials, sharpe_std = len(sharpes), _population_std(sharpes)
    elif n_trials is None or sharpe_std is None:
        raise ValueError("pass trial_sharpes, or both n_trials and sharpe_std")
    sr0 = expected_max_sharpe(n_trials, sharpe_std)
    return probabilistic_sharpe_ratio(returns, sr0)


def minimum_track_record_length(
    returns: Iterable[float], benchmark_sr: float = 0.0, alpha: float = 0.05
) -> float:
    """MinTRL: observations needed for PSR(`benchmark_sr`) to reach 1 - `alpha`.

    MinTRL = 1 + (1 - γ3·SR + (γ4 - 1)/4·SR²)·(Z(1 - α) / (SR - SR*))².

    Returns `math.inf` when the observed Sharpe ratio is at or below the benchmark: no
    track record of this quality is long enough.
    """
    m = return_moments(returns)
    benchmark_sr = _finite(benchmark_sr, "benchmark_sr")
    alpha = _check_alpha(alpha)
    if m.sharpe <= benchmark_sr:
        return math.inf
    return _bs.minimum_track_record_length(m.sharpe, benchmark_sr, m.skewness, m.kurtosis, alpha)


def _n_trials(n_trials: int) -> int:
    if isinstance(n_trials, bool) or not isinstance(n_trials, int):
        raise TypeError(f"n_trials must be an int, got {type(n_trials).__name__}")
    if n_trials < 2:
        raise ValueError(f"n_trials must be at least 2, got {n_trials}")
    return n_trials


def _sharpe_std(sharpe_std: float) -> float:
    sd = _finite(sharpe_std, "sharpe_std")
    if sd < 0.0:
        raise ValueError(f"sharpe_std must be non-negative, got {sd}")
    return sd


def _as_trial_sharpes(values: Sequence[float]) -> list[float]:
    out = [_finite(v, "trial_sharpes") for v in values]
    if len(out) < 2:
        raise ValueError(f"trial_sharpes needs at least 2 trials, got {len(out)}")
    return out


def _population_std(values: Sequence[float]) -> float:
    mean = sum(values) / len(values)
    return math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))


# --- meta-labeling -----------------------------------------------------------------------


def meta_label_metrics(
    meta_labels: Sequence[int], meta_predictions: Sequence[float], threshold: float = 0.5
) -> dict[str, float | int]:
    """Precision, recall and F1 of a meta-labeling overlay, and of the primary model alone.

    `meta_labels[i]` is 1 when the primary model's bet `i` was profitable, else 0.
    `meta_predictions[i]` is the secondary model's probability (or 0/1 decision) that it
    was; the overlay takes the bet when it is at least `threshold`. The primary model alone
    takes every bet, so its precision is the share of profitable bets and its recall is 1.
    A ratio with a zero denominator is reported as 0.0.
    """
    labels = [_binary(v) for v in meta_labels]
    preds = [float(v) for v in meta_predictions]
    if len(labels) != len(preds):
        raise ValueError(
            f"meta_labels and meta_predictions differ in length: {len(labels)} vs {len(preds)}"
        )
    if not labels:
        raise ValueError("meta_labels must not be empty")
    if not all(math.isfinite(p) and 0.0 <= p <= 1.0 for p in preds):
        raise ValueError("meta_predictions must be probabilities or decisions in [0, 1]")
    threshold = float(threshold)
    if not 0.0 < threshold <= 1.0:
        raise ValueError(f"threshold must be in (0, 1], got {threshold}")

    act = [p >= threshold for p in preds]
    tp = sum(1 for y, a in zip(labels, act) if a and y == 1)
    fp = sum(1 for y, a in zip(labels, act) if a and y == 0)
    fn = sum(1 for y, a in zip(labels, act) if not a and y == 1)
    tn = len(labels) - tp - fp - fn
    precision = _ratio(tp, tp + fp)
    recall = _ratio(tp, tp + fn)
    positives = tp + fn
    primary_precision = _ratio(positives, len(labels))
    primary_recall = 1.0 if positives else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": _f1(precision, recall),
        "accuracy": (tp + tn) / len(labels),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn,
        "primary_precision": primary_precision,
        "primary_recall": primary_recall,
        "primary_f1": _f1(primary_precision, primary_recall),
    }


def _binary(value: Any) -> int:
    if value in (0, 1):
        return int(value)
    raise ValueError(f"meta_labels must be 0 or 1, got {value!r}")


def _ratio(num: int, den: int) -> float:
    return num / den if den else 0.0


def _f1(precision: float, recall: float) -> float:
    return 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0


# --- strategy risk -----------------------------------------------------------------------


def strategy_failure_probability(
    bet_outcomes: Iterable[float],
    years_elapsed: float,
    target_sharpe: float,
    investor_horizon_years: float,
    *,
    bootstrap_iterations: int = 1000,
    seed: int = 42,
    kde_bandwidth: float | None = None,
) -> dict[str, Any]:
    """Probability that the strategy misses an annualised `target_sharpe` (AFML §15.4).

    Calls `strategy_risk.estimate_strategy_failure_probability` on the per-bet outcomes
    and returns its report; `failure_probability` is its `empirical_failure_probability`,
    the share of bootstrapped precisions at or below the precision the target requires.
    """
    bets = [float(b) for b in bet_outcomes]
    report = dict(
        _sr.estimate_strategy_failure_probability(
            bets,
            years_elapsed=float(years_elapsed),
            target_sharpe=float(target_sharpe),
            investor_horizon_years=float(investor_horizon_years),
            bootstrap_iterations=int(bootstrap_iterations),
            seed=int(seed),
            kde_bandwidth=None if kde_bandwidth is None else float(kde_bandwidth),
        )
    )
    report["failure_probability"] = report["empirical_failure_probability"]
    return report


# --- trial registry ----------------------------------------------------------------------


def config_hash(config: Any) -> str:
    """SHA-256 of the configuration's canonical JSON (sorted keys, no whitespace).

    Two configurations hash equal exactly when they serialise to the same JSON, so key
    order does not matter but `1` and `1.0` differ. Raises `TypeError` for a configuration
    JSON cannot represent.
    """
    try:
        canonical = json.dumps(config, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"config must be JSON-serialisable: {exc}") from exc
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class TrialRegistry:
    """Every configuration tried, persisted to a JSON file so the count survives restarts.

    Record each backtest with `record(config, returns)`. The file is rewritten atomically
    (a temporary file in the same directory, then `os.replace`) and re-read before every
    write, so separate processes or notebook sessions pointed at the same `path` share one
    count. Recording a configuration that is already registered replaces its entry: re-running
    the same notebook does not inflate the trial count. Two processes recording at the same
    instant can still lose one write; the registry is not a database.
    """

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self._path = Path(path)
        self._trials: list[Trial] = self._load()

    @property
    def path(self) -> Path:
        return self._path

    @property
    def trials(self) -> list[Trial]:
        """The registered trials, oldest first."""
        return list(self._trials)

    @property
    def n_trials(self) -> int:
        return len(self._trials)

    def __len__(self) -> int:
        return len(self._trials)

    def reload(self) -> None:
        """Re-read the file, picking up trials other processes have recorded."""
        self._trials = self._load()

    def record(self, config: Any, returns: Iterable[float]) -> Trial:
        """Register one configuration's backtest returns and persist the registry."""
        m = return_moments(returns)
        trial = Trial(
            config_hash=config_hash(config),
            timestamp=datetime.now(UTC).isoformat(timespec="seconds"),
            sharpe=m.sharpe,
            n_obs=m.n_obs,
            skewness=m.skewness,
            kurtosis=m.kurtosis,
            config=config,
        )
        trials = [t for t in self._load() if t.config_hash != trial.config_hash]
        trials.append(trial)
        self._write(trials)
        self._trials = trials
        return trial

    def sharpe_std(self) -> float:
        """Standard deviation (population) of the registered per-period Sharpe ratios."""
        self._require_trials()
        return _population_std([t.sharpe for t in self._trials])

    def expected_max_sharpe(self) -> float:
        """SR0 for this registry's trial count and Sharpe-ratio dispersion."""
        self._require_trials()
        return expected_max_sharpe(self.n_trials, self.sharpe_std())

    def deflated_sharpe_ratio(self, returns: Iterable[float]) -> float:
        """DSR of `returns`, deflated by every trial in the registry.

        Record the evaluated configuration too: it is one of the trials.
        """
        self._require_trials()
        return deflated_sharpe_ratio(returns, n_trials=self.n_trials, sharpe_std=self.sharpe_std())

    def _require_trials(self) -> None:
        if self.n_trials < 2:
            raise ValueError(
                f"the registry at {self._path} holds {self.n_trials} trial(s); "
                "deflating needs at least 2"
            )

    def _load(self) -> list[Trial]:
        if not self._path.exists():
            return []
        try:
            payload = json.loads(self._path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{self._path} is not a trial registry: {exc}") from exc
        if not isinstance(payload, Mapping) or payload.get("schema") != REGISTRY_SCHEMA:
            raise ValueError(f"{self._path} is not a schema-{REGISTRY_SCHEMA} trial registry")
        try:
            return [Trial(**entry) for entry in payload["trials"]]
        except (KeyError, TypeError) as exc:
            raise ValueError(f"{self._path} has a malformed trial entry: {exc}") from exc

    def _write(self, trials: list[Trial]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"schema": REGISTRY_SCHEMA, "trials": [asdict(t) for t in trials]}
        fd, tmp = tempfile.mkstemp(
            prefix=f".{self._path.name}.", suffix=".tmp", dir=self._path.parent
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2, allow_nan=False)
                fh.write("\n")
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, self._path)
        except BaseException:
            try:
                os.unlink(tmp)
            except FileNotFoundError:
                pass
            raise


__all__ = [
    "ReturnMoments",
    "Trial",
    "TrialRegistry",
    "config_hash",
    "deflated_sharpe_ratio",
    "expected_max_sharpe",
    "meta_label_metrics",
    "minimum_track_record_length",
    "probabilistic_sharpe_ratio",
    "return_moments",
    "strategy_failure_probability",
]

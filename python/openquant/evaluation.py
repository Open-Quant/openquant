"""Research evaluation: is a backtest's Sharpe ratio evidence of skill?

An evaluation layer built on the compiled `backtest_stats` and `strategy_risk` modules,
working from a returns series instead of pre-computed moments. It also retains the
original returns for numerical recovery when rounded moments lose uncertainty:


- `probabilistic_sharpe_ratio`, `deflated_sharpe_ratio`, `minimum_track_record_length`
  (Bailey and López de Prado 2012, 2014; AFML Chapter 14, §14.7);
- `meta_label_metrics`: precision, recall and F1 of a meta-labeling overlay against the
  primary model it filters (AFML Chapter 3, §3.6);
- `strategy_failure_probability` (AFML Chapter 15, §15.4);
- `TrialRegistry`: a JSON file keyed by explicit configurations. Re-recording a key
  replaces its entry; the DSR count includes only distinct recorded keys. Include the
  data and protocol choices that distinguish trials; unrecorded choices are not inferred.

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
from decimal import Context, Decimal, localcontext
from fractions import Fraction
from pathlib import Path
from statistics import NormalDist
from statistics import mean as _exact_mean
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


class _ConstantReturns(ValueError):
    """Raised by `return_moments` for a series with zero variance."""


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
    Ratios are computed before rounding the dimensional mean and standard
    deviation; these may underflow to zero for subnormal data even when the
    ratios remain defined. A standard deviation beyond float range is infinity.
    """
    return _return_moments(_as_returns(returns))


def _return_moments(r: list[float]) -> ReturnMoments:
    """Moment calculation for an already validated, materialized return series."""
    n = len(r)
    # Compared by value: a rounded mean (Python < 3.12 sums without compensation) leaves
    # identical values with a tiny non-zero variance and an absurd Sharpe ratio.
    smallest, largest = min(r), max(r)
    if smallest == largest:
        raise _ConstantReturns("returns are constant; the Sharpe ratio is undefined")
    # Translate before centering: subtracting a rounded mean from adjacent floats
    # can erase one side of the distribution. Power-of-two scaling keeps these
    # differences exact while preventing overflow in the subtraction and powers.
    exponent = math.frexp(max(abs(smallest), abs(largest)))[1]
    anchor = math.ldexp(smallest, -exponent)
    offsets = [math.ldexp(x, -exponent) - anchor for x in r]
    center = math.fsum(offsets) / n
    dev = [x - center for x in offsets]
    m2 = math.fsum(x * x for x in dev) / n
    m3 = math.fsum(x**3 for x in dev) / n
    m4 = math.fsum(x**4 for x in dev) / n
    scaled_std = math.sqrt(m2 * n / (n - 1))
    # Sum first to retain small cancelling residuals and subnormal means. Only
    # use the standard library's exact-ratio mean only when summation overflows.
    try:
        mean = math.fsum(r) / n
    except OverflowError:
        mean = _exact_mean(r)
    try:
        std = math.ldexp(scaled_std, exponent)
    except OverflowError:
        std = math.inf
    # Dimensional division preserves tiny cancelling residuals lost by scaling
    # all inputs. Use the scaled ratio when either dimensional moment rounds
    # subnormal or outside the float range, where division would lose accuracy.
    smallest_normal = float.fromhex("0x1.0p-1022")
    if abs(mean) >= smallest_normal and smallest_normal <= std < math.inf:
        sharpe = mean / std
    else:
        scaled_sum = math.fsum(math.ldexp(x, -exponent) for x in r)
        sharpe = (scaled_sum / scaled_std) / n
    return ReturnMoments(
        n_obs=n,
        mean=mean,
        std=std,
        sharpe=sharpe,
        skewness=m3 / m2**1.5,
        kurtosis=m4 / m2**2,
    )


def probabilistic_sharpe_ratio(returns: Iterable[float], benchmark_sr: float = 0.0) -> float:
    """PSR: an approximate one-sided confidence measure against `benchmark_sr`.

    PSR(SR*) = Z[(SR - SR*)·sqrt(T - 1) / sqrt(1 - γ3·SR + (γ4 - 1)/4·SR²)].
    Sharpe ratios are per period and kurtosis is raw. This moment-based normal
    approximation does not correct for serial dependence or give a posterior
    probability of skill.
    Severely cancelled uncertainty is recovered from the original float inputs
    with exact rational arithmetic and 80-digit square roots. This exceptional
    path is slower and can differ from passing rounded moments to the native API.
    """
    r = _as_returns(returns)
    m = _return_moments(r)
    benchmark_sr = _finite(benchmark_sr, "benchmark_sr")
    if _needs_precise_sharpe_variance(m):
        with localcontext(Context(prec=80)):
            difference, variance = _precise_sharpe_difference_and_variance(r, benchmark_sr)
            z = difference * Decimal(m.n_obs - 1).sqrt() / variance.sqrt()
            return math.erfc(-float(z) / math.sqrt(2.0)) / 2.0
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
    """DSR: the PSR against the expected maximum Sharpe under a zero-skill trial model.

    SR0 is an expected value, not an upper bound on chance outcomes. The trial model
    assumes independent configurations; this function does not estimate their
    dependence or an effective trial count. PSR's interpretation limits also apply.

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
    track record of this quality is long enough. For severely cancelled uncertainty,
    the comparison and calculation use the original float inputs rather than the
    rounded `ReturnMoments` fields, as in `probabilistic_sharpe_ratio`.
    """
    r = _as_returns(returns)
    m = _return_moments(r)
    benchmark_sr = _finite(benchmark_sr, "benchmark_sr")
    alpha = _check_alpha(alpha)
    if _needs_precise_sharpe_variance(m):
        with localcontext(Context(prec=80)):
            difference, variance = _precise_sharpe_difference_and_variance(r, benchmark_sr)
            if difference <= 0:
                return math.inf
            # Symmetry avoids rounding 1 - alpha to 1 for very small alpha.
            quantile = Decimal.from_float(NormalDist().inv_cdf(alpha))
            return float(1 + variance * (quantile / difference) ** 2)
    if m.sharpe <= benchmark_sr:
        return math.inf
    return _bs.minimum_track_record_length(m.sharpe, benchmark_sr, m.skewness, m.kurtosis, alpha)


def _needs_precise_sharpe_variance(m: ReturnMoments) -> bool:
    skew_term = m.skewness * m.sharpe
    kurtosis_term = (m.kurtosis - 1.0) / 4.0 * m.sharpe * m.sharpe
    variance = 1.0 - skew_term + kurtosis_term
    # Keep kurtosis and its subtracted 1 separate: a rounded kurtosis near 1
    # can hide substantial uncertainty after multiplication by a large SR².
    scale = 1.0 + abs(skew_term) + (abs(m.kurtosis) + 1.0) / 4.0 * m.sharpe * m.sharpe
    # A cancellation diagnostic, not a universal error bound: recover when the
    # subtraction threatens roughly half of binary64's significant digits.
    return not math.isfinite(variance) or variance <= math.sqrt(math.ulp(1.0)) * scale


def _precise_sharpe_difference_and_variance(
    returns: list[float], benchmark: float
) -> tuple[Decimal, Decimal]:
    """Recover uncertainty from exact float inputs when rounded moments cancel.

    The caller sets a Decimal precision of 80. Exact rational numerators avoid
    both problematic subtractions before the remaining square roots/divisions.
    This information is unavailable to the native moment-only API.
    """
    # Float denominators are powers of two. Align them once, then accumulate
    # exact centered integer powers instead of reducing a Fraction for every
    # observation and power. The common scale cancels from all ratios below.
    ratios = [x.as_integer_ratio() for x in returns]
    denominator_bits = max(denominator.bit_length() for _, denominator in ratios)
    values = [
        numerator << (denominator_bits - denominator.bit_length())
        for numerator, denominator in ratios
    ]
    n = len(values)
    total = sum(values)
    sum2 = sum3 = sum4 = 0
    for value in values:
        centered = n * value - total
        squared = centered * centered
        sum2 += squared
        sum3 += squared * centered
        sum4 += squared * squared
    c = Fraction(n - 1, n)
    sr_squared = Fraction(total * total * (n - 1), sum2)
    squared_sum2 = sum2 * sum2
    a = 1 + Fraction(n * sum4 - squared_sum2, 4 * squared_sum2) * sr_squared
    b = Fraction(n * total * sum3, squared_sum2)
    benchmark_ratio = Fraction(benchmark)

    def _decimal(value: Fraction) -> Decimal:
        return Decimal(value.numerator) / Decimal(value.denominator)

    sr = _decimal(sr_squared).sqrt()
    if total < 0:
        sr = -sr
    if total * benchmark_ratio > 0:
        difference = _decimal(sr_squared - benchmark_ratio**2) / (sr + _decimal(benchmark_ratio))
    else:
        difference = sr - _decimal(benchmark_ratio)
    if b > 0:
        # Variance = A - B sqrt(c). Its rationalized numerator is exact.
        # Pearson's inequality makes it nonnegative; equality is impossible for
        # nonconstant rational inputs because (n-1)/n is not a rational square.
        discriminant = a * a - b * b * c
        variance = _decimal(discriminant) / (_decimal(a) + _decimal(b) * _decimal(c).sqrt())
    else:
        variance = _decimal(a) - _decimal(b) * _decimal(c).sqrt()
    return difference, variance


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
    exponent = math.frexp(max(abs(x) for x in values))[1]
    scaled = [math.ldexp(x, -exponent) for x in values]
    anchor = min(scaled)
    offsets = [x - anchor for x in scaled]
    center = math.fsum(offsets) / len(values)
    variance = math.fsum((x - center) ** 2 for x in offsets) / len(values)
    return math.ldexp(math.sqrt(variance), exponent)


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
    """Conditional bootstrap estimate of missing an annualised target (AFML §15.4).

    The fitted model uses mean winning/non-winning payouts, observed annual bet
    frequency and IID resampling of individual historical outcomes. It does not
    model payout uncertainty, serial dependence, changing frequency or edge decay.
    This is not a calibrated future failure probability for an arbitrary strategy.

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
        """Register one configuration's backtest returns and persist the registry.

        Constant returns (for example all zeros from a configuration that took no bet) have
        no Sharpe ratio, but the configuration was still tried, so it is recorded rather than
        rejected: with Sharpe 0, skewness 0 and kurtosis 3 as bookkeeping placeholders.
        This convention also applies to constant nonzero returns, whose Sharpe is not zero.
        The trial counts towards `n_trials` and its 0 enters `sharpe_std()`. Dropping it
        undercounts trials, but adding it changes both count and dispersion, so the resulting
        deflation hurdle need not increase monotonically.
        """
        r = _as_returns(returns)
        try:
            m = return_moments(r)
        except _ConstantReturns:
            m = ReturnMoments(
                n_obs=len(r), mean=r[0], std=0.0, sharpe=0.0, skewness=0.0, kurtosis=3.0
            )
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

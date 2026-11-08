"""Offline self-financing proportional-fee controls, not a portfolio product."""

import math
from fractions import Fraction

import numpy as np


def _fee(value):
    if (
        isinstance(value, (bool, np.bool_, str, bytes))
        or not np.isscalar(value)
        or np.iscomplexobj(value)
    ):
        raise TypeError("fee must be a real scalar")
    value = float(value)
    if not math.isfinite(value) or not 0 <= value < 1:
        raise ValueError("fee must be finite and in [0,1)")
    return value


def _weights(values, *, fully_invested):
    array = np.asarray(values)
    if array.ndim != 1 or not len(array) or array.dtype.kind not in "iuf":
        raise TypeError("weights must be a nonempty real numeric vector")
    if any(isinstance(x, (bool, np.bool_)) for x in np.asarray(values, dtype=object).flat):
        raise TypeError("boolean weights are not accepted")
    array = array.astype(float)
    if not np.all(np.isfinite(array)) or np.any(array < 0) or np.any(array > 1):
        raise ValueError("weights must be finite, nonnegative and at most one")
    total = math.fsum(array)
    tolerance = 16 * math.ulp(1.0)
    if fully_invested:
        if abs(total - 1) > tolerance:
            raise ValueError("target weights must sum to one within 16 ulps")
        # Explicit roundoff policy, also returned for an independent ledger audit.
        array = array / total
    elif total > 1:
        if total - 1 > tolerance:
            raise ValueError("pretrade weights cannot exceed available wealth")
        array = array / total
    return array


def _fraction_root(w, v, fee):
    """Exact-rational piecewise root for the supplied binary64 coefficients."""
    w, v = list(map(Fraction, w)), list(map(Fraction, v))
    c = Fraction(fee)
    cuts = sorted({Fraction(0), Fraction(1)} | {x / y for x, y in zip(w, v) if y and 0 < x / y < 1})
    for lo, hi in zip(cuts, cuts[1:]):
        mid = (lo + hi) / 2
        signs = [1 if mid * y >= x else -1 for x, y in zip(w, v)]
        numerator = 1 + c * sum(s * x for s, x in zip(signs, w))
        denominator = 1 + c * sum(s * y for s, y in zip(signs, v))
        if denominator <= 0:
            continue
        alpha = numerator / denominator
        if lo <= alpha <= hi:
            return alpha
    raise ArithmeticError("no qualified self-financing root")


def _root(w, v, fee):
    if fee == 0 or np.array_equal(w, v):
        return 1.0, "identity"
    cuts = sorted({0.0, 1.0} | {x / y for x, y in zip(w, v) if y and 0 < x / y < 1})
    for lo, hi in zip(cuts, cuts[1:]):
        mid = (lo + hi) / 2
        signs = [1 if mid * y >= x else -1 for x, y in zip(w, v)]
        alpha = (1 + fee * math.fsum(s * x for s, x in zip(signs, w))) / (
            1 + fee * math.fsum(s * y for s, y in zip(signs, v))
        )
        if lo <= alpha <= hi:
            residual = math.fsum(
                [alpha, fee * math.fsum(abs(alpha * y - x) for x, y in zip(w, v)), -1]
            )
            if abs(residual) <= 8 * math.ulp(1.0):
                return alpha, "piecewise_float"
    return float(_fraction_root(w, v, fee)), "rational_boundary"


def rebalance(pretrade_weights, target_weights, fee):
    """Approximate the exact fee model, retaining normalized inputs and residual.

    Pretrade risky weights sum to at most one; the remainder is cash. Targets
    invest fully. Sums within 16 ulps of one are normalized only as documented
    by _weights. For fee>0.25, a further exact rational simplex normalization
    prevents amplified coefficient roundoff; those exact coefficients are returned.
    Rates apply equally to buys and sales, including initial entry.
    Cash/no-trade is a separate ledger action, not a renormalized target.
    """
    fee = _fee(fee)
    w = _weights(pretrade_weights, fully_invested=False)
    v = _weights(target_weights, fully_invested=True)
    if len(w) != len(v):
        raise ValueError("pretrade and target vectors must have matching lengths")
    exact_coefficients = None
    if fee > 0.25:
        # Near fee=1, even a half-ulp simplex-sum error can move the root greatly.
        # Normalize exactly before solving; returned float fields remain rounded.
        rw, rv = list(map(Fraction, w)), list(map(Fraction, v))
        if sum(rw) > 1:
            total = sum(rw)
            rw = [x / total for x in rw]
        total = sum(rv)
        rv = [x / total for x in rv]
        exact_alpha = _fraction_root(rw, rv, fee)
        exact_trades = [exact_alpha * y - x for x, y in zip(rw, rv)]
        exact_turnover = sum(abs(x) for x in exact_trades)
        alpha = float(exact_alpha)
        trades = np.array([float(x) for x in exact_trades])
        turnover = float(exact_turnover)
        actual_fee = float(Fraction(fee) * exact_turnover)
        w, v = np.array(list(map(float, rw))), np.array(list(map(float, rv)))
        route = "rational_stress"
        normalization = "exact_rational_stress"
        exact_coefficients = {
            "pretrade_weights": list(map(str, rw)),
            "target_weights": list(map(str, rv)),
            "wealth_fraction": str(exact_alpha),
        }
    else:
        alpha, route = _root(w, v, fee)
        trades = alpha * v - w
        turnover = math.fsum(abs(x) for x in trades)
        actual_fee = fee * turnover
        normalization = "binary64_roundoff"
    residual = math.fsum([alpha, actual_fee, -1])
    if not math.isfinite(alpha) or not 0 < alpha <= 1:
        raise ArithmeticError("invalid postfee wealth")
    return {
        "wealth_fraction": alpha,
        "fee_fraction": actual_fee,
        "trade_fractions": trades.tolist(),
        "traded_fraction": turnover,
        "self_financing_residual": residual,
        "residual_condition_indicator": abs(residual) / (1 - fee),
        "pretrade_weights": w.tolist(),
        "target_weights": v.tolist(),
        "route": route,
        "normalization": normalization,
        "exact_coefficients": exact_coefficients,
    }


POLICIES = ("cash", "buy_hold", "daily", "periodic_21", "drift_02", "drift_10")


def simulate(returns, *, policy, fee, accounting="self_financing", keep_records=False):
    """Run a predictable long-only policy; no final liquidation or cash interest."""
    if policy not in POLICIES:
        raise ValueError("unknown fixed policy")
    if accounting not in ("self_financing", "toy_debit"):
        raise ValueError("unknown accounting convention")
    fee = _fee(fee)
    values = np.asarray(returns)
    if values.ndim != 2 or not all(values.shape) or values.dtype.kind not in "iuf":
        raise TypeError("returns must be a nonempty numeric matrix")
    if any(isinstance(x, (bool, np.bool_)) for x in np.asarray(returns, dtype=object).flat):
        raise TypeError("boolean returns are not accepted")
    values = values.astype(float)
    if not np.all(np.isfinite(values)) or np.any(values <= -1):
        raise ValueError("simple returns must be finite and greater than -1")
    target = np.full(values.shape[1], 1 / values.shape[1])
    weights = np.zeros(values.shape[1])
    cash_weight = 1.0
    wealth = 1.0
    total_fees = 0.0
    total_turnover = 0.0
    count = 0
    nonzero_trades = 0
    max_gap = 0.0
    records = []
    for t, row in enumerate(values):
        trade = policy != "cash" and (
            t == 0
            or policy == "daily"
            or (policy == "periodic_21" and t % 21 == 0)
            or (
                policy.startswith("drift_")
                and np.max(np.abs(weights - target)) > (0.02 if policy == "drift_02" else 0.10)
            )
        )
        before = wealth
        before_weights = weights.copy()
        alpha = 1.0
        debit = 0.0
        turnover = 0.0
        gap = 0.0
        if trade:
            if accounting == "self_financing":
                result = rebalance(weights, target, fee)
                alpha = result["wealth_fraction"]
                debit = result["fee_fraction"]
                turnover = result["traded_fraction"]
                gap = -result["self_financing_residual"]
                weights = np.array(result["target_weights"])
            else:
                # Same declared proxy as the rolling-risk lab, not its solver.
                turnover = math.fsum(abs(x) for x in target - weights)
                debit = fee * turnover
                alpha = 1 - debit
                if alpha <= 0:
                    raise ArithmeticError("toy debit exhausts wealth")
                gap = debit - fee * math.fsum(abs(x) for x in alpha * target - weights)
                weights = target.copy()
            cash_weight = 0.0
            wealth *= alpha
            total_fees += before * debit
            total_turnover += turnover
            count += 1
            nonzero_trades += int(turnover > 0)
            max_gap = max(max_gap, abs(gap))
        # Sum positive holdings and explicit cash, avoiding near-total-loss cancellation.
        growth = math.fsum([cash_weight, *list(weights * (1 + row))])
        wealth *= growth
        if not math.isfinite(wealth) or wealth <= 0:
            raise ArithmeticError("wealth overflow or underflow")
        weights = weights * (1 + row) / growth
        cash_weight /= growth
        if keep_records:
            records.append(
                {
                    "t": t,
                    "trade": trade,
                    "pretrade_weights": before_weights.tolist(),
                    "wealth_before": before,
                    "wealth_fraction": alpha,
                    "fee_fraction": debit,
                    "traded_fraction": turnover,
                    "fee_model_gap_fraction": gap,
                    "wealth_after": wealth,
                    "ending_weights": weights.tolist(),
                    "cash_weight": cash_weight,
                }
            )
    result = {
        "terminal_wealth": wealth,
        "terminal_log_wealth": math.log(wealth),
        "total_fees": total_fees,
        "total_traded_fraction": total_turnover,
        "rebalance_attempt_count": count,
        "nonzero_trade_count": nonzero_trades,
        "max_fee_model_gap_fraction": max_gap,
    }
    if keep_records:
        result["records"] = records
    return result


def _integer(value, name, minimum):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    value = int(value)
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


def run_lab(*, horizon=252, repetitions=128, fee_rates=(0, 0.0001, 0.001, 0.01), seed=20261108):
    """Fixed policy grid; no fitted/selected policy and no investment inference."""
    import hashlib
    import time
    from pathlib import Path

    start = time.perf_counter()
    horizon = _integer(horizon, "horizon", 1)
    repetitions = _integer(repetitions, "repetitions", 1)
    seed = _integer(seed, "seed", 0)
    fee_rates = tuple(_fee(c) for c in fee_rates)
    if not fee_rates or len(set(fee_rates)) != len(fee_rates):
        raise ValueError("fee grid must be nonempty without duplicates")
    if horizon > 10000 or repetitions > 10000 or horizon * repetitions * len(fee_rates) > 2000000:
        raise ValueError("study budget exceeds explicit offline limit")
    paths = []
    a = 0.02
    alternating = np.array([[a, -a / (1 + a)], [-a / (1 + a), a]])
    paths.append(("alternating", 0, alternating[np.arange(horizon) % 2]))
    paths.append(("trend", 0, np.tile([0.002, -0.001], (horizon, 1))))
    for family, index in [("iid_zero_mean", 0), ("common_zero_mean", 1)]:
        for rep in range(repetitions):
            rng = np.random.default_rng(np.random.SeedSequence([seed, index, rep]))
            shape = (horizon, 2) if index == 0 else (horizon, 1)
            values = 0.02 * (2 * rng.integers(0, 2, size=shape) - 1)
            if index == 1:
                values = np.repeat(values, 2, axis=1)
            paths.append((family, rep, values))
    path_records = []
    reports = []
    for family, rep, values in paths:
        digest = hashlib.sha256(np.asarray(values, dtype="<f8").tobytes()).hexdigest()
        path_records.append(
            {"family": family, "replication": rep, "sha256": digest, "returns": values.tolist()}
        )
        for fee in fee_rates:
            for policy in POLICIES:
                for accounting in ("self_financing", "toy_debit"):
                    row = {
                        "family": family,
                        "replication": rep,
                        "path_sha256": digest,
                        "fee": fee,
                        "policy": policy,
                        "accounting": accounting,
                    }
                    try:
                        row.update(
                            status="ok",
                            **simulate(values, policy=policy, fee=fee, accounting=accounting),
                        )
                    except ArithmeticError as exc:
                        row.update(status="failed", reason=str(exc))
                    reports.append(row)
    return {
        "lab": "self_financing",
        "schema_version": 1,
        "synthetic_only": True,
        "config": {
            "horizon": horizon,
            "repetitions": repetitions,
            "fee_rates": list(fee_rates),
            "seed": seed,
        },
        "provenance": {
            "numpy": np.__version__,
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
        "performance": {"wall_seconds": time.perf_counter() - start},
        "paths": path_records,
        "reports": reports,
    }


def main():
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("output already exists; preserve it and choose a new path")
    report = run_lab()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()

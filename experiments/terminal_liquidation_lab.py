"""Research-only finite-horizon inventory versus cash realization controls."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments import self_financing_lab as financing  # noqa: E402

POLICIES = ("cash", "buy_hold", "daily")


def _nonnegative(value, name):
    if (
        isinstance(value, (bool, np.bool_, str, bytes))
        or not np.isscalar(value)
        or np.iscomplexobj(value)
    ):
        raise TypeError(f"{name} must be a real scalar")
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return value


def liquidate(holdings, *, cash, exit_fee):
    """Sell nonnegative marked risky notionals once; cash is never charged.

    All amounts use units of initial capital. Exit is at the terminal mark with
    a separately declared proportional sale charge, no extra holding period.
    """
    fee = financing._fee(exit_fee)
    cash = _nonnegative(cash, "cash")
    h = np.asarray(holdings)
    if h.ndim != 1 or not len(h) or h.dtype.kind not in "iuf":
        raise TypeError("holdings must be a nonempty real numeric vector")
    if any(isinstance(x, (bool, np.bool_)) for x in np.asarray(holdings, dtype=object).flat):
        raise TypeError("boolean holdings are not accepted")
    h = h.astype(float)
    if not np.isfinite(h).all() or np.any(h < 0):
        raise ValueError("holdings must be finite and nonnegative")
    try:
        risky = math.fsum(h)
        mark = math.fsum([cash, risky])
        cost = fee * risky
        realized = math.fsum([cash, (1 - fee) * risky])
    except OverflowError as exc:
        raise ArithmeticError("terminal inventory overflow") from exc
    if not all(math.isfinite(x) for x in (risky, mark, cost, realized)):
        raise ArithmeticError("terminal inventory overflow")
    if mark > 0 and realized <= 0:
        raise ArithmeticError("terminal wealth underflow")
    return {
        "mark_to_market_wealth": mark,
        "terminal_risky_notional": risky,
        "terminal_cash_before_exit": cash,
        "exit_traded_notional": risky,
        "exit_fee_amount": cost,
        "liquidated_wealth": realized,
        "terminal_cash_after_exit": realized,
        "terminal_risky_notional_after_exit": 0.0,
        "terminal_accounting_residual": math.fsum([realized, cost, -mark]),
    }


def simulate(returns, *, policy, entry_fee, trade_fee, exit_fee, keep_records=False):
    """Fixed equal-weight policies with initial cash=1 and zero cash yield."""
    if policy not in POLICIES:
        raise ValueError("unknown fixed policy")
    entry_fee, trade_fee, exit_fee = map(financing._fee, (entry_fee, trade_fee, exit_fee))
    values = np.asarray(returns)
    if values.ndim != 2 or not all(values.shape) or values.dtype.kind not in "iuf":
        raise TypeError("returns must be a nonempty real numeric matrix")
    if any(isinstance(x, (bool, np.bool_)) for x in np.asarray(returns, dtype=object).flat):
        raise TypeError("boolean returns are not accepted")
    values = values.astype(float)
    if not np.isfinite(values).all() or np.any(values <= -1):
        raise ValueError("simple returns must be finite and greater than -1")
    target = np.full(values.shape[1], 1 / values.shape[1])
    holdings = np.zeros(values.shape[1])
    cash = 1.0
    entry_cost = ongoing_cost = traded = 0.0
    records = []
    for t, row in enumerate(values):
        before = math.fsum([cash, *holdings])
        charge = turnover = 0.0
        trade = policy != "cash" and (t == 0 or policy == "daily")
        if trade:
            step = financing.rebalance(
                holdings / before, target, entry_fee if t == 0 else trade_fee
            )
            charge = before * step["fee_fraction"]
            turnover = before * step["traded_fraction"]
            holdings = before * step["wealth_fraction"] * np.array(step["target_weights"])
            cash = 0.0
            if t == 0:
                entry_cost += charge
            else:
                ongoing_cost += charge
            traded += turnover
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            holdings *= 1 + row
        try:
            after = math.fsum([cash, *holdings])
        except OverflowError as exc:
            raise ArithmeticError("wealth overflow") from exc
        if not math.isfinite(after) or after <= 0:
            raise ArithmeticError("wealth overflow or underflow")
        if keep_records:
            records.append(
                {
                    "t": t,
                    "trade": trade,
                    "wealth_before": before,
                    "fee_amount": charge,
                    "traded_notional": turnover,
                    "wealth_after": after,
                    "ending_holdings": holdings.tolist(),
                    "cash": cash,
                }
            )
    result = liquidate(holdings, cash=cash, exit_fee=exit_fee)
    mark, realized = result["mark_to_market_wealth"], result["liquidated_wealth"]
    result.update(
        entry_fees=entry_cost,
        ongoing_fees=ongoing_cost,
        total_fees=math.fsum([entry_cost, ongoing_cost, result["exit_fee_amount"]]),
        total_traded_notional=math.fsum([traded, result["exit_traded_notional"]]),
        cash_benchmark_wealth=1.0,
        mark_to_market_excess_vs_cash=mark - 1,
        liquidated_excess_vs_cash=realized - 1,
        cash_comparison_reversal=bool(mark > 1 and realized < 1),
        mark_to_market_log_wealth=math.log(mark),
        liquidated_log_wealth=math.log(realized),
    )
    if keep_records:
        result["records"] = records
    return result


def run_lab(
    *,
    horizon=21,
    repetitions=32,
    entry_fee=0.001,
    trade_fee=0.001,
    exit_fee_rates=(0, 0.001, 0.03),
    seed=20261109,
):
    """Paired prespecified paths and fees; no optimizer or policy selection."""
    start = time.perf_counter()
    horizon = financing._integer(horizon, "horizon", 1)
    repetitions = financing._integer(repetitions, "repetitions", 1)
    seed = financing._integer(seed, "seed", 0)
    entry_fee, trade_fee = map(financing._fee, (entry_fee, trade_fee))
    exit_fee_rates = tuple(financing._fee(x) for x in exit_fee_rates)
    if not exit_fee_rates or len(set(exit_fee_rates)) != len(exit_fee_rates):
        raise ValueError("exit fee grid must be nonempty without duplicates")
    work = horizon * (4 + 2 * repetitions) * len(POLICIES) * len(exit_fee_rates)
    if horizon > 10000 or repetitions > 10000 or work > 2000000:
        raise ValueError("study budget exceeds explicit offline limit")
    alternating = np.array([[0.02, -0.02 / 1.02], [-0.02 / 1.02, 0.02]])
    paths = [
        ("flat", 0, np.zeros((horizon, 2))),
        ("common_small_gain", 0, np.tile([0.001, 0.001], (horizon, 1))),
        ("trend", 0, np.tile([0.002, -0.001], (horizon, 1))),
        ("alternating", 0, alternating[np.arange(horizon) % 2]),
    ]
    for family, index in (("iid_zero_mean", 0), ("common_zero_mean", 1)):
        for rep in range(repetitions):
            rng = np.random.default_rng(np.random.SeedSequence([seed, index, rep]))
            values = 0.02 * (2 * rng.integers(0, 2, size=(horizon, 2 if index == 0 else 1)) - 1)
            if index == 1:
                values = np.repeat(values, 2, axis=1)
            paths.append((family, rep, values))
    path_records, reports = [], []
    for family, rep, values in paths:
        digest = hashlib.sha256(np.asarray(values, dtype="<f8").tobytes()).hexdigest()
        path_records.append(
            {"family": family, "replication": rep, "sha256": digest, "returns": values.tolist()}
        )
        for fee in exit_fee_rates:
            for policy in POLICIES:
                row = {
                    "family": family,
                    "replication": rep,
                    "path_sha256": digest,
                    "policy": policy,
                    "entry_fee": entry_fee,
                    "trade_fee": trade_fee,
                    "exit_fee": fee,
                }
                try:
                    row.update(
                        status="ok",
                        **simulate(
                            values,
                            policy=policy,
                            entry_fee=entry_fee,
                            trade_fee=trade_fee,
                            exit_fee=fee,
                        ),
                    )
                except ArithmeticError as exc:
                    row.update(status="failed", reason=str(exc))
                reports.append(row)
    return {
        "lab": "terminal_liquidation",
        "schema_version": 1,
        "synthetic_only": True,
        "config": {
            "horizon": horizon,
            "repetitions": repetitions,
            "entry_fee": entry_fee,
            "trade_fee": trade_fee,
            "exit_fee_rates": list(exit_fee_rates),
            "seed": seed,
        },
        "provenance": {
            "numpy": np.__version__,
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "dependency_sha256": {
                "self_financing_lab.py": hashlib.sha256(
                    Path(financing.__file__).read_bytes()
                ).hexdigest()
            },
        },
        "performance": {"wall_seconds": time.perf_counter() - start},
        "paths": path_records,
        "reports": reports,
    }


def main():
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

"""Gross sign evidence versus arithmetic payoff under bounded stopping."""

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
from experiments import sequential_sign_lab as sequential  # noqa: E402

DEFAULT_MAPS = tuple(
    {"id": f"p60-w{win}-l{loss}-c{cost}", "p": 0.6, "win": win, "loss": loss, "cost": cost}
    for win, loss in ((1, 1), (1, 2), (2, 1))
    for cost in (0.0, 0.25)
) + tuple(
    {"id": f"p85-w02-l1-c{cost}", "p": 0.85, "win": 0.2, "loss": 1.0, "cost": cost}
    for cost in (0.0, 0.25)
)


def _magnitude(value, name, *, allow_zero=False):
    if (
        isinstance(value, (bool, np.bool_, str, bytes))
        or np.iscomplexobj(value)
        or not np.isscalar(value)
    ):
        raise TypeError(f"{name} must be a real scalar")
    value = float(value)
    if not math.isfinite(value) or (value < 0 if allow_zero else value <= 0):
        raise ValueError(f"{name} must be finite and {'nonnegative' if allow_zero else 'positive'}")
    return value


def economics(p, win, loss, cost=0.0):
    p = sequential._probability(p, "p", closed=True)
    win = _magnitude(win, "win")
    loss = _magnitude(loss, "loss")
    cost = _magnitude(cost, "cost", allow_zero=True)
    gross = p * win - (1 - p) * loss
    net = gross - cost
    if not math.isfinite(gross) or not math.isfinite(net):
        raise ValueError("payoff expectation exceeds finite float range")
    return {
        "p": p,
        "win": win,
        "loss": loss,
        "cost": cost,
        "expected_gross_per_row": gross,
        "expected_net_per_row": net,
        "gross_positive_probability": p,
        "net_positive_probability": p if win > cost else 0.0,
        "net_zero_probability": p if win == cost else 0.0,
    }


def terminal_distribution(boundary, p):
    """IID terminal (time,successes,crossed,mass) law, retaining no-cross paths."""
    boundary = np.asarray(boundary)
    if boundary.ndim != 1 or len(boundary) < 2 or boundary.dtype.kind not in "iu":
        raise ValueError("boundary must include unused time zero and integer counts")
    horizon = len(boundary) - 1
    if (
        horizon > sequential.MAX_HORIZON
        or np.any(boundary < 0)
        or np.any(boundary > np.arange(horizon + 1) + 1)
    ):
        raise ValueError("invalid or unsupported boundary")
    p = sequential._probability(p, "p", closed=True)
    alive = np.zeros(horizon + 1)
    alive[0] = 1.0
    terminal = []
    for t in range(1, horizon + 1):
        next_alive = alive * (1 - p)
        next_alive[1:] += alive[:-1] * p
        alive = next_alive
        for s in range(t + 1):
            crossed = s >= boundary[t]
            if alive[s] and (crossed or t == horizon):
                terminal.append(
                    {"time": t, "successes": s, "crossed": bool(crossed), "mass": float(alive[s])}
                )
                alive[s] = 0.0
    if abs(math.fsum(r["mass"] for r in terminal) - 1) > 1e-10:
        raise ArithmeticError("terminal law failed mass conservation")
    return terminal


def terminal_payoffs(terminal, *, win, loss, cost):
    win = _magnitude(win, "win")
    loss = _magnitude(loss, "loss")
    cost = _magnitude(cost, "cost", allow_zero=True)
    maximum_time = max(r["time"] for r in terminal)
    if not math.isfinite(maximum_time * (win + loss + cost)):
        raise ValueError("payoff magnitude/time combination exceeds supported float range")
    values = [
        r["successes"] * win - (r["time"] - r["successes"]) * loss - r["time"] * cost
        for r in terminal
    ]
    crossed_mass = math.fsum(r["mass"] for r in terminal if r["crossed"])
    not_crossed_mass = math.fsum(r["mass"] for r in terminal if not r["crossed"])
    crossed_contribution = math.fsum(
        r["mass"] * y for r, y in zip(terminal, values, strict=True) if r["crossed"]
    )
    not_crossed_contribution = math.fsum(
        r["mass"] * y for r, y in zip(terminal, values, strict=True) if not r["crossed"]
    )
    return {
        "expected_net_payoff": math.fsum(
            r["mass"] * y for r, y in zip(terminal, values, strict=True)
        ),
        "probability_net_loss": math.fsum(
            r["mass"] for r, y in zip(terminal, values, strict=True) if y < 0
        ),
        "expected_rows": math.fsum(r["mass"] * r["time"] for r in terminal),
        "crossing_probability": crossed_mass,
        "no_crossing_probability": not_crossed_mass,
        "crossed_net_contribution": crossed_contribution,
        "no_cross_net_contribution": not_crossed_contribution,
        "conditional_crossed_mean_net_descriptive": crossed_contribution / crossed_mass
        if crossed_mass
        else None,
        "conditional_no_cross_mean_net_descriptive": not_crossed_contribution / not_crossed_mass
        if not_crossed_mass
        else None,
    }


def _sample_summary(values):
    values = np.asarray(values, dtype=float)
    scale = float(np.max(np.abs(values))) or 1.0
    scaled = values / scale
    return {
        "mean": float(scaled.mean() * scale),
        "mcse": float((scaled.std(ddof=1) / math.sqrt(len(values))) * scale)
        if len(values) > 1
        else None,
        "probability_net_loss": float(np.mean(values < 0)),
        "values": values.tolist(),
    }


def run_lab(
    *,
    horizons=(64, 256),
    repetitions=5000,
    seed=23,
    alpha=0.05,
    qs=(0.55, 0.65, 0.8),
    maps=DEFAULT_MAPS,
):
    start = time.perf_counter()
    horizons = tuple(sequential._integer(h, "horizon") for h in horizons)
    if (
        not horizons
        or len(set(horizons)) != len(horizons)
        or max(horizons) > sequential.MAX_HORIZON
    ):
        raise ValueError("horizons must be unique supported positive horizons")
    repetitions = sequential._integer(repetitions, "repetitions")
    seed = sequential._integer(seed, "seed", 0)
    alpha = sequential._probability(alpha, "alpha")
    qs = sequential._mixture(qs)
    normalized = []
    for m in maps:
        if (
            not isinstance(m, dict)
            or set(m) != {"id", "p", "win", "loss", "cost"}
            or not isinstance(m["id"], str)
            or not m["id"]
        ):
            raise ValueError("maps require id,p,win,loss,cost")
        e = economics(m["p"], m["win"], m["loss"], m["cost"])
        if not math.isfinite(max(horizons) * (e["win"] + e["loss"] + e["cost"])):
            raise ValueError("payoff magnitude/time combination exceeds supported float range")
        normalized.append({"id": m["id"], **{k: e[k] for k in ("p", "win", "loss", "cost")}})
    if not normalized or len({m["id"] for m in normalized}) != len(normalized):
        raise ValueError("maps must be nonempty with unique ids")
    probabilities = sorted({m["p"] for m in normalized})
    groups = []
    for horizon in horizons:
        boundaries = sequential.crossing_boundaries(horizon, alpha=alpha, qs=qs)
        boundary = boundaries["fixed_mixture"]
        for index, p in enumerate(probabilities):
            rng = np.random.default_rng(np.random.SeedSequence([seed, horizon, index]))
            paths = (rng.random((repetitions, horizon)) < p).astype(np.uint8)
            first = sequential.evaluate_paths(paths, boundaries)["fixed_mixture"]
            tau = np.where(first > 0, first, horizon)
            cumulative = np.cumsum(paths, axis=1)
            stopped_successes = cumulative[np.arange(repetitions), tau - 1]
            final_successes = cumulative[:, -1]
            terminal = terminal_distribution(boundary, p)
            fixed_terminal = terminal_distribution(np.arange(horizon + 1) + 1, p)
            reference = sequential.crossing_probability(
                boundary, p_after_zero=p, p_after_one=p, initial_positive=p
            )
            scenarios = []
            for m in normalized:
                if m["p"] != p:
                    continue
                e = economics(p, m["win"], m["loss"], m["cost"])
                exact = terminal_payoffs(terminal, win=e["win"], loss=e["loss"], cost=e["cost"])
                fixed_exact = terminal_payoffs(
                    fixed_terminal, win=e["win"], loss=e["loss"], cost=e["cost"]
                )
                stopped = (
                    stopped_successes * e["win"]
                    - (tau - stopped_successes) * e["loss"]
                    - tau * e["cost"]
                )
                fixed = (
                    final_successes * e["win"]
                    - (horizon - final_successes) * e["loss"]
                    - horizon * e["cost"]
                )
                scenarios.append(
                    {
                        "id": m["id"],
                        "economics": e,
                        "stopped": {
                            "finite_sum": exact,
                            "wald_expected_net_payoff": e["expected_net_per_row"]
                            * reference["expected_observed_rows"],
                            "sample": _sample_summary(stopped),
                        },
                        "fixed_horizon": {
                            "finite_sum": fixed_exact,
                            "iid_expected_net_payoff": horizon * e["expected_net_per_row"],
                            "sample": _sample_summary(fixed),
                        },
                    }
                )
            groups.append(
                {
                    "horizon": horizon,
                    "p": p,
                    "sample_sha256": hashlib.sha256(
                        json.dumps(
                            {"shape": list(paths.shape), "dtype": "uint8"}, sort_keys=True
                        ).encode()
                        + paths.tobytes()
                    ).hexdigest(),
                    "gross_sign_first_crossing_rows": first.tolist(),
                    "observed_rows": tau.tolist(),
                    "stopped_successes": stopped_successes.tolist(),
                    "fixed_successes": final_successes.tolist(),
                    "gross_sign_reference": reference,
                    "terminal_distribution": terminal,
                    "scenarios": scenarios,
                }
            )
    return {
        "lab": "payoff_target",
        "schema_version": 1,
        "synthetic_only": True,
        "config": {
            "horizons": list(horizons),
            "repetitions": repetitions,
            "seed": seed,
            "alpha": alpha,
            "qs": list(qs),
            "maps": normalized,
        },
        "provenance": {
            "numpy": np.__version__,
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "dependency_sha256": {
                "sequential_sign_lab.py": hashlib.sha256(
                    Path(sequential.__file__).read_bytes()
                ).hexdigest()
            },
        },
        "performance": {"wall_seconds": time.perf_counter() - start},
        "groups": groups,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = run_lab()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()

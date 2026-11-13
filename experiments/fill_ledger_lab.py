"""Synthetic accounting controls for partial fills and incomplete terminal sales."""

from __future__ import annotations

import hashlib
import itertools
import platform
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from fractions import Fraction as F
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments import fill_ledger as ledger  # noqa: E402


def _at(second: int) -> datetime:
    return datetime(2026, 1, 1, tzinfo=UTC) + timedelta(seconds=second)


def _decision(name: str, shares: int, second: int = 0) -> ledger.Decision:
    return ledger.Decision(name, _at(second), _at(second), _at(second), _at(3), F(shares))


def _quote(second: int, mark: str, bid: str, ask: str, volume: str) -> ledger.MarketBucket:
    return ledger.MarketBucket(_at(second), F(mark), F(bid), F(ask), F(volume))


def _encode(value: Any) -> Any:
    if isinstance(value, F):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {key: _encode(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_encode(item) for item in value]
    return value


def _grid(values: Any, name: str, upper: F | None = None) -> list[str]:
    if not isinstance(values, (list, tuple)):
        raise ValueError(f"{name} must be an ordered array (list or tuple)")
    try:
        converted = [ledger._number(value, name) for value in values]
    except TypeError as exc:
        raise ValueError(f"{name} must be a nonempty grid") from exc
    if not converted:
        raise ValueError(f"{name} must be nonempty")
    if upper is not None and any(value > upper for value in converted):
        raise ValueError(f"{name} must be at most {upper}")
    return [str(value) for value in converted]


def _controls() -> tuple[dict[str, Any], dict[str, Any]]:
    state = ledger.InitialState(_at(0), F(100), F(0), F(10))
    entry = _quote(1, "11", "9.5", "10", "10")
    exit_ = _quote(2, "11", "10.5", "11", "10")
    hand = ledger.run_ledger(
        state,
        [_decision("buy", 2)],
        [entry],
        fixed_fee=1,
        terminal_policy="liquidate_at_declared_bid",
        terminal_bucket=exit_,
    )
    shared = ledger.run_ledger(
        state, [_decision("a", 6), _decision("b", 6)], [entry], participation="0.5"
    )
    partial = ledger.run_ledger(
        state,
        [_decision("buy", 2)],
        [entry],
        terminal_policy="liquidate_at_declared_bid",
        terminal_bucket=_quote(2, "11", "10.5", "11", "1"),
    )
    zero = ledger.run_ledger(
        state, [_decision("buy", 2)], [_quote(1, "11", "9.5", "10", "0")], fixed_fee=1
    )
    hand_values = {
        "entry_cash": str(hand.rows[0].cash),
        "entry_equity": str(hand.rows[0].equity),
        "terminal_cash": str(hand.terminal.cash),
    }
    shared_values = {
        "filled": str(sum(abs(fill.shares) for fill in shared.fills)),
        "remainders": [str(order.remaining) for order in shared.orders],
    }
    partial_values = {
        "inventory": str(partial.terminal.inventory),
        "cash": str(partial.terminal.cash),
        "marked_equity": str(partial.terminal.marked_equity),
        "completed": partial.terminal.completed,
    }
    controls: dict[str, Any] = {
        "hand_entry_and_exit": {
            "observed": hand_values,
            "expected": {"entry_cash": "79", "entry_equity": "101", "terminal_cash": "99"},
        },
        "shared_capacity": {
            "observed": shared_values,
            "expected": {"filled": "5", "remainders": ["1", "6"]},
        },
        "partial_terminal": {
            "observed": partial_values,
            "expected": {
                "inventory": "1",
                "cash": "181/2",
                "marked_equity": "203/2",
                "completed": False,
            },
        },
        "zero_fill_zero_fee": {"observed": str(zero.terminal.cash), "expected": "100"},
    }
    for control in controls.values():
        control["passed"] = control["observed"] == control["expected"]
    negatives = {
        "reuse_bucket_capacity_per_order": {
            "wrong_total_fill": "10",
            "correct_total_fill": shared_values["filled"],
            "detected": shared_values["filled"] != "10",
        },
        "erase_terminal_residual": {
            "wrong_inventory": "0",
            "correct_inventory": partial_values["inventory"],
            "detected": partial_values["inventory"] != "0",
        },
        "treat_marked_equity_as_terminal_cash": {
            "wrong_cash": "101",
            "correct_cash": hand_values["terminal_cash"],
            "detected": hand_values["terminal_cash"] != "101",
        },
        "charge_fee_without_fill": {
            "wrong_cash": "99",
            "correct_cash": str(zero.terminal.cash),
            "detected": zero.terminal.cash != 99,
        },
    }
    return controls, negatives


def run_lab(
    *,
    participations: Any = (0, 0.25, 0.5, 1),
    fixed_fees: Any = (0, 1),
    fee_rates: Any = (0, 0.01),
    terminal_volumes: Any = (0, 1, 20),
) -> dict[str, Any]:
    """Retain all prespecified liquidity/cost scenarios, without policy selection."""
    start = time.perf_counter()
    config = {
        "participations": _grid(participations, "participations", F(1)),
        "fixed_fees": _grid(fixed_fees, "fixed_fees"),
        "fee_rates": _grid(fee_rates, "fee_rates"),
        "terminal_volumes": _grid(terminal_volumes, "terminal_volumes"),
    }
    initial = ledger.InitialState(_at(0), F(500), F(0), F(10))
    decisions = [_decision("first", 6), _decision("second", 6, 1)]
    quotes = [
        _quote(1, "10", "9.5", "10", "10"),
        _quote(2, "11", "10", "10.5", "0"),
        _quote(3, "9", "9", "9.5", "8"),
    ]
    scenarios = []
    for p, fixed, rate, terminal in itertools.product(*config.values()):
        result = ledger.run_ledger(
            initial,
            decisions,
            quotes,
            participation=p,
            fixed_fee=fixed,
            fee_rate=rate,
            terminal_policy="liquidate_at_declared_bid",
            terminal_bucket=_quote(4, "9", "8.5", "9.5", terminal),
        )
        scenarios.append(
            {
                "assumptions": {
                    "participation": p,
                    "fixed_fee": fixed,
                    "fee_rate": rate,
                    "terminal_volume": terminal,
                },
                "ledger": _encode(asdict(result)),
                "all_obligations_completed": result.decisions_completed
                and result.terminal.completed,
            }
        )
    controls, negative = _controls()
    source = Path(__file__).resolve()
    try:
        git_sha = subprocess.check_output(
            ["git", "-C", str(source.parent), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        git_sha = "unavailable"
    return {
        "lab": "fill_ledger",
        "schema_version": 1,
        "synthetic_only": True,
        "numeric_encoding": "exact rational strings from decimal inputs",
        "config": config,
        "scenarios": scenarios,
        "controls": controls,
        "negative_controls": negative,
        "units": {
            "shares": "SYNTH shares",
            "cash": "USD",
            "price": "USD/share",
            "time": "aware UTC datetime",
            "fee_rate": "fraction of executed notional per nonzero fill",
        },
        "limitations": [
            "Original synthetic accounting controls, not calibrated fills or investment evidence.",
            "No stochastic queueing, borrowing, shorts, funding, taxes or market impact.",
            "Cash/inventory constraints reject candidate fills instead of sizing them down.",
            "Same-timestamp fills are excluded; declared quotes use shared gross volume capacity.",
            "Exact fractions are accounting on declared decimal inputs, not market precision.",
            "Mark-only completion means that valuation policy was met, not cash liquidation.",
            "No cross-scenario ranking when completion and terminal obligations differ.",
            "Trusted local inputs only; no computational resource sandbox.",
        ],
        "provenance": {
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "dependency_sha256": {
                "fill_ledger.py": hashlib.sha256(Path(ledger.__file__).read_bytes()).hexdigest()
            },
            "git_sha": git_sha,
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        "performance": {"wall_seconds": time.perf_counter() - start},
    }

"""Original cash-handoff controls, not a CPCV execution adapter or return model."""

from __future__ import annotations

import hashlib
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


def cash_handoff(result: ledger.LedgerResult, *, at: datetime, mark: F) -> ledger.InitialState:
    """Carry realized cash to a new segment only after complete cash liquidation.

    This conservative helper refuses unfinished decision obligations. It transfers
    no stock, invents no cash deposit and makes no statement about independence,
    model training, selection, CPCV path validity, financing or performance.
    It requires genuine, unmodified run_ledger output: terminal consistency checks
    do not re-audit fill history or validate a manually constructed LedgerResult.
    """
    if not isinstance(result, ledger.LedgerResult):
        raise ValueError("result must be a LedgerResult")
    if not result.decisions_completed or any(order.remaining for order in result.orders):
        raise ValueError("unfinished decisions require an explicit transition policy")
    terminal = result.terminal
    if terminal.policy != "liquidate_at_declared_bid" or not terminal.completed:
        raise ValueError("cash handoff requires completed liquidation")
    if terminal.inventory != 0:
        raise ValueError("cash handoff cannot erase residual inventory")
    if not result.rows:
        raise ValueError("terminal summary has no ledger row")
    last = result.rows[-1]
    if (
        last.phase != "terminal"
        or last.cash != terminal.cash
        or last.inventory != 0
        or last.equity != terminal.cash
        or terminal.marked_equity != terminal.cash
    ):
        raise ValueError("terminal summary does not reconcile to its cash ledger row")
    next_state = ledger.InitialState(
        at, terminal.cash, F(0), mark, result.initial.instrument, result.initial.currency
    )
    if next_state.at < last.at:
        raise ValueError("next segment timestamp precedes terminal liquidation")
    return next_state


def _at(second: int) -> datetime:
    return datetime(2026, 1, 1, tzinfo=UTC) + timedelta(seconds=second)


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


def _first(volume: F, fee: F) -> ledger.LedgerResult:
    state = ledger.InitialState(_at(0), F(100), F(0), F(10))
    decision = ledger.Decision("buy", _at(0), _at(0), _at(0), _at(1), F(2))
    entry = ledger.MarketBucket(_at(1), F(11), F("9.5"), F(10), F(10))
    terminal = ledger.MarketBucket(_at(2), F(11), F("10.5"), F(11), volume)
    return ledger.run_ledger(
        state,
        [decision],
        [entry],
        fixed_fee=fee,
        terminal_policy="liquidate_at_declared_bid",
        terminal_bucket=terminal,
    )


def _scenario(volume: F, fee: F) -> dict[str, Any]:
    first = _first(volume, fee)
    second = None
    reason = None
    try:
        next_state = cash_handoff(first, at=_at(2), mark=F(11))
    except ValueError as exc:
        reason = str(exc)
    else:
        second = ledger.run_ledger(
            next_state, [], [ledger.MarketBucket(_at(3), F(12), F("11.5"), F(12), F(10))]
        )
    return {
        "terminal_volume": str(volume),
        "first_segment": _encode(asdict(first)),
        "handoff_accepted": second is not None,
        "handoff_rejection": reason,
        "second_segment": _encode(asdict(second)) if second is not None else None,
    }


def _controls() -> tuple[dict[str, Any], dict[str, Any]]:
    full = _first(F(2), F(1))
    partial = _first(F(1), F(1))
    next_state = cash_handoff(full, at=_at(2), mark=F(11))
    next_quote = ledger.MarketBucket(_at(3), F(12), F("11.5"), F(12), F(10))
    second = ledger.run_ledger(next_state, [], [next_quote])
    # This is a different inventory policy, intentionally not the reset strategy.
    carried = ledger.run_ledger(
        full.initial, full.decisions, [full.buckets[0], next_quote], fixed_fee=1
    )
    controls: dict[str, Any] = {
        "cash_after_paid_boundary": {"observed": str(second.terminal.cash), "expected": "99"},
        "handoff_does_not_replenish_initial_budget": {
            "observed": str(next_state.cash),
            "expected": "99",
        },
        "partial_boundary_preserves_stock": {
            "observed": str(partial.terminal.inventory),
            "expected": "1",
        },
    }
    for control in controls.values():
        control["passed"] = control["observed"] == control["expected"]
    negatives = {
        "omit_boundary_costs": {
            "wrong_final_equity": str(full.rows[0].equity),
            "correct_final_cash": str(second.terminal.cash),
            "detected": full.rows[0].equity != second.terminal.cash,
        },
        "carry_unrequested_stock": {
            "wrong_final_equity": str(carried.terminal.marked_equity),
            "correct_final_cash": str(second.terminal.cash),
            "detected": carried.terminal.marked_equity != second.terminal.cash,
        },
        "erase_boundary_stock": {
            "wrong_cash": str(partial.terminal.marked_equity),
            "correct_cash": str(partial.terminal.cash),
            "residual_inventory": str(partial.terminal.inventory),
            "detected": partial.terminal.inventory != 0,
        },
    }
    return controls, negatives


def run_lab(*, terminal_volumes: Any = (0, 1, 2, 4), fixed_fee: Any = 1) -> dict[str, Any]:
    """Retain successful and rejected transitions for a fixed synthetic decision tape."""
    start = time.perf_counter()
    if not isinstance(terminal_volumes, (list, tuple)) or not terminal_volumes:
        raise ValueError("terminal_volumes must be a nonempty ordered array")
    volumes = [ledger._number(value, "terminal_volumes") for value in terminal_volumes]
    fee = ledger._number(fixed_fee, "fixed_fee")
    scenarios = [_scenario(volume, fee) for volume in volumes]
    controls, negatives = _controls()
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
        "lab": "execution_segments",
        "schema_version": 1,
        "synthetic_only": True,
        "config": {"terminal_volumes": [str(v) for v in volumes], "fixed_fee": str(fee)},
        "numeric_encoding": "exact rational strings from decimal inputs",
        "scenarios": scenarios,
        "controls": controls,
        "negative_controls": negatives,
        "limitations": [
            "Synthetic state-transfer examples, not investment performance or calibrated liquidity.",
            "No adapter from CPCV paths; statistical validity and training causality are not certified.",
            "Only completed decisions and a fully cash-liquidated boundary can hand off.",
            "No state reset, new deposit, borrowing, inventory transfer or financing between segments.",
            "Uncompleted paths are retained and rejected, never dropped from the study.",
            "Exact decimal-input fractions are not market precision; trusted local workloads only.",
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

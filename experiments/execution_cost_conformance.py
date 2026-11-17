"""Portable exact cost fixtures with one offline OpenQuant adapter.

No external engine, strategy selection, market data, broker, or trading service.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta
from fractions import Fraction as F
from pathlib import Path
from typing import Any

from experiments.fill_ledger import Decision, InitialState, MarketBucket, run_ledger

FIXTURE = Path(__file__).with_name("fixtures") / "execution_cost_conformance_v1.json"
CONTRACT = {
    "instrument": "SYNTH",
    "currency": "USD",
    "price_unit": "USD/share",
    "fee_rate_unit": "fraction_of_absolute_executed_notional",
    "fixed_fee_unit": "USD_per_nonzero_fill",
    "timestamp_origin": "2026-01-01T00:00:00+00:00",
    "timestamp_unit": "integer_seconds",
    "ordering": "decision_strictly_before_fill",
    "account": "long_only_no_borrowing",
    "numeric_encoding": "exact_rational_strings",
}


def compare(expected: dict[str, Any], observed: dict[str, Any]) -> list[str]:
    """Compare complete paths under the fixture's rational-string/bool contract."""
    if expected.keys() != observed.keys():
        raise ValueError("comparison fields must match exactly")
    differences = []
    for name, wanted in expected.items():
        actual = observed[name]
        if type(wanted) is bool:
            if type(actual) is not bool or actual != wanted:
                differences.append(name)
            continue
        if not isinstance(wanted, list) or not isinstance(actual, list):
            raise ValueError(f"{name} must be a list of rational strings")
        if any(not isinstance(value, str) for value in wanted + actual):
            raise ValueError(f"{name} must contain rational strings")
        if [F(value) for value in wanted] != [F(value) for value in actual]:
            differences.append(name)
    return differences


def _observe(document: dict[str, Any], case: dict[str, Any], *, rate_divisor: int = 1):
    scenario = document["scenario"]
    origin = datetime.fromisoformat(document["contract"]["timestamp_origin"])

    def timed(record: dict[str, Any], fields: tuple[str, ...]) -> dict[str, Any]:
        return {k: origin + timedelta(seconds=v) if k in fields else v for k, v in record.items()}

    initial = InitialState(**timed(scenario["initial"], ("at",)))
    decision = Decision(
        **timed(
            scenario["decision"], ("information_at", "decision_at", "earliest_fill_at", "deadline")
        )
    )
    buckets = [
        MarketBucket(**timed(record, ("at",)), volume=volume)
        for record, volume in zip(scenario["buckets"], case["volumes"], strict=True)
    ]
    terminal = None
    if case["terminal_volume"] is not None:
        terminal = MarketBucket(
            **timed(scenario["terminal_bucket"], ("at",)), volume=case["terminal_volume"]
        )
    result = run_ledger(
        initial,
        [decision],
        buckets,
        participation=scenario["participation"],
        fixed_fee=case["fixed_fee"],
        fee_rate=F(case["fee_rate"]) / rate_divisor,
        terminal_policy="mark_only" if terminal is None else "liquidate_at_declared_bid",
        terminal_bucket=terminal,
    )
    observed: dict[str, Any] = {
        name: [str(getattr(row, name)) for row in result.rows]
        for name in ("cash", "inventory", "equity", "commission")
    }
    observed.update(
        fill_shares=[str(fill.shares) for fill in result.fills],
        fill_prices=[str(fill.price) for fill in result.fills],
        fill_commissions=[str(fill.commission) for fill in result.fills],
        terminal_completed=result.terminal.completed,
        decisions_completed=result.decisions_completed,
    )
    return observed


def run() -> dict[str, Any]:
    """Check bundled original fixtures; report deliberate wrong-model controls."""
    raw = FIXTURE.read_bytes()
    document = json.loads(raw)
    if document["schema_version"] != 1:
        raise ValueError("unsupported fixture schema_version")
    if document["contract"] != CONTRACT:
        raise ValueError("unsupported fixture contract")
    identities = [case["id"] for case in document["cases"]]
    if len(identities) != len(set(identities)):
        raise ValueError("duplicate fixture case IDs")
    cases = []
    for case in document["cases"]:
        observed = _observe(document, case)
        differences = compare(case["expected"], observed)
        cases.append(
            dict(
                id=case["id"],
                passed=not differences,
                differing_fields=differences,
                observed=observed,
            )
        )

    by_id = {case["id"]: case for case in document["cases"]}
    paid = by_id["ten_basis_points_plus_per_fill_fee"]
    full_exit = by_id["paid_terminal_liquidation"]
    partial_exit = by_id["paid_incomplete_terminal_liquidation"]
    wrong_rate = _observe(document, paid, rate_divisor=100)
    # Hand-constructed wrong conventions, NOT claimed defects in any engine.
    wrong_once = F(100) - 2 * 10 - 3 * 8 - 1 - F(44, 1000)
    wrong_exit = F(paid["expected"]["cash"][-1]) + 5 * 9
    negatives = {
        "rate_divided_by_100": {
            "wrong_cash": wrong_rate["cash"][-1],
            "correct_cash": paid["expected"]["cash"][-1],
            "detected": bool(compare(paid["expected"], wrong_rate)),
        },
        "one_fee_per_order": {
            "wrong_cash": str(wrong_once),
            "correct_cash": paid["expected"]["cash"][-1],
            "detected": wrong_once != F(paid["expected"]["cash"][-1]),
        },
        "omit_terminal_charge": {
            "wrong_cash": str(wrong_exit),
            "correct_cash": full_exit["expected"]["cash"][-1],
            "detected": wrong_exit != F(full_exit["expected"]["cash"][-1]),
        },
        "erase_terminal_inventory": {
            "wrong_inventory": "0",
            "correct_inventory": partial_exit["expected"]["inventory"][-1],
            "detected": F(partial_exit["expected"]["inventory"][-1]) != 0,
        },
    }
    return dict(
        schema_version=1,
        synthetic_only=True,
        adapters_tested=["experimental_openquant_fill_ledger"],
        fixture_sha256=hashlib.sha256(raw).hexdigest(),
        cases=cases,
        negative_controls=negatives,
        limitation="One adapter only. Agreement is not external validation or investment evidence.",
    )


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))

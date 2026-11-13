"""Reproducible synthetic shared-cash accounting, not investment research results.

Run: python -m experiments.cash_reservation_study > /tmp/reservations.json
Replay: python -m experiments.cash_reservation_study /tmp/reservations.json
"""

from __future__ import annotations

import json
import random
import sys
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from fractions import Fraction as F
from pathlib import Path

from experiments.cash_budget_lab import size_buy
from experiments.cash_reservations import CashReservations
from experiments.fill_ledger import Fill

T = datetime(2026, 1, 1, tzinfo=UTC)


def run_study(seed: int = 4815, scenarios: int = 100) -> dict:
    if type(seed) is not int or type(scenarios) is not int or not 1 <= scenarios <= 1000:
        raise ValueError("seed must be an integer and scenarios in [1, 1000]")
    rng = random.Random(seed)
    runs = []
    for scenario in range(scenarios):
        cash = F(rng.randint(30, 300))
        floor = cash / 10
        book = CashReservations(cash, cash_floor=floor)
        oracle_cash = cash
        oracle_orders: dict[str, list[F]] = {}
        receipts = []
        count = 0

        def check(kind, detail):
            # Separate reference: enumerate future unit quantities and unused
            # fees from recorded requests/fills, never use _Order.remaining.
            reserved = sum(
                (
                    max(F(0), q - executed) * cap * (1 + rate)
                    + ((n - used) * fixed if q > executed else 0)
                    for q, executed, cap, rate, n, used, fixed in oracle_orders.values()
                ),
                F(0),
            )
            s = book.snapshot()
            if (
                s.cash != oracle_cash
                or s.reserved != reserved
                or s.available != oracle_cash - floor - reserved
                or s.available < 0
            ):
                raise ValueError("cash/reservation reference reconciliation failed")
            receipts.append(
                {
                    "kind": kind,
                    "detail": detail,
                    "snapshot": {k: str(v) for k, v in asdict(s).items()},
                }
            )

        # Three waves exercise reservation alongside partial fills/finalization.
        for wave in range(3):
            budgets = []
            for i in range(4):
                oid = f"{wave}-{i}"
                cap = F(rng.randint(1, 15))
                fixed = F(rng.randint(0, 3), 10)
                rate = F(rng.randint(0, 2), 100)
                n = rng.randint(1, 4)
                desired = F(rng.randint(0, 30))
                b = book.reserve(
                    oid,
                    desired_shares=desired,
                    price_cap=cap,
                    lot_size=F(1, 2),
                    fixed_fee=fixed,
                    fee_rate=rate,
                    max_fills=n,
                    information_at=T,
                    decision_at=T + timedelta(seconds=count),
                )
                oracle_orders[oid] = [b.shares, F(0), cap, rate, F(n), F(0), fixed]
                budgets.append((oid, b))
                check("reserve", {"order": oid, "shares": str(b.shares)})
            for oid, b in budgets:
                q = b.shares
                # No fills, partial fills, completion, and repeated fixed fees.
                pieces = rng.randint(0, b.max_fills)
                for j in range(pieces):
                    if not q:
                        break
                    quantity = q if j == pieces - 1 and rng.randrange(2) else q / 2
                    q -= quantity
                    price = b.price_cap * F(rng.randint(1, 10), 10)
                    fee = b.fixed_fee + quantity * price * b.fee_rate
                    count += 1
                    f = Fill(oid, T + timedelta(seconds=count), quantity, price, fee, False)
                    event = f"{scenario}-{count}"
                    book.fill(event, f)
                    oracle_cash -= quantity * price + fee
                    oracle_orders[oid][1] += quantity
                    oracle_orders[oid][5] += 1
                    check(
                        "fill",
                        {
                            "event": event,
                            "order": oid,
                            "quantity": str(quantity),
                            "price": str(price),
                            "fee": str(fee),
                        },
                    )
                    before = book.snapshot()
                    book.fill(event, f)
                    if book.snapshot() != before:
                        raise ValueError("duplicate execution mutated account")
                book.finalize(oid)
                del oracle_orders[oid]
                check("finalize_complete_history", {"order": oid})
        runs.append(
            {
                "scenario": scenario,
                "initial_cash": str(cash),
                "cash_floor": str(floor),
                "receipts": receipts,
            }
        )
    independent = [
        size_buy(
            cash=100,
            desired_shares=10,
            price_cap=10,
            lot_size=1,
            fixed_fee=1,
            fee_rate=0,
            max_fills=2,
            information_at=T,
            decision_at=T,
        )
        for _ in range(2)
    ]
    oversubscribed = sum(b.worst_case_spend for b in independent)
    if not oversubscribed == 184 > 100:
        raise ValueError("negative control no longer demonstrates oversubscription")
    return {
        "schema": "cash-reservation-study-v1",
        "seed": seed,
        "scenarios": scenarios,
        "scope": "synthetic exact accounting; no execution or return forecast",
        "negative_control": {
            "independent_budget_total": str(oversubscribed),
            "actual_shared_cash": "100",
        },
        "runs": runs,
    }


def main():
    if len(sys.argv) == 1:
        print(json.dumps(run_study(), indent=2))
    elif len(sys.argv) == 2:
        report = json.loads(Path(sys.argv[1]).read_text())
        actual = run_study(report["seed"], report["scenarios"])
        if report != actual:
            raise ValueError("report differs from exact deterministic replay")
        print("Exact deterministic replay matched")
    else:
        raise SystemExit("usage: python -m experiments.cash_reservation_study [report.json]")


if __name__ == "__main__":
    main()

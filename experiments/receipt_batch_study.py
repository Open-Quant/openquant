"""Paired synthetic receipt-batching study, not latency or investment evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import sqlite3
import tempfile
import time
from datetime import UTC, datetime, timedelta
from fractions import Fraction as F
from pathlib import Path
from typing import Any

from experiments.fill_ledger import Fill
from experiments.receipt_journal import ReceiptJournal


def _workload(seed: int, orders: int) -> tuple[int, int, list[dict[str, Any]]]:
    """Construct receipts and independent integer-cent cash reference."""
    rng = random.Random(seed)
    initial = orders * 50 + 100
    expected_cents = initial * 100
    t = datetime(2026, 1, 1, tzinfo=UTC)
    unique: list[dict[str, Any]] = []
    for i in range(orders):
        order = str(i)
        for quantity in (1, 2):
            cents = rng.randrange(800, 1201)
            sequence = len(unique) + 1
            expected_cents -= quantity * cents + 100
            unique.append(
                dict(
                    event_id=str(sequence),
                    sequence=sequence,
                    kind="fill",
                    order_id=order,
                    fill=Fill(
                        order,
                        t + timedelta(seconds=sequence),
                        F(quantity),
                        F(cents, 100),
                        F(1),
                        False,
                    ),
                )
            )
        sequence = len(unique) + 1
        unique.append(
            dict(event_id=str(sequence), sequence=sequence, kind="terminal", order_id=order)
        )
    delivered = unique + unique
    rng.shuffle(delivered)
    return initial, expected_cents, delivered


def _reserve(book: ReceiptJournal, orders: int) -> None:
    t = datetime(2026, 1, 1, tzinfo=UTC)
    for i in range(orders):
        book.reserve(
            "reserve" + str(i),
            str(i),
            desired_shares=3,
            price_cap=12,
            lot_size=1,
            fixed_fee=1,
            fee_rate=0,
            max_fills=2,
            information_at=t,
            decision_at=t,
        )


def _journal_digest(path: Path) -> str:
    with sqlite3.connect(path) as db:
        rows = db.execute("SELECT seq,payload FROM commands ORDER BY seq").fetchall()
    return hashlib.sha256(json.dumps(rows, separators=(",", ":")).encode()).hexdigest()


def run(
    seed: int = 20261114,
    orders: int = 30,
    batch_size: int = 16,
    restart_every: int = 0,
) -> dict[str, Any]:
    """Compare final and boundary states against one-at-a-time replay.

    Oracle runtime is excluded from candidate elapsed time. Ingestion includes
    requested close/reopen cycles at fixed receipt counts; total also includes creation, reservations,
    final snapshot and close. Timing is descriptive for this local workload.
    """
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    if type(orders) is not int or not 1 <= orders <= 10_000:
        raise ValueError("orders must be an integer from 1 to 10000")
    if type(batch_size) is not int or not 1 <= batch_size <= 256:
        raise ValueError("batch_size must be an integer from 1 to 256")
    if type(restart_every) is not int or restart_every < 0:
        raise ValueError("restart_every must be a nonnegative integer")
    initial, expected_cents, delivered = _workload(seed, orders)
    # Split at identical receipt-count restart boundaries for every batch size.
    # Otherwise a larger batch would quietly receive fewer restart penalties.
    chunks = []
    offset = 0
    while offset < len(delivered):
        size = min(batch_size, len(delivered) - offset)
        if restart_every:
            size = min(size, restart_every - offset % restart_every)
        chunks.append(delivered[offset : offset + size])
        offset += size
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "candidate.sqlite"
        oracle_path = Path(directory) / "oracle.sqlite"
        validation_start = time.perf_counter()
        oracle = ReceiptJournal(oracle_path, cash=initial, cash_floor=100)
        try:
            _reserve(oracle, orders)
            boundaries = []
            for chunk in chunks:
                for receipt in chunk:
                    state = oracle.receive(**receipt)
                boundaries.append(state)
            oracle_final = oracle.snapshot()
        finally:
            oracle.close()
        oracle_seconds = time.perf_counter() - validation_start
        start = time.perf_counter()
        candidate = ReceiptJournal(path, cash=initial, cash_floor=100)
        restarts = 0
        try:
            _reserve(candidate, orders)
            ingest_start = time.perf_counter()
            candidate_states = []
            processed = 0
            for chunk in chunks:
                if batch_size == 1:
                    state = candidate.receive(**chunk[0])
                else:
                    state = candidate.receive_batch(chunk)
                candidate_states.append(state)
                processed += len(chunk)
                if restart_every and processed % restart_every == 0:
                    candidate.close()
                    candidate = ReceiptJournal(path, cash=initial, cash_floor=100)
                    restarts += 1
            ingest_seconds = time.perf_counter() - ingest_start
            final = candidate.snapshot()
            count = candidate.command_count()
        finally:
            candidate.close()
        total_seconds = time.perf_counter() - start
        expected = F(expected_cents, 100)
        if final.cash != expected or final.available != expected - 100 or final.reserved != 0:
            raise AssertionError("independent integer-cent cash reference mismatch")
        if final != oracle_final or candidate_states != boundaries:
            raise AssertionError("batch boundaries differ from single-receipt replay")
        if count != orders * 4 or final.watermark != orders * 3 or final.quarantined:
            raise AssertionError("unexpected durable command count or source state")
        digest = _journal_digest(path)
        if digest != _journal_digest(oracle_path):
            raise AssertionError("journal bytes differ from single-receipt replay")
    return dict(
        seed=seed,
        orders=orders,
        batch_size=batch_size,
        delivered_receipts=len(delivered),
        batch_count=len(chunks),
        restart_every=restart_every,
        restarts=restarts,
        durable_commands=count,
        source_watermark=final.watermark,
        final_cash=str(final.cash),
        independent_cash_match=True,
        identical_journal_to_sequential=True,
        journal_sha256=digest,
        ingest_seconds=round(ingest_seconds, 6),
        total_seconds=round(total_seconds, 6),
        oracle_validation_seconds=round(oracle_seconds, 6),
        evidence="Synthetic batch/accounting equivalence only; no market or broker validation",
    )


def _source_identity() -> dict[str, str]:
    return {
        name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
        for name in (
            "receipt_batch_study.py",
            "receipt_journal.py",
            "cash_reservations.py",
            "cash_budget_lab.py",
            "fill_ledger.py",
        )
    }


def campaign(
    *,
    order_counts: list[int],
    batch_sizes: list[int],
    repeats: int = 1,
    seed: int = 20261114,
    restart_every: int = 0,
):
    """Stream completed cases; only a final record certifies campaign completion."""
    if not 1 <= repeats <= 10:
        raise ValueError("repeats must be from 1 to 10")
    identity = _source_identity()
    yield dict(
        record="start",
        complete=False,
        provenance=dict(
            source_sha256=identity,
            python=platform.python_version(),
            sqlite=sqlite3.sqlite_version,
            platform=platform.platform(),
        ),
        workload=dict(
            order_counts=order_counts,
            batch_sizes=batch_sizes,
            repeats=repeats,
            seed=seed,
            restart_every=restart_every,
        ),
    )
    count = 0
    for repeat in range(repeats):
        for orders in order_counts:
            for batch in batch_sizes:
                if _source_identity() != identity:
                    raise RuntimeError("Source changed during campaign")
                result = run(
                    seed=seed + repeat, orders=orders, batch_size=batch, restart_every=restart_every
                )
                if _source_identity() != identity:
                    raise RuntimeError("Source changed during campaign")
                count += 1
                yield dict(record="result", result=result)
    yield dict(record="complete", complete=True, result_count=count)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--orders", type=int, nargs="+", default=[10, 30, 60])
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 16, 64])
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20261114)
    parser.add_argument(
        "--restart-every",
        type=int,
        default=0,
        help="Restart after this many delivered receipts (0 disables)",
    )
    parser.add_argument("--jsonl", action="store_true", help="Flush each completed case")
    args = parser.parse_args()
    records = campaign(
        order_counts=args.orders,
        batch_sizes=args.batch_sizes,
        repeats=args.repeats,
        seed=args.seed,
        restart_every=args.restart_every,
    )
    if args.jsonl:
        for record in records:
            print(json.dumps(record), flush=True)
    else:
        print(json.dumps(list(records), indent=2))


if __name__ == "__main__":
    main()

"""Deterministic synthetic receipt replay, not market or broker validation."""

import json
import random
import tempfile
import time
from datetime import UTC, datetime, timedelta
from fractions import Fraction as F
from pathlib import Path

from experiments.fill_ledger import Fill
from experiments.receipt_journal import ReceiptJournal


def run(seed=20261113, orders=30):
    rng = random.Random(seed)
    start = time.perf_counter()
    initial = 10000
    expected = F(initial)
    t = datetime(2026, 1, 1, tzinfo=UTC)
    receipts = []
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "journal.sqlite"
        book = ReceiptJournal(path, cash=initial, cash_floor=100)
        for i in range(orders):
            order = str(i)
            book.reserve(
                "reserve" + order,
                order,
                desired_shares=3,
                price_cap=12,
                lot_size=1,
                fixed_fee=1,
                fee_rate=0,
                max_fills=2,
                information_at=t,
                decision_at=t,
            )
            for quantity in (1, 2):
                price = F(rng.randrange(800, 1201), 100)
                seq = len(receipts) + 1
                fill = Fill(order, t + timedelta(seconds=seq), F(quantity), price, F(1), False)
                expected -= quantity * price + 1
                receipts.append(
                    dict(event_id=str(seq), sequence=seq, kind="fill", order_id=order, fill=fill)
                )
            seq = len(receipts) + 1
            receipts.append(dict(event_id=str(seq), sequence=seq, kind="terminal", order_id=order))
        rng.shuffle(receipts)
        for i, receipt in enumerate(receipts):
            book.receive(**receipt)
            book.receive(**receipt)  # at-least-once delivery
            if i % 11 == 0:
                book.close()
                book = ReceiptJournal(path, cash=initial, cash_floor=100)
        final = book.snapshot()
        assert final.cash == expected
        assert final.reserved == 0 and final.available == expected - 100
        assert final.watermark == orders * 3 and not final.quarantined
        assert book.command_count() == orders * 4
        count = book.command_count()
        book.close()
    return dict(
        seed=seed,
        orders=orders,
        source_receipts=orders * 3,
        durable_commands=count,
        cash=str(expected),
        elapsed_seconds=round(time.perf_counter() - start, 4),
        evidence="synthetic arithmetic and replay only",
    )


if __name__ == "__main__":
    print(json.dumps([run(orders=n) for n in (10, 30, 60)], indent=2))

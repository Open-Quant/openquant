"""Atomic offline receipt batches; synthetic accounting, not broker qualification."""

import itertools
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from fractions import Fraction as F
from pathlib import Path

from experiments.fill_ledger import Fill
from experiments.receipt_journal import ReceiptJournal

T = datetime(2026, 1, 1, tzinfo=UTC)


def bounds():
    return dict(
        desired_shares=10,
        price_cap=10,
        lot_size=1,
        fixed_fee=1,
        fee_rate=0,
        max_fills=2,
        information_at=T,
        decision_at=T,
    )


def receipts():
    return [
        dict(
            event_id="f1",
            sequence=1,
            kind="fill",
            order_id="a",
            fill=Fill("a", T + timedelta(seconds=1), F(2), F(9), F(1), False),
        ),
        dict(
            event_id="f2",
            sequence=2,
            kind="fill",
            order_id="a",
            fill=Fill("a", T + timedelta(seconds=2), F(3), F(8), F(1), False),
        ),
        dict(event_id="done", sequence=3, kind="terminal", order_id="a"),
    ]


class ReceiptBatchTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "batch.sqlite"
        self.book = ReceiptJournal(self.path, cash=100, cash_floor=5)
        self.book.reserve("r1", "a", **bounds())

    def tearDown(self):
        self.book.close()
        self.tmp.cleanup()

    def test_batch_matches_exact_independent_arithmetic(self):
        self.assertTrue(
            callable(getattr(self.book, "receive_batch", None)),
            "bounded atomic receive_batch is missing",
        )
        result = self.book.receive_batch(receipts())
        self.assertEqual((result.cash, result.reserved, result.available), (56, 0, 51))
        self.assertEqual((result.watermark, result.pending, result.quarantined), (3, 0, False))
        self.assertEqual(self.book.command_count(), 4)

    def test_every_permutation_and_batch_boundary_matches_single_receipt_oracle(self):
        self.assertTrue(hasattr(self.book, "receive_batch"))
        for permutation in itertools.permutations(receipts()):
            for partition in ((3,), (1, 2), (2, 1), (1, 1, 1)):
                with self.subTest(order=[r["event_id"] for r in permutation], partition=partition):
                    path = Path(self.tmp.name) / str(len(list(Path(self.tmp.name).iterdir())))
                    batch = ReceiptJournal(path, cash=100, cash_floor=5)
                    oracle = ReceiptJournal(str(path) + "-oracle", cash=100, cash_floor=5)
                    try:
                        for book in (batch, oracle):
                            book.reserve("r1", "a", **bounds())
                        offset = 0
                        for size in partition:
                            chunk = permutation[offset : offset + size]
                            offset += size
                            for r in chunk:
                                oracle.receive(**r)
                            self.assertEqual(batch.receive_batch(chunk), oracle.snapshot())
                        self.assertEqual(batch.snapshot().cash, 56)
                    finally:
                        batch.close()
                        oracle.close()

    def test_syntax_error_anywhere_rejects_whole_batch(self):
        self.assertTrue(hasattr(self.book, "receive_batch"))
        original = self.book.snapshot()
        bad = dict(receipts()[1], sequence=0)
        with self.assertRaisesRegex(ValueError, "sequence"):
            self.book.receive_batch([receipts()[0], bad])
        self.assertEqual(self.book.snapshot(), original)
        self.assertEqual(self.book.command_count(), 1)

    def test_semantic_breach_retains_entire_batch_and_quarantines(self):
        self.assertTrue(hasattr(self.book, "receive_batch"))
        bad = dict(receipts()[0], fill=Fill("a", T, F(99), F(9), F(1), False))
        state = self.book.receive_batch([bad, receipts()[2]])
        self.assertTrue(state.quarantined)
        self.assertEqual((state.cash, state.available, state.watermark), (100, 0, 0))
        self.assertEqual(self.book.command_count(), 3)

    def test_exact_retries_inside_and_across_batches_do_not_grow_history(self):
        self.assertTrue(hasattr(self.book, "receive_batch"))
        payload = receipts()
        result = self.book.receive_batch(payload + payload)
        self.assertEqual(self.book.receive_batch(payload), result)
        self.assertEqual(self.book.command_count(), 4)
        self.book.close()
        self.book = ReceiptJournal(self.path, cash=100, cash_floor=5)
        self.assertEqual(self.book.receive_batch(payload), result)
        self.assertEqual(self.book.command_count(), 4)

    def test_bounded_iterable_rejected_without_writes(self):
        self.assertTrue(hasattr(self.book, "receive_batch"))
        consumed = []

        def stream():
            for i in range(1000):
                consumed.append(i)
                yield receipts()[0]

        with self.assertRaisesRegex(ValueError, "256"):
            self.book.receive_batch(stream())
        self.assertEqual(len(consumed), 257)
        self.assertEqual(self.book.command_count(), 1)
        with self.assertRaisesRegex(ValueError, "empty"):
            self.book.receive_batch([])

    def test_iterator_failure_preserves_original_database(self):
        self.assertTrue(hasattr(self.book, "receive_batch"))

        def broken():
            yield receipts()[0]
            raise RuntimeError("source disconnected")

        with self.assertRaisesRegex(RuntimeError, "disconnected"):
            self.book.receive_batch(broken())
        self.assertEqual(self.book.command_count(), 1)

    def test_sql_failure_mid_batch_rolls_back_all_rows(self):
        self.assertTrue(hasattr(self.book, "receive_batch"))
        with sqlite3.connect(self.path) as db:
            db.execute("""CREATE TRIGGER reject_second BEFORE INSERT ON commands
                          WHEN NEW.payload LIKE '%"id":"f2"%'
                          BEGIN SELECT RAISE(ABORT, 'disk fault simulation'); END""")
        with self.assertRaisesRegex(sqlite3.IntegrityError, "disk fault"):
            self.book.receive_batch(receipts())
        self.assertEqual(self.book.command_count(), 1)
        self.assertEqual(self.book.snapshot().cash, 100)
        with sqlite3.connect(self.path) as db:
            db.execute("DROP TRIGGER reject_second")
        self.assertEqual(self.book.receive_batch(receipts()).cash, 56)

    def test_each_item_is_frozen_before_iterator_resumes(self):
        item = receipts()[0]

        def reused_mapping():
            yield item
            item.clear()
            item.update(receipts()[1])
            yield item
            item.clear()
            item.update(receipts()[2])
            yield item

        self.assertEqual(self.book.receive_batch(reused_mapping()).cash, 56)
        self.assertEqual(self.book.command_count(), 4)

    def test_byte_limit_is_encoded_bytes_not_character_count(self):
        original = self.book.snapshot()
        oversized = dict(receipts()[0], event_id="😀" * 300_000)
        with self.assertRaisesRegex(ValueError, "1 MiB"):
            self.book.receive_batch([oversized])
        self.assertEqual(self.book.snapshot(), original)
        self.assertEqual(self.book.command_count(), 1)

    def test_conflict_retained_after_valid_prefix_with_later_evidence(self):
        conflict = dict(receipts()[0], fill=Fill("a", T, F(3), F(9), F(1), False))
        result = self.book.receive_batch([receipts()[0], conflict, receipts()[1], receipts()[2]])
        self.assertEqual((result.cash, result.available, result.watermark), (81, 0, 1))
        self.assertTrue(result.quarantined)
        self.assertEqual(self.book.command_count(), 5)
        with sqlite3.connect(self.path) as db:
            rows = db.execute("SELECT payload FROM commands ORDER BY seq").fetchall()
        self.assertIn('"shares":"2"', rows[1][0])
        self.assertIn('"shares":"3"', rows[2][0])

    def test_gap_withholds_cash_until_later_batch_completes_prefix(self):
        first = self.book.receive_batch([receipts()[2]])
        self.assertEqual((first.available, first.pending, first.watermark), (0, 1, 0))
        second = self.book.receive_batch(receipts()[:2])
        self.assertEqual((second.available, second.pending, second.watermark), (51, 0, 3))

    def test_independent_writers_retry_same_batch_once(self):
        def send(_):
            other = ReceiptJournal(self.path, cash=100, cash_floor=5)
            try:
                return other.receive_batch(receipts())
            finally:
                other.close()

        with ThreadPoolExecutor(max_workers=4) as pool:
            snapshots = list(pool.map(send, range(12)))
        self.assertEqual({s.cash for s in snapshots}, {F(56)})
        self.assertEqual(self.book.command_count(), 4)

    def test_process_exit_before_commit_keeps_none_after_commit_keeps_all(self):
        code = """
import os, sqlite3, sys
from datetime import UTC, datetime, timedelta
from fractions import Fraction as F
from unittest.mock import patch
from experiments.receipt_journal import ReceiptJournal
from experiments.fill_ledger import Fill
class CrashConnection(sqlite3.Connection):
    count = 0
    inserts = 0
    def execute(self, sql, *args, **kwargs):
        result = super().execute(sql, *args, **kwargs)
        if sql.startswith("INSERT INTO commands"):
            self.inserts += 1
            if self.inserts == 2 and sys.argv[2] == "middle": os._exit(75)
        return result
    def commit(self):
        self.count += 1
        if self.count == 2 and sys.argv[2] == "before": os._exit(73)
        super().commit()
        if self.count == 2: os._exit(74)
original = sqlite3.connect
with patch("sqlite3.connect", lambda *a, **kw: original(*a, factory=CrashConnection, **kw)):
    book = ReceiptJournal(sys.argv[1], cash=100, cash_floor=5)
    t = datetime(2026, 1, 1, tzinfo=UTC)
    book.receive_batch([
        dict(event_id="f1", sequence=1, kind="fill", order_id="a",
             fill=Fill("a", t + timedelta(seconds=1), F(2), F(9), F(1), False)),
        dict(event_id="f2", sequence=2, kind="fill", order_id="a",
             fill=Fill("a", t + timedelta(seconds=2), F(3), F(8), F(1), False)),
        dict(event_id="done", sequence=3, kind="terminal", order_id="a"),
    ])
"""
        for phase, expected_count, expected_cash in (
            ("before", 1, 100),
            ("middle", 1, 100),
            ("after", 4, 56),
        ):
            with self.subTest(phase=phase):
                path = Path(self.tmp.name) / phase
                book = ReceiptJournal(path, cash=100, cash_floor=5)
                book.reserve("r1", "a", **bounds())
                book.close()
                result = subprocess.run(
                    [sys.executable, "-c", code, str(path), phase],
                    capture_output=True,
                    text=True,
                    timeout=20,
                )
                self.assertEqual(
                    result.returncode,
                    {"before": 73, "middle": 75, "after": 74}[phase],
                    result.stderr,
                )
                book = ReceiptJournal(path, cash=100, cash_floor=5)
                try:
                    self.assertEqual(book.command_count(), expected_count)
                    self.assertEqual(book.snapshot().cash, expected_cash)
                    self.assertEqual(book.receive_batch(receipts()).cash, 56)
                    self.assertEqual(book.command_count(), 4)
                finally:
                    book.close()


if __name__ == "__main__":
    unittest.main()

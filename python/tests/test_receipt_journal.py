"""Offline durable receipt experiments; no broker or investment-performance claim."""

import importlib.util
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

T = datetime(2026, 1, 1, tzinfo=UTC)


class AvailabilityTest(unittest.TestCase):
    def test_durable_receipt_journal_exists(self):
        self.assertIsNotNone(importlib.util.find_spec("experiments.receipt_journal"))


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


class JournalTest(unittest.TestCase):
    def setUp(self):
        from experiments.receipt_journal import ReceiptJournal

        self.cls = ReceiptJournal
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "journal.sqlite"
        self.book = self.cls(self.path, cash=100, cash_floor=5)

    def tearDown(self):
        self.book.close()
        self.tmp.cleanup()

    def fill(self, seq=1, event_id="e1", quantity=2):
        from experiments.fill_ledger import Fill

        return self.book.receive(
            event_id,
            sequence=seq,
            kind="fill",
            order_id="a",
            fill=Fill("a", T + timedelta(seconds=seq), F(quantity), F(9), F(1), False),
        )

    def test_restart_preserves_reservation_and_fill_idempotency(self):
        self.book.reserve("r1", "a", **bounds())
        self.fill()
        before = self.book.snapshot()
        self.assertEqual((before.cash, before.reserved, before.available), (81, 71, 5))
        self.book.close()
        self.book = self.cls(self.path, cash=100, cash_floor=5)
        self.assertEqual(self.book.snapshot(), before)
        self.fill()
        self.assertEqual(self.book.snapshot(), before)

    def test_terminal_cannot_cross_missing_receipt(self):
        self.book.reserve("r1", "a", **bounds())
        self.book.receive("terminal", sequence=2, kind="terminal", order_id="a")
        s = self.book.snapshot()
        self.assertEqual((s.reserved, s.available, s.pending), (92, 0, 1))
        with self.assertRaisesRegex(ValueError, "pending"):
            self.book.reserve("r2", "b", **bounds())
        self.fill()
        s = self.book.snapshot()
        self.assertEqual((s.cash, s.reserved, s.available, s.watermark), (81, 0, 76, 2))

    def test_conflicting_receipt_is_durable_quarantine(self):
        self.book.reserve("r1", "a", **bounds())
        self.fill()
        self.fill(quantity=3)
        self.assertTrue(self.book.snapshot().quarantined)
        self.assertEqual(self.book.snapshot().available, 0)
        self.book.close()
        self.book = self.cls(self.path, cash=100, cash_floor=5)
        with self.assertRaisesRegex(ValueError, "quarantined"):
            self.book.reserve("r2", "b", **bounds())

    def test_late_new_fill_is_retained_and_quarantined_not_forgotten(self):
        self.book.reserve("r1", "a", **bounds())
        self.book.receive("terminal", sequence=1, kind="terminal", order_id="a")
        self.fill(seq=2, event_id="late")
        self.assertTrue(self.book.snapshot().quarantined)
        self.assertEqual(self.book.snapshot().available, 0)
        self.assertEqual(self.book.command_count(), 3)

    def test_initial_balance_mismatch_is_rejected_on_reopen(self):
        with self.assertRaisesRegex(ValueError, "initial"):
            self.cls(self.path, cash=101, cash_floor=5)

    def test_reservation_retry_and_conflict(self):
        self.book.reserve("r1", "a", **bounds())
        before = self.book.snapshot()
        self.book.reserve("r1", "a", **bounds())
        self.assertEqual(self.book.snapshot(), before)
        with self.assertRaisesRegex(ValueError, "conflicting"):
            self.book.reserve("r1", "b", **bounds())
        self.assertEqual(self.book.snapshot(), before)

    def test_duplicate_delivery_does_not_grow_durable_history(self):
        self.book.reserve("r1", "a", **bounds())
        self.fill()
        count = self.book.command_count()
        for _ in range(5):
            self.fill()
        self.assertEqual(self.book.command_count(), count)

    def test_partial_schema_or_missing_metadata_is_rejected(self):
        for damage in ("DELETE FROM metadata", "DROP TABLE metadata", "DROP TABLE commands"):
            with self.subTest(damage=damage):
                path = Path(self.tmp.name) / damage.replace(" ", "_")
                book = self.cls(path, cash=100, cash_floor=5)
                book.reserve("r1", "a", **bounds())
                book.close()
                with sqlite3.connect(path) as db:
                    db.execute(damage)
                with self.assertRaisesRegex(ValueError, "incomplete"):
                    self.cls(path, cash=1000, cash_floor=5)

    def test_source_sequence_collision_freezes_account(self):
        self.book.reserve("r1", "a", **bounds())
        self.fill()
        self.fill(event_id="new-id")
        self.assertTrue(self.book.snapshot().quarantined)

    def test_all_delivery_orders_reconcile_with_independent_cash(self):
        from experiments.fill_ledger import Fill

        for permutation in itertools.permutations((1, 2, 3)):
            with self.subTest(permutation=permutation):
                path = Path(self.tmp.name) / ("perm" + "".join(map(str, permutation)))
                book = self.cls(path, cash=100, cash_floor=5)
                book.reserve("reserve", "a", **bounds())
                for seq in permutation:
                    if seq < 3:
                        payload = Fill("a", T + timedelta(seconds=seq), F(seq), F(9), F(1), False)
                        book.receive(
                            f"fill{seq}", sequence=seq, kind="fill", order_id="a", fill=payload
                        )
                    else:
                        book.receive("done", sequence=3, kind="terminal", order_id="a")
                    # Restart at every delivery, including before gaps are filled.
                    book.close()
                    book = self.cls(path, cash=100, cash_floor=5)
                s = book.snapshot()
                self.assertEqual((s.cash, s.reserved, s.available), (100 - 9 - 1 - 18 - 1, 0, 66))
                self.assertEqual(s.watermark, 3)
                self.assertFalse(s.quarantined)
                book.close()

    def test_separate_connections_cannot_spend_same_cash(self):
        def reserve_one(i):
            book = self.cls(self.path, cash=100, cash_floor=5)
            try:
                return book.reserve(f"r{i}", f"order{i}", **bounds())
            finally:
                book.close()

        with ThreadPoolExecutor(max_workers=8) as executor:
            list(executor.map(reserve_one, range(8)))
        s = self.book.snapshot()
        self.assertEqual((s.cash, s.reserved, s.available), (100, 92, 3))

    def test_breach_is_evidence_not_discarded(self):
        self.book.reserve("r1", "a", **bounds())
        self.fill(quantity=99)
        self.assertTrue(self.book.snapshot().quarantined)
        self.assertEqual(self.book.command_count(), 2)
        # Last valid prefix is NOT reported as an actual reconciled account balance.
        self.assertEqual(self.book.snapshot().cash, 100)
        self.assertEqual(self.book.snapshot().available, 0)

    def test_cancel_is_not_supported_terminal_shortcut(self):
        with self.assertRaisesRegex(ValueError, "kind"):
            self.book.receive("cancel", sequence=1, kind="cancel_ack", order_id="a")
        self.assertEqual(self.book.command_count(), 0)

    def test_invalid_sequences_rejected_without_mutation(self):
        for seq in (True, 0, -1, 1.0, "1"):
            with self.subTest(sequence=seq), self.assertRaises(ValueError):
                self.book.receive("terminal", sequence=seq, kind="terminal", order_id="a")
        self.assertEqual(self.book.command_count(), 0)

    def test_local_sequence_corruption_fails_reopen(self):
        self.book.reserve("r1", "a", **bounds())
        self.fill()
        with sqlite3.connect(self.path) as db:
            db.execute("DELETE FROM commands WHERE seq=1")
        with self.assertRaisesRegex(ValueError, "corrupted"):
            self.cls(self.path, cash=100, cash_floor=5)

    def test_process_exit_before_commit_rolls_back_and_after_commit_replays(self):
        # The test connection terminates the real child process at the second
        # commit (first is initialization). No crash hooks in production code.
        code = """
import os, sqlite3, sys
from datetime import UTC, datetime
from unittest.mock import patch
from experiments.receipt_journal import ReceiptJournal
class CrashConnection(sqlite3.Connection):
    count = 0
    def commit(self):
        self.count += 1
        if self.count == 2 and sys.argv[2] == "before": os._exit(73)
        super().commit()
        if self.count == 2: os._exit(74)
original = sqlite3.connect
with patch("sqlite3.connect", lambda *a, **kw: original(*a, factory=CrashConnection, **kw)):
    book = ReceiptJournal(sys.argv[1], cash=100, cash_floor=5)
    t = datetime(2026, 1, 1, tzinfo=UTC)
    book.reserve("r1", "a", desired_shares=10, price_cap=10, lot_size=1,
                 fixed_fee=1, fee_rate=0, max_fills=2, information_at=t, decision_at=t)
"""
        for phase, expected in (("before", 0), ("after", 92)):
            path = Path(self.tmp.name) / f"crash-{phase}"
            result = subprocess.run(
                [sys.executable, "-c", code, str(path), phase],
                capture_output=True,
                text=True,
                timeout=20,
            )
            self.assertEqual(result.returncode, 73 if phase == "before" else 74, result.stderr)
            book = self.cls(path, cash=100, cash_floor=5)
            self.assertEqual(book.snapshot().reserved, expected)
            book.reserve("r1", "a", **bounds())
            self.assertEqual(book.snapshot().reserved, 92)
            book.close()

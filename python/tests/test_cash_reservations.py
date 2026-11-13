"""Offline cash reservations: no broker, model or empirical performance claims."""

import unittest
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from fractions import Fraction as F

from experiments.cash_reservations import CashReservations
from experiments.fill_ledger import Fill

T = datetime(2026, 1, 1, tzinfo=UTC)


def reserve(book, name, **changes):
    args = dict(
        desired_shares=10,
        price_cap=10,
        lot_size=1,
        fixed_fee=1,
        fee_rate=0,
        max_fills=2,
        information_at=T,
        decision_at=T,
    )
    args.update(changes)
    return book.reserve(name, **args)


def fill(name="a", quantity=2, price=9, fee=1, seconds=1):
    return Fill(name, T + timedelta(seconds=seconds), F(quantity), F(price), F(fee), False)


class ReservationTests(unittest.TestCase):
    def test_second_order_cannot_reuse_first_cash(self):
        book = CashReservations(100, cash_floor=5)
        a = reserve(book, "a")
        b = reserve(book, "b")
        self.assertEqual((a.shares, b.shares), (9, 0))
        self.assertEqual(
            (book.snapshot().cash, book.snapshot().reserved, book.snapshot().available),
            (100, 92, 3),
        )

    def test_partial_fill_releases_only_price_improvement_and_consumed_fee(self):
        book = CashReservations(100, cash_floor=5)
        reserve(book, "a")
        book.fill("execution-1", fill())
        s = book.snapshot()
        self.assertEqual((s.cash, s.reserved, s.available), (81, 71, 5))
        book.fill("execution-2", fill(quantity=7, price=10, seconds=2))
        self.assertEqual((book.snapshot().cash, book.snapshot().reserved), (10, 0))

    def test_duplicate_fill_is_idempotent_but_mutation_is_rejected(self):
        book = CashReservations(100)
        reserve(book, "a")
        book.fill("x", fill())
        before = book.snapshot()
        book.fill("x", fill())
        self.assertEqual(book.snapshot(), before)
        with self.assertRaises(ValueError):
            book.fill("x", fill(quantity=3))
        self.assertEqual(book.snapshot(), before)

    def test_bad_events_leave_state_unchanged(self):
        for bad in (
            fill(price=11),
            fill(fee=2),
            fill(quantity=11),
            fill(seconds=0),
            fill(name="unknown"),
            fill(quantity=-1),
        ):
            book = CashReservations(100)
            reserve(book, "a")
            before = book.snapshot()
            with self.assertRaises(ValueError):
                book.fill("bad", bad)
            self.assertEqual(book.snapshot(), before)

    def test_close_is_explicit_not_cancel_request_and_late_fill_rejected(self):
        book = CashReservations(100)
        reserve(book, "a")
        book.fill("x", fill())
        book.finalize("a")
        self.assertEqual(book.snapshot().reserved, 0)
        with self.assertRaises(ValueError):
            book.fill("late", fill(seconds=2))
        book.fill("x", fill())  # Redelivery after finalization is still harmless.

    def test_fills_exhausted_does_not_assume_cancellation(self):
        book = CashReservations(100)
        reserve(book, "a", max_fills=1)
        book.fill("x", fill())
        self.assertEqual(book.snapshot().reserved, 70)
        with self.assertRaises(ValueError):
            book.fill("y", fill(seconds=2))
        book.finalize("a")
        self.assertEqual(book.snapshot().reserved, 0)

    def test_zero_quantity_and_order_ids(self):
        book = CashReservations(0)
        self.assertEqual(reserve(book, "a").shares, 0)
        with self.assertRaises(ValueError):
            reserve(book, "a")
        with self.assertRaises(ValueError):
            reserve(book, "")

    def test_multithreaded_reserve_cannot_oversubscribe(self):
        book = CashReservations(1000)
        with ThreadPoolExecutor(max_workers=8) as pool:
            budgets = list(pool.map(lambda i: reserve(book, str(i)), range(50)))
        self.assertLessEqual(sum(b.worst_case_spend for b in budgets), 1000)
        self.assertGreaterEqual(book.snapshot().available, 0)

    def test_fractional_boundary_is_exact(self):
        book = CashReservations("1/3", cash_floor="1/30")
        b = reserve(
            book, "a", desired_shares=1, price_cap=1, lot_size="1/10", fixed_fee=0, max_fills=1
        )
        self.assertEqual(b.shares, F(3, 10))
        self.assertEqual(book.snapshot().available, 0)


class ReplayAndRaceTests(unittest.TestCase):
    def test_duplicate_delivery_race_debits_once(self):
        book = CashReservations(100)
        reserve(book, "a")
        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(lambda _: book.fill("same", fill()), range(100)))
        self.assertEqual(book.snapshot().cash, 81)
        self.assertEqual(book.snapshot().revision, 2)

    def test_finalize_and_fill_race_has_only_two_serial_outcomes(self):
        for _ in range(40):
            book = CashReservations(100)
            reserve(book, "a")

            def accept_or_reject():
                try:
                    book.fill("x", fill())
                    return True
                except ValueError:
                    return False

            with ThreadPoolExecutor(max_workers=2) as pool:
                f = pool.submit(accept_or_reject)
                c = pool.submit(book.finalize, "a")
                accepted = f.result()
                c.result()
            self.assertEqual(book.snapshot().cash, 81 if accepted else 100)
            self.assertEqual(book.snapshot().reserved, 0)

    def test_replay_and_mutations(self):
        import copy

        from experiments.cash_reservation_study import run_study

        report = run_study(17, 20)
        self.assertEqual(report, run_study(17, 20))
        mutated = copy.deepcopy(report)
        mutated["runs"][0]["receipts"][0]["snapshot"]["cash"] = "-1"
        self.assertNotEqual(mutated, run_study(17, 20))
        self.assertEqual(report["negative_control"]["independent_budget_total"], "184")

    def test_unsupported_numeric_or_budget_override_does_not_mutate(self):
        book = CashReservations(100)
        reserve(book, "a")
        before = book.snapshot()
        with self.assertRaises(ValueError):
            book.fill("bad", Fill("a", T + timedelta(seconds=1), 2.0, F(9), F(1), False))
        with self.assertRaises(ValueError):
            reserve(book, "b", cash=100000)
        with self.assertRaises(ValueError):
            reserve(book, "b", information_at=T + timedelta(seconds=1))
        self.assertEqual(before, book.snapshot())


class FutureEnvelopeEnumeration(unittest.TestCase):
    def test_all_small_future_fill_paths_stay_within_reserve(self):
        from itertools import product

        checked = 0
        for q in range(1, 7):
            for slots in range(1, 4):
                for fixed in (F(0), F(1, 3)):
                    book = CashReservations(1000)
                    reserve(
                        book,
                        "a",
                        desired_shares=q,
                        max_fills=slots,
                        fixed_fee=fixed,
                        fee_rate=F(1, 10),
                    )
                    available_reserve = book.snapshot().reserved
                    # Independent finite path oracle: sum actual execution costs
                    # across every integral-share split and endpoint-price path.
                    for k in range(1, slots + 1):
                        for amounts in product(range(1, q + 1), repeat=k):
                            if sum(amounts) > q:
                                continue
                            for prices in product((F(1), F(10)), repeat=k):
                                cost = sum(
                                    (a * p + a * p / 10 + fixed for a, p in zip(amounts, prices)),
                                    F(0),
                                )
                                self.assertLessEqual(cost, available_reserve)
                                checked += 1
        self.assertGreater(checked, 1000)


if __name__ == "__main__":
    unittest.main()

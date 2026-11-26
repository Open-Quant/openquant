"""Independent exact expectations for the offline receipt comparator."""

from fractions import Fraction
from unittest import TestCase

from experiments.receipt_order_conformance import Execution, Receipt, compare


def trade(order, ordinal, shares, price=10, fee=1, *, at=4):
    return Execution(order, ordinal, at, Fraction(shares), Fraction(price), Fraction(fee))


def receipt(sequence, execution, identity=None):
    return Receipt(sequence, identity or f"source-{sequence}", execution)


class OrderConformanceTest(TestCase):
    def test_equal_portfolio_can_hide_swapped_order_assignments(self):
        expected = [trade("buy", 1, 2), trade("sell", 1, -2)]
        observed = [receipt(1, trade("buy", 1, -2)), receipt(2, trade("sell", 1, 2))]
        result = compare(expected, observed, through=2)
        self.assertTrue(result["account_equal"])
        self.assertEqual(result["status"], "mismatch")
        self.assertEqual(result["order_mismatches"], ["buy", "sell"])
        self.assertFalse(result["external_engine_validated"])

    def test_equal_order_balances_can_hide_execution_level_errors(self):
        expected = [trade("buy", 1, 1, 9), trade("buy", 2, 1, 11, at=5)]
        observed = [receipt(1, trade("buy", 1, 1, 10)), receipt(2, trade("buy", 2, 1, 10, at=5))]
        result = compare(expected, observed, through=2)
        self.assertTrue(result["account_equal"])
        self.assertEqual(result["order_mismatches"], [])
        self.assertEqual(result["changed_executions"], [("buy", 1), ("buy", 2)])
        self.assertEqual(result["status"], "mismatch")

    def test_exact_decimal_cash_direction_and_fees(self):
        expected = [trade("sell", 1, "-1/2", "20.1", "0.003")]
        result = compare(expected, [receipt(1, expected[0])], through=1)
        self.assertEqual(result["status"], "matched_declared_synthetic_tape")
        self.assertEqual(result["expected_account"], ("-1/2", "10047/1000", "3/1000"))
        self.assertFalse(result["terminal_authority"])

    def test_shuffled_redelivery_is_not_a_second_fill(self):
        tape = [trade("A", 1, 1), trade("B", 1, 2), trade("A", 2, 3)]
        rows = [receipt(i, x) for i, x in enumerate(tape, 1)]
        result = compare(tape, [rows[2], rows[0], rows[1], rows[0]], through=3)
        self.assertEqual(result["status"], "matched_declared_synthetic_tape")
        self.assertEqual(result["redeliveries"], 1)
        self.assertEqual(result["verified_prefix_through"], 3)
        self.assertEqual(result["buffered_receipts"], 0)

    def test_gap_blocks_absence_claim_but_not_a_known_prefix_mismatch(self):
        tape = [trade("A", 1, 1), trade("B", 1, 2), trade("A", 2, 3)]
        delayed = receipt(3, trade("unrelated", 1, 9))
        partial = compare(tape, [receipt(1, tape[0]), delayed], through=3)
        self.assertEqual(partial["status"], "incomplete_declared_prefix")
        self.assertIsNone(partial["account_equal"])
        self.assertEqual(partial["missing_executions"], [])
        self.assertEqual(partial["unexpected_executions"], [])
        self.assertEqual(partial["verified_prefix_through"], 1)
        self.assertEqual(partial["buffered_receipts"], 1)
        wrong = compare(tape, [receipt(1, trade("A", 1, 2)), delayed], through=3)
        self.assertEqual(wrong["status"], "mismatch")
        self.assertEqual(wrong["changed_executions"], [("A", 1)])
        self.assertIsNone(wrong["account_equal"])

    def test_identity_sequence_and_execution_key_are_independent(self):
        tr = trade("A", 1, 1)
        for rows, message in (
            ([receipt(1, tr), receipt(2, tr, "source-1")], "conflicting receipt"),
            ([receipt(1, tr), receipt(1, tr, "another")], "conflicting receipt"),
            ([receipt(1, tr), receipt(2, tr)], "duplicate order-local"),
        ):
            with self.subTest(rows=rows), self.assertRaisesRegex(ValueError, message):
                compare([tr], rows, through=2)

    def test_missing_and_unexpected_keys_are_not_paired_by_amount(self):
        expected = [trade("expected", 1, 1)]
        wrong = [receipt(1, trade("other", 1, 1))]
        result = compare(expected, wrong, through=1)
        self.assertTrue(result["account_equal"])
        self.assertEqual(result["missing_executions"], [("expected", 1)])
        self.assertEqual(result["unexpected_executions"], [("other", 1)])
        self.assertEqual(
            result["order_account_differences"],
            {"expected": (("1", "-11", "1"), None), "other": (None, ("1", "-11", "1"))},
        )

    def test_events_after_declared_cutoff_cannot_change_prefix(self):
        tr = trade("A", 1, 1)
        at_one = [receipt(1, tr)]
        self.assertEqual(
            compare([tr], at_one, through=1),
            compare([tr], at_one + [receipt(2, trade("X", 1, 3), "source-1")], through=1),
        )

    def test_seeded_independent_integer_cents_oracle(self):
        from random import Random

        rng = Random(291122)
        for case in range(128):
            n = rng.randrange(1, 12)
            # Embedded reference prices and fees are integer cents; only the
            # input adapter and output decoder use fractions.
            rows = [
                (
                    rng.choice((-1, 1)) * rng.randrange(1, 10),
                    rng.randrange(1, 9999),
                    rng.randrange(41),
                )
                for _ in range(n)
            ]
            sample = [
                trade(str(i), 1, shares, Fraction(price, 100), Fraction(fee, 100), at=i)
                for i, (shares, price, fee) in enumerate(rows, 1)
            ]
            source = [receipt(i, row) for i, row in enumerate(sample, 1)]
            copies = rng.choices(source, k=3)
            received = source + copies
            rng.shuffle(received)
            independent_totals = (
                sum(shares for shares, _, _ in rows),
                Fraction(sum(-shares * price - fee for shares, price, fee in rows), 100),
                Fraction(sum(fee for _, _, fee in rows), 100),
            )
            with self.subTest(case=case):
                exact = compare(sample, received, through=n)
                self.assertEqual(
                    tuple(map(Fraction, exact["observed_account"])),
                    independent_totals,
                )
                self.assertEqual(exact["status"], "matched_declared_synthetic_tape")
                self.assertEqual(exact["redeliveries"], 3)
                gap = rng.randrange(1, n + 1)
                without_one = [r for r in received if r.sequence != gap]
                withheld = compare(sample, without_one, through=n)
                self.assertEqual(withheld["status"], "incomplete_declared_prefix")
                self.assertEqual(withheld["verified_prefix_through"], gap - 1)
                self.assertEqual(withheld["buffered_receipts"], n - gap)

    def test_execution_time_difference_not_masked_by_equal_accounting(self):
        original = trade("A", 1, 1, at=1)
        shifted = trade("A", 1, 1, at=2)
        result = compare([original], [receipt(1, shifted)], through=1)
        self.assertTrue(result["account_equal"])
        self.assertEqual(result["order_account_differences"], {})
        self.assertEqual(result["changed_executions"], [("A", 1)])
        self.assertEqual(result["status"], "mismatch")

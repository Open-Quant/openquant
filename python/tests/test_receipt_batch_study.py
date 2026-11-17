"""Contract and independent cash arithmetic for the synthetic batching study."""

import importlib.util
import unittest


class StudyTest(unittest.TestCase):
    def test_synthetic_study_exists(self):
        self.assertIsNotNone(importlib.util.find_spec("experiments.receipt_batch_study"))

    def test_chunk_sizes_and_restarts_match_exact_cash(self):
        from experiments.receipt_batch_study import run

        results = [run(seed=41, orders=3, batch_size=n, restart_every=2) for n in (1, 2, 7, 256)]
        self.assertEqual(len({r["final_cash"] for r in results}), 1)
        self.assertEqual({r["durable_commands"] for r in results}, {12})
        self.assertEqual({r["source_watermark"] for r in results}, {9})
        for result in results:
            self.assertTrue(result["independent_cash_match"])
            self.assertTrue(result["identical_journal_to_sequential"])
            self.assertGreaterEqual(result["ingest_seconds"], 0)
            self.assertGreaterEqual(result["total_seconds"], result["ingest_seconds"])

    def test_restart_schedule_is_receipt_count_not_batch_count(self):
        from experiments.receipt_batch_study import run

        results = [run(seed=1, orders=3, batch_size=n, restart_every=5) for n in (1, 4, 256)]
        self.assertEqual({r["restarts"] for r in results}, {3})

    def test_invalid_workload_rejected(self):
        from experiments.receipt_batch_study import run

        for values in (
            {"orders": 0},
            {"orders": True},
            {"batch_size": 0},
            {"batch_size": 257},
            {"restart_every": -1},
        ):
            with self.subTest(values=values), self.assertRaises(ValueError):
                run(**values)

    def test_campaign_retains_completed_case_before_later_failure(self):
        from experiments.receipt_batch_study import campaign

        records = campaign(order_counts=[1, 0], batch_sizes=[2], repeats=1)
        start = next(records)
        self.assertEqual(start["record"], "start")
        self.assertFalse(start["complete"])
        first = next(records)
        self.assertEqual(first["record"], "result")
        self.assertEqual(first["result"]["orders"], 1)
        with self.assertRaisesRegex(ValueError, "orders"):
            next(records)

    def test_campaign_completion_has_bound_source_identity(self):
        from experiments.receipt_batch_study import campaign

        records = list(campaign(order_counts=[1], batch_sizes=[2], repeats=1))
        self.assertEqual([r["record"] for r in records], ["start", "result", "complete"])
        provenance = records[0]["provenance"]
        self.assertEqual(len(provenance["source_sha256"]), 5)
        self.assertTrue(all(len(value) == 64 for value in provenance["source_sha256"].values()))
        self.assertTrue(records[-1]["complete"])
        self.assertEqual(records[-1]["result_count"], 1)

    def test_campaign_aborts_if_sources_change(self):
        from unittest.mock import patch

        from experiments.receipt_batch_study import campaign

        records = campaign(order_counts=[1], batch_sizes=[2])
        next(records)
        with patch("experiments.receipt_batch_study._source_identity", return_value={}):
            with self.assertRaisesRegex(RuntimeError, "Source changed"):
                next(records)

"""Portable exact execution-cost fixtures, not strategy-performance tests."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


class ConformanceTest(unittest.TestCase):
    def test_adapter_exists(self):
        self.assertIsNotNone(importlib.util.find_spec("experiments.execution_cost_conformance"))

    def test_all_five_exact_contract_fixtures_pass(self):
        from experiments.execution_cost_conformance import run

        report = run()
        self.assertEqual(report["schema_version"], 1)
        self.assertEqual(len(report["cases"]), 5)
        self.assertTrue(all(case["passed"] for case in report["cases"]))
        self.assertTrue(report["synthetic_only"])
        self.assertEqual(report["adapters_tested"], ["experimental_openquant_fill_ledger"])

    def test_wrong_rate_scale_and_fee_frequency_negative_controls_detected(self):
        from experiments.execution_cost_conformance import run

        report = run()
        self.assertTrue(all(item["detected"] for item in report["negative_controls"].values()))
        self.assertEqual(
            set(report["negative_controls"]),
            {
                "rate_divided_by_100",
                "one_fee_per_order",
                "omit_terminal_charge",
                "erase_terminal_inventory",
            },
        )

    def test_full_paths_compared_not_only_final_equity(self):
        from experiments.execution_cost_conformance import compare

        expected = {"cash": ["78.98", "53.956"], "terminal_completed": True}
        observed = {"cash": ["80", "53.956"], "terminal_completed": True}
        differences = compare(expected, observed)
        self.assertEqual(differences, ["cash"])
        self.assertEqual(
            compare(expected, {"cash": ["3949/50", "13489/250"], "terminal_completed": True}), []
        )

    def test_schema_mismatch_is_not_silently_compared(self):
        from experiments.execution_cost_conformance import compare

        with self.assertRaisesRegex(ValueError, "fields"):
            compare({"cash": ["1"]}, {"equity": ["1"]})

    def test_unknown_account_or_ordering_contract_is_rejected(self):
        import experiments.execution_cost_conformance as module

        source = json.loads(module.FIXTURE.read_text())
        for field, wrong in (
            ("account", "margin"),
            ("ordering", "same_bar_close"),
            ("currency", "EUR"),
            ("fee_rate_unit", "basis_points"),
        ):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                document = json.loads(json.dumps(source))
                document["contract"][field] = wrong
                path = Path(directory) / "fixture.json"
                path.write_text(json.dumps(document))
                with (
                    patch.object(module, "FIXTURE", path),
                    self.assertRaisesRegex(ValueError, "contract"),
                ):
                    module.run()

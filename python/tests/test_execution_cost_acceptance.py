"""Independent, synthetic execution-contract checks, not external-engine evidence."""

import copy
import json
import subprocess
import sys
import unittest

from experiments.execution_cost_conformance import FIXTURE


class AcceptanceTests(unittest.TestCase):
    def test_new_gate_reports_two_independent_implementations(self):
        from experiments.execution_cost_acceptance import run

        result = run()
        self.assertTrue(result["passed"])
        self.assertGreaterEqual(len(result["cases"]), 10)
        self.assertEqual(
            result["implementations"], ["openquant_experimental", "rational_reference"]
        )
        self.assertFalse(result["external_engine_validated"])

    def test_original_expected_paths_unchanged(self):
        from experiments.execution_cost_reference import observe

        d = json.loads(FIXTURE.read_text())
        from experiments.execution_cost_conformance import compare

        for case in d["cases"]:
            with self.subTest(case=case["id"]):
                self.assertEqual(compare(case["expected"], observe(d, case)), [])

    def test_reference_has_no_ledger_dependency(self):
        import ast
        from pathlib import Path

        source = Path("experiments/execution_cost_reference.py").read_text()
        imports = [
            node
            for node in ast.walk(ast.parse(source))
            if isinstance(node, (ast.Import, ast.ImportFrom))
        ]
        self.assertFalse(any("fill_ledger" in ast.unparse(node) for node in imports))

    def test_bad_contract_returns_unsupported_not_pass(self):
        from experiments.execution_cost_acceptance import evaluate

        d = json.loads(FIXTURE.read_text())
        d["contract"]["fee_rate_unit"] = "bps"
        result = evaluate(d)
        self.assertFalse(result["passed"])
        self.assertEqual(result["status"], "unsupported")

    def test_missing_path_or_forged_outcome_cannot_pass(self):
        from experiments.execution_cost_acceptance import evaluate

        source = json.loads(FIXTURE.read_text())
        for edit in ("missing", "cash", "empty"):
            d = copy.deepcopy(source)
            if edit == "missing":
                del d["cases"][0]["expected"]["cash"]
            elif edit == "cash":
                d["cases"][0]["expected"]["cash"][0] = "81"
            else:
                d["cases"] = []
            self.assertFalse(evaluate(d)["passed"])

    def test_cli_has_nonzero_exit_on_bad_fixture(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / "bad.json"
            d = json.loads(FIXTURE.read_text())
            d["cases"][0]["expected"]["cash"][0] = "81"
            p.write_text(json.dumps(d))
            r = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "experiments.execution_cost_acceptance",
                    "--fixture",
                    str(p),
                ],
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(r.returncode, 0)
            self.assertFalse(json.loads(r.stdout)["passed"])

    def test_diagnostics_identify_exact_row(self):
        from experiments.execution_cost_acceptance import differences

        self.assertEqual(
            differences({"cash": ["80", "56"]}, {"cash": ["81", "56"]}),
            [{"field": "cash", "index": 0, "expected": "80", "observed": "81"}],
        )

    def test_supported_fixtures_reject_unknown_fields(self):
        from experiments.execution_cost_acceptance import FIXTURE as V2
        from experiments.execution_cost_acceptance import evaluate

        source = json.loads(V2.read_text())
        for where in ("root", "case", "decision", "initial", "bucket"):
            d = copy.deepcopy(source)
            target = {
                "root": d,
                "case": d["cases"][0],
                "decision": d["cases"][0]["scenario"]["decision"],
                "initial": d["cases"][0]["scenario"]["initial"],
                "bucket": d["cases"][0]["scenario"]["buckets"][0],
            }[where]
            target["hidden_netting"] = True
            with self.subTest(where=where):
                self.assertFalse(evaluate(d)["passed"])

    def test_reference_does_not_consume_expected_values(self):
        from experiments.execution_cost_reference import observe

        source = json.loads(FIXTURE.read_text())
        case = source["cases"][0]
        before = observe(source, case)
        case["expected"] = {"invented": "should never be read"}
        self.assertEqual(before, observe(source, case))

    def test_unsupported_and_invalid_fixture_negative_controls(self):
        from experiments.execution_cost_acceptance import FIXTURE as V2
        from experiments.execution_cost_acceptance import evaluate

        source = json.loads(V2.read_text())
        edits = [
            lambda d: d.update(schema_version=True),
            lambda d: d["cases"][0].update(id=""),
            lambda d: d["cases"].append(copy.deepcopy(d["cases"][0])),
            lambda d: d["cases"][0].update(fixed_fee="-1"),
            lambda d: d["cases"][0].update(fee_rate="NaN"),
            lambda d: d["cases"][0].update(volumes=["2"]),
            lambda d: d["cases"][0]["scenario"].update(participation="2"),
            lambda d: d["cases"][0]["scenario"]["decision"].update(decision_at=1.5),
            lambda d: d["cases"][0]["scenario"]["buckets"][0].update(bid="11"),
            lambda d: d["cases"][0]["expected"].update(terminal_completed=1),
            lambda d: d["cases"][0]["expected"].update(fill_prices=[]),
        ]
        for index, edit in enumerate(edits):
            with self.subTest(index=index):
                d = copy.deepcopy(source)
                edit(d)
                self.assertFalse(evaluate(d)["passed"])

    def test_seeded_single_order_differential(self):
        from fractions import Fraction as Q
        from random import Random

        from experiments.execution_cost_conformance import _observe, compare
        from experiments.execution_cost_reference import observe

        source = json.loads(FIXTURE.read_text())
        rng = Random(1171)
        accepted = rejected = 0
        for index in range(400):
            d = copy.deepcopy(source)
            s, c = d["scenario"], d["cases"][0]
            s["initial"].update(cash=str(rng.randrange(0, 90)), inventory=str(rng.randrange(0, 8)))
            s["decision"].update(
                shares=str(rng.choice((-7, -3, 2, 6))),
                decision_at=rng.choice((0, 1)),
                earliest_fill_at=1,
                deadline=rng.choice((1, 2)),
            )
            s["participation"] = rng.choice(("0", "1/3", "1/2", "1"))
            c.update(
                fixed_fee=rng.choice(("0", "1/10", "2")),
                fee_rate=rng.choice(("0", "1/1000", "1/100")),
                volumes=[str(rng.randrange(0, 8)) for _ in range(2)],
                terminal_volume=rng.choice((None, "0", "2", "9")),
            )
            results = []
            for f in (_observe, observe):
                try:
                    results.append(f(d, c))
                except ValueError as exc:
                    results.append("cash" if "cash" in str(exc) else "inventory")
            with self.subTest(index=index):
                if isinstance(results[0], str):
                    self.assertEqual(results[0], results[1])
                    rejected += 1
                else:
                    self.assertIsInstance(results[1], dict)
                    self.assertEqual(compare(results[0], results[1]), [])
                    self.assertTrue(all(Q(x) >= 0 for x in results[1]["cash"]))
                    accepted += 1
        self.assertGreater(accepted, 150)
        self.assertGreater(rejected, 50)

    def test_duplicate_json_keys_fail_closed(self):
        import tempfile
        from pathlib import Path

        from experiments.execution_cost_acceptance import run

        raw = FIXTURE.read_text().replace('"cash": "100"', '"cash": "10", "cash": "100"')
        self.assertIn('"cash": "10", "cash": "100"', raw)
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / "duplicate.json"
            p.write_text(raw)
            with self.assertRaisesRegex(ValueError, "duplicate"):
                run(p)

    def test_consensus_is_not_acceptance_when_expected_path_disagrees(self):
        from experiments.execution_cost_acceptance import evaluate

        d = json.loads(FIXTURE.read_text())
        d["cases"][0]["expected"]["cash"][0] = "81"
        result = evaluate(d)
        self.assertFalse(result["passed"])
        outcomes = result["cases"][0]["outcomes"]
        self.assertEqual(
            outcomes["openquant_experimental"]["observed"],
            outcomes["rational_reference"]["observed"],
        )
        self.assertFalse(outcomes["openquant_experimental"]["passed"])
        self.assertFalse(outcomes["rational_reference"]["passed"])

    def test_near_matching_exception_is_not_a_qualified_rejection(self):
        from unittest.mock import patch

        from experiments.execution_cost_acceptance import FIXTURE as V2
        from experiments.execution_cost_acceptance import evaluate

        d = json.loads(V2.read_text())
        d["cases"] = [c for c in d["cases"] if c["id"] == "cash_exhaustion_rejected"]
        with patch(
            "experiments.execution_cost_acceptance._observe",
            side_effect=ValueError("cash would become negatively unrelated"),
        ):
            result = evaluate(d)
        self.assertFalse(result["passed"])

    def test_report_binds_fixture_and_implementation_sources(self):
        import hashlib
        from pathlib import Path

        from experiments.execution_cost_acceptance import FIXTURE as V2
        from experiments.execution_cost_acceptance import run

        report = run()
        self.assertEqual(report["fixture_sha256"], hashlib.sha256(V2.read_bytes()).hexdigest())
        hashes = report["implementation_source_sha256"]
        self.assertEqual(
            set(hashes),
            {
                "execution_cost_acceptance.py",
                "execution_cost_reference.py",
                "execution_cost_conformance.py",
                "fill_ledger.py",
            },
        )
        for name, value in hashes.items():
            self.assertEqual(
                value, hashlib.sha256((Path("experiments") / name).read_bytes()).hexdigest()
            )

"""The performance gate must fail closed on unusable timing measurements."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GATE = ROOT / "scripts" / "check_bench_thresholds.py"


class BenchmarkThresholdTests(unittest.TestCase):
    def run_gate(self, baseline=None, latest=None, *, overrides=None, options=()):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = {"example": {"mean_ms": 1.0}} if baseline is None else baseline
            latest = {"example": {"mean_ms": 1.1}} if latest is None else latest
            for name, benchmarks in (("baseline", baseline), ("latest", latest)):
                (root / f"{name}.json").write_text(json.dumps({"benchmarks": benchmarks}))
            command = [
                sys.executable,
                str(GATE),
                "--baseline",
                str(root / "baseline.json"),
                "--latest",
                str(root / "latest.json"),
            ]
            if overrides is not None:
                (root / "overrides.json").write_text(json.dumps(overrides))
                command += ["--overrides", str(root / "overrides.json")]
            return subprocess.run(command + list(options), capture_output=True, text=True)

    def test_finite_timings_below_and_at_threshold_pass(self):
        for value in (1, 1.1, 1.25):
            with self.subTest(value=value):
                result = self.run_gate(latest={"example": {"mean_ms": value}})
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_regression_above_threshold_fails(self):
        result = self.run_gate(latest={"example": {"mean_ms": 1.251}})
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("regressions above threshold", result.stdout)

    def test_invalid_timings_never_pass(self):
        for source in ("baseline", "latest"):
            for value in (float("nan"), float("inf"), -float("inf"), 0.0, -1.0, True):
                with self.subTest(source=source, value=value):
                    result = self.run_gate(**{source: {"example": {"mean_ms": value}}})
                    self.assertNotEqual(result.returncode, 0, result.stdout)
                    self.assertIn("example", result.stderr)
                    self.assertIn("mean_ms", result.stderr)

    def test_invalid_timing_below_cutoff_is_not_silently_skipped(self):
        result = self.run_gate(
            baseline={"example": {"mean_ms": 1.0}, "tiny": {"mean_ms": 0.01}},
            latest={"example": {"mean_ms": 1.0}, "tiny": {"mean_ms": float("nan")}},
        )
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn("tiny", result.stderr)

    def test_invalid_new_benchmark_is_not_silently_skipped(self):
        result = self.run_gate(
            latest={"example": {"mean_ms": 1.0}, "new": {"mean_ms": float("nan")}}
        )
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn("new", result.stderr)

    def test_invalid_global_thresholds_never_disable_gate(self):
        for option in ("--max-regression-pct", "--min-baseline-ms"):
            for value in ("nan", "inf", "-1"):
                with self.subTest(option=option, value=value):
                    result = self.run_gate(options=(option, value))
                    self.assertNotEqual(result.returncode, 0, result.stdout)
                    self.assertIn(option, result.stderr)

    def test_invalid_override_never_disables_gate(self):
        for value in (float("nan"), float("inf"), -1.0, True):
            with self.subTest(value=value):
                result = self.run_gate(overrides={"example": value})
                self.assertNotEqual(result.returncode, 0, result.stdout)
                self.assertIn("example", result.stderr)

    def test_malformed_measurements_fail_with_actionable_errors(self):
        for measurement in ({}, None, [], {"mean_ms": "fast"}, {"mean_ms": 10**400}):
            with self.subTest(measurement=measurement):
                result = self.run_gate(latest={"example": measurement})
                self.assertNotEqual(result.returncode, 0, result.stdout)
                self.assertIn("example", result.stderr)
                self.assertIn("mean_ms", result.stderr)
                self.assertNotIn("Traceback", result.stderr)

    def test_malformed_benchmark_containers_are_rejected(self):
        for benchmarks in ([], "invalid"):
            with self.subTest(benchmarks=benchmarks):
                result = self.run_gate(baseline=benchmarks)
                self.assertNotEqual(result.returncode, 0, result.stdout)
                self.assertIn("baseline", result.stderr)
                self.assertNotIn("Traceback", result.stderr)

    def test_malformed_overrides_are_rejected(self):
        for overrides in ([], {"example": "nan"}, {"example": 10**400}):
            with self.subTest(overrides=overrides):
                result = self.run_gate(overrides=overrides)
                self.assertNotEqual(result.returncode, 0, result.stdout)
                self.assertIn("override", result.stderr)
                self.assertNotIn("Traceback", result.stderr)

    def test_finite_override_is_applied(self):
        result = self.run_gate(latest={"example": {"mean_ms": 1.5}}, overrides={"example": 50.0})
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_zero_threshold_and_cutoff_are_valid(self):
        result = self.run_gate(
            latest={"example": {"mean_ms": 1.0}},
            options=("--max-regression-pct", "0", "--min-baseline-ms", "0"),
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_missing_and_empty_results_still_fail(self):
        for baseline, latest in (({"example": {"mean_ms": 1.0}}, {}), ({}, {})):
            with self.subTest(baseline=baseline, latest=latest):
                result = self.run_gate(baseline, latest)
                self.assertNotEqual(result.returncode, 0, result.stdout)


if __name__ == "__main__":
    unittest.main()

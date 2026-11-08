"""Executable multi-source-to-training flow, beyond isolated projection tests."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "required_features_example", ROOT / "examples/required_features.py"
)
assert spec is not None and spec.loader is not None
example = importlib.util.module_from_spec(spec)
spec.loader.exec_module(example)


def test_missing_inputs_are_not_a_complete_timestamp_or_training_row():
    report = example.run(*example.modules(True))
    assert report["feature_order"] == ["rate", "price"]
    assert report["requested_ids"] == ["D", "B", "A", "C"]
    assert report["complete_ids"] == ["D", "A"]
    assert report["values"] == [[3.0, 40], [1.0, 10]]
    assert report["clock_only_baseline_ids"] == ["D", "B", "A", "C"]
    assert report["excluded"]["B"][0]["reason"] == "missing_source"
    assert report["excluded"]["C"][0]["reason"] == "missing_field"
    assert report["downstream_train_ids"] == ["D"]
    assert not report["withheld_complete_row_restored"]


def test_coverage_does_not_recover_the_unselected_target_distribution():
    _, assembly, provenance = example.modules(True)
    result = example.informative_coverage(assembly, provenance)
    assert (result["complete"], result["requested"]) == (10, 16)
    assert result["selected_target_mean"] == 0.6
    assert result["full_target_mean"] == 0.0


def test_isolated_example_runs_without_site_packages():
    checked = subprocess.run(
        [sys.executable, "-S", str(ROOT / "examples/required_features.py"), "--source-tree"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert checked.returncode == 0, checked.stderr
    result = json.loads(checked.stdout)
    assert result["import_mode"] == "isolated_source_namespace"
    assert result["downstream_train_ids"] == ["D"]

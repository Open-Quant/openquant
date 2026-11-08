"""Independent process-level source-change and exclusive-output contracts."""

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _runner():
    path = ROOT / "experiments/run_offline_lab.py"
    assert path.is_file(), "configured runner is not implemented"
    return path


@pytest.mark.parametrize("changed_dependency", [False, True])
def test_changed_loaded_source_is_not_blessed_by_new_report_hashes(tmp_path, changed_dependency):
    """A post-run hash alone would pass; the before/after snapshot must reject it."""
    runner = _runner()
    experiments = tmp_path / "checkout/experiments"
    experiments.mkdir(parents=True)
    shutil.copy2(runner, experiments / runner.name)
    shutil.copy2(
        ROOT / "experiments/replay_offline_report.py", experiments / "replay_offline_report.py"
    )
    lab_name = "adaptive_holdout" if changed_dependency else "self_financing"
    filename = "adaptive_holdout_lab.py" if changed_dependency else "self_financing_lab.py"
    dependency = "validation_lab.py" if changed_dependency else filename
    if changed_dependency:
        (experiments / dependency).write_text("# original trusted dependency\n")
    source = """import hashlib
from pathlib import Path
import numpy as np

def run_lab(*, seed=17):
    source = Path(__file__)
    changed = source.parent / REPLACE_DEPENDENCY
    changed.write_text(changed.read_text() + "\\n# changed during execution\\n")
    provenance = {"numpy": np.__version__, "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
    if REPLACE_CHANGED_DEPENDENCY:
        provenance["dependency_sha256"] = {REPLACE_DEPENDENCY: hashlib.sha256(changed.read_bytes()).hexdigest()}
    return {"schema_version": 1, "synthetic_only": True, "lab": REPLACE_LAB, "config": {"seed": seed}, "provenance": provenance}
"""
    source = (
        source.replace("REPLACE_CHANGED_DEPENDENCY", repr(changed_dependency))
        .replace("REPLACE_DEPENDENCY", repr(dependency))
        .replace("REPLACE_LAB", repr(lab_name))
    )
    (experiments / filename).write_text(source)
    config, output = tmp_path / "config.json", tmp_path / "output.json"
    config.write_bytes(b'{"seed": 17}\n')
    result = subprocess.run(
        [
            sys.executable,
            str(experiments / runner.name),
            "run",
            "--lab",
            lab_name,
            "--config",
            str(config),
            "--out",
            str(output),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode != 0, result.stdout
    assert "changed during execution" in (experiments / dependency).read_text()
    assert not output.exists()
    assert config.read_bytes() == b'{"seed": 17}\n'
    assert "source" in result.stderr.lower() or "snapshot" in result.stderr.lower()


def test_two_real_runner_processes_publish_exactly_one_complete_report(tmp_path):
    runner = _runner()
    config, output = tmp_path / "config.json", tmp_path / "result.json"
    config.write_text(
        json.dumps(
            {
                "scales": [1000],
                "volume_scales": [1],
                "impact_coefficients": [0.1],
                "gross_returns": [0.001],
            }
        )
    )
    original = config.read_bytes()
    command = [
        sys.executable,
        str(runner),
        "run",
        "--lab",
        "capacity_scaling",
        "--config",
        str(config),
        "--out",
        str(output),
    ]
    first = subprocess.Popen(command, cwd=tmp_path, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    second = subprocess.Popen(command, cwd=tmp_path, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    first_stdout, first_stderr = first.communicate(timeout=30)
    second_stdout, second_stderr = second.communicate(timeout=30)
    codes = [first.returncode, second.returncode]
    assert codes.count(0) == 1, (codes, first_stderr, second_stderr)
    raw = output.read_bytes()
    report = json.loads(raw)
    assert report["lab"] == "capacity_scaling"
    assert len(report["cells"]) == 1
    assert config.read_bytes() == original
    assert sorted(p.name for p in tmp_path.iterdir()) == ["config.json", "result.json"]
    winner_stdout = first_stdout if first.returncode == 0 else second_stdout
    receipt = json.loads(winner_stdout)
    assert receipt["scientific_replay_performed"] is False
    checked = subprocess.run(
        [
            sys.executable,
            str(ROOT / "experiments/replay_offline_report.py"),
            str(output),
            "--lab",
            "capacity_scaling",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert checked.returncode == 0, checked.stderr
    assert json.loads(checked.stdout)["matched"] is True
    assert hashlib.sha256(output.read_bytes()).digest() == hashlib.sha256(raw).digest()

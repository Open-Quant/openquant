"""Tiny frozen protocol fixtures cover every registered lab, not statistical validity.

Each JSON fixture supplies every run_lab parameter. At most two replications and
small fixed grids are used; source-defined fixed cases remain intact. No default
study is executed by these tests. Optional required SciPy cases skip without SciPy.
"""

from __future__ import annotations

import gzip
import hashlib
import importlib.util
import inspect
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS = ROOT / "experiments"
FIXTURES = EXPERIMENTS / "configs" / "offline-smoke"
LAB_NAMES = sorted(path.stem for path in FIXTURES.glob("*.json"))


def load_replay():
    spec = importlib.util.spec_from_file_location(
        "_offline_runner_matrix_replay", EXPERIMENTS / "replay_offline_report.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_cli(script, *args, module=False, cwd=ROOT):
    invocation = ["-m", f"experiments.{script}"] if module else [str(EXPERIMENTS / f"{script}.py")]
    result = subprocess.run(
        [sys.executable, *invocation, *map(str, args)],
        cwd=cwd,
        env=dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1"),
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return json.loads(result.stdout)


def require_dependencies(replay, name):
    for package in replay.EXTRA_RUNTIME_PACKAGES.get(name, ()):
        pytest.importorskip(package, reason=f"optional offline dependency for {name}")


def test_frozen_smoke_configs_cover_the_authoritative_registry():
    replay = load_replay()
    assert set(LAB_NAMES) == set(replay.LABS)
    assert len(LAB_NAMES) == len(set(LAB_NAMES))


@pytest.mark.parametrize("name", LAB_NAMES)
def test_registered_lab_fresh_process_run_identity_and_replay(name, tmp_path):
    replay = load_replay()
    require_dependencies(replay, name)
    config, output = tmp_path / "config.json", tmp_path / "nested" / "report.json"
    raw = (FIXTURES / f"{name}.json").read_bytes()
    config.write_bytes(raw)
    # Alternate both public invocations across the registry; scripts also work
    # from an unrelated directory, while modules are invoked at the repo root.
    module = bool(LAB_NAMES.index(name) % 2)
    receipt = run_cli(
        "run_offline_lab",
        "run",
        "--lab",
        name,
        "--config",
        config,
        "--out",
        output,
        module=module,
        cwd=ROOT if module else tmp_path,
    )
    saved = output.read_bytes()
    assert receipt["lab"] == name
    assert receipt["identity_matched"] is True
    assert receipt["scientific_replay_performed"] is False
    assert receipt["config_sha256"] == hashlib.sha256(raw).hexdigest()
    assert receipt["report_sha256"] == hashlib.sha256(saved).hexdigest()
    source_hash = hashlib.sha256((EXPERIMENTS / replay.LABS[name]).read_bytes()).hexdigest()
    assert receipt["source_sha256"] == source_hash
    identity = run_cli(
        "replay_offline_report", output, "--lab", name, "--check-identity", module=not module
    )
    assert identity["identity_matched"] is True
    assert identity["scientific_replay_performed"] is False
    verdict = run_cli("replay_offline_report", output, "--lab", name, module=module)
    assert verdict["matched"] is True
    assert verdict["first_difference"] is None
    assert verdict["saved_scientific_sha256"] == verdict["replayed_scientific_sha256"]
    assert config.read_bytes() == raw
    assert output.read_bytes() == saved
    assert list(output.parent.iterdir()) == [output]


@pytest.mark.parametrize("name", LAB_NAMES)
def test_defaults_match_signature_and_smoke_configs_bind(name, tmp_path):
    replay = load_replay()
    require_dependencies(replay, name)
    signature = inspect.signature(replay._load_lab(name).run_lab)
    expected = json.loads(
        json.dumps(
            {key: value.default for key, value in signature.parameters.items()}, allow_nan=False
        )
    )
    invocation = subprocess.run(
        [sys.executable, str(EXPERIMENTS / "run_offline_lab.py"), "defaults", "--lab", name],
        cwd=tmp_path,
        env=dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1"),
        capture_output=True,
        check=False,
        timeout=30,
    )
    assert invocation.returncode == 0, invocation.stderr
    saved = tmp_path / "defaults.json"
    saved.write_bytes(invocation.stdout)
    assert json.loads(saved.read_bytes()) == expected
    signature.bind(**json.loads(saved.read_bytes()))
    fixture = json.loads((FIXTURES / f"{name}.json").read_bytes())
    assert set(fixture) == set(signature.parameters)
    signature.bind(**fixture)
    assert list(tmp_path.iterdir()) == [saved]


def test_config_run_compress_and_replay_preserves_original_bytes(tmp_path):
    config, report, packed = (
        tmp_path / filename for filename in ("config.json", "report.json", "report.json.gz")
    )
    original = (FIXTURES / "point_in_time.json").read_bytes()
    config.write_bytes(original)
    run_cli("run_offline_lab", "run", "--lab", "point_in_time", "--config", config, "--out", report)
    raw_report = report.read_bytes()
    receipt = run_cli("compress_offline_report", report, "--out", packed)
    assert receipt["scientific_replay_performed"] is False
    assert receipt["byte_round_trip_verified"] is True
    assert receipt["source_deleted"] is False
    assert gzip.decompress(packed.read_bytes()) == raw_report
    verdict = run_cli("replay_offline_report", packed, "--lab", "point_in_time")
    assert verdict["matched"] is True
    assert verdict["saved_scientific_sha256"] == verdict["replayed_scientific_sha256"]
    assert config.read_bytes() == original
    assert report.read_bytes() == raw_report


def without_scipy(script, *arguments):
    """Model a clean process where the optional package is not installed."""
    code = """
import importlib.abc
import runpy
import sys

class BlockScipy(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'scipy' or fullname.startswith('scipy.'):
            raise ModuleNotFoundError("No module named 'scipy'", name='scipy')

sys.meta_path.insert(0, BlockScipy())
sys.argv = sys.argv[1:]
runpy.run_module(sys.argv[0], run_name='__main__')
"""
    return subprocess.run(
        [sys.executable, "-c", code, f"experiments.{script}", *map(str, arguments)],
        cwd=ROOT,
        env=dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1"),
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )


def test_listing_and_numpy_defaults_work_without_scipy():
    listing = without_scipy("run_offline_lab", "list")
    assert listing.returncode == 0, listing.stderr
    assert {entry["name"] for entry in json.loads(listing.stdout)["labs"]} == set(LAB_NAMES)
    defaults = without_scipy("run_offline_lab", "defaults", "--lab", "point_in_time")
    assert defaults.returncode == 0, defaults.stderr
    assert json.loads(defaults.stdout)["days"] == 64


def test_missing_required_scipy_fails_without_publishing(tmp_path):
    report = tmp_path / "report.json"
    result = without_scipy(
        "run_offline_lab",
        "run",
        "--lab",
        "selection_interval",
        "--config",
        FIXTURES / "selection_interval.json",
        "--out",
        report,
    )
    assert result.returncode != 0
    assert "scipy" in result.stderr.lower()
    assert not report.exists()


def test_covariance_optional_scipy_fallback_replays_in_fresh_process(tmp_path):
    report = tmp_path / "report.json"
    result = without_scipy(
        "run_offline_lab",
        "run",
        "--lab",
        "covariance_estimation",
        "--config",
        FIXTURES / "covariance_estimation.json",
        "--out",
        report,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(report.read_bytes())["provenance"]["scipy"] is None
    checked = without_scipy("replay_offline_report", report, "--lab", "covariance_estimation")
    assert checked.returncode == 0, checked.stdout + checked.stderr
    assert json.loads(checked.stdout)["matched"] is True

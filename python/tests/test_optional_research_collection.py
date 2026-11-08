"""Optional research packages must not break the default test environment."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_zero_fit_collection_without_optional_scipy(tmp_path: Path) -> None:
    sentinel = tmp_path / "test_available.py"
    sentinel.write_text("def test_available():\n    pass\n", encoding="utf-8")
    # A subprocess isolates this dependency-absence scenario even when the
    # enclosing science environment has SciPy installed. All pytest collection
    # and the target module remain real; no OpenQuant behavior is mocked.
    script = """
import importlib.abc
import sys
import pytest

class MissingSciPy(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "scipy" or fullname.startswith("scipy."):
            raise ModuleNotFoundError("SciPy is unavailable for this check", name=fullname)

sys.meta_path.insert(0, MissingSciPy())
raise SystemExit(pytest.main([
    "--collect-only", "-q", "-rs", "-p", "no:cacheprovider", sys.argv[1], sys.argv[2],
]))
"""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(ROOT / "python/tests/test_zero_fit_probability_lab.py"),
            str(sentinel),
        ],
        cwd=ROOT,
        env={**os.environ, "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"},
        capture_output=True,
        text=True,
        timeout=30,
    )
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert "SKIPPED" in output, output

"""Reference generation includes public importable modules outside package __all__."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def test_reference_includes_public_modules_without_root_reexports(tmp_path: Path) -> None:
    package = tmp_path / "openquant"
    package.mkdir()
    (package / "__init__.py").write_text(
        "from . import _core\n_CORE_REEXPORTS = []\n__all__ = []\n", encoding="utf-8"
    )
    (package / "_core.py").write_text("", encoding="utf-8")
    (package / "available.py").write_text(
        '"""An explicitly importable public module."""\n'
        'def select(value: int) -> int:\n    """Return the supplied value."""\n'
        "    return value\n",
        encoding="utf-8",
    )
    (package / "_private.py").write_text(
        'raise AssertionError("Private modules must not be imported by discovery")\n',
        encoding="utf-8",
    )
    generator = Path(__file__).resolve().parents[2] / "scripts/generate_python_api_reference.py"
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            'import json, runpy, sys; print(json.dumps(runpy.run_path(sys.argv[1])["build"]()))',
            str(generator),
        ],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(tmp_path)},
        check=True,
        text=True,
        capture_output=True,
    )
    modules = json.loads(completed.stdout)["modules"]
    assert [module["import"] for module in modules] == ["openquant.available"]
    assert modules[0]["functions"][0]["name"] == "select"
    assert modules[0]["functions"][0]["doc"] == "Return the supplied value."

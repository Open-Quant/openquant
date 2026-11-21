"""The public-doc coverage gate should ignore private Python helpers only."""

import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / "check_coverage.py"
SPEC = importlib.util.spec_from_file_location("check_coverage", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
coverage = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(coverage)


class PublicSurfacesTest(unittest.TestCase):
    def test_excludes_private_python_helpers_but_keeps_public_modules(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            for name in ("__init__.py", "_estimator.py", "risk.py"):
                (root / name).touch()
            with patch.object(coverage, "PY_SRC", root):
                self.assertEqual(coverage.public_surfaces()["python"], ["risk"])

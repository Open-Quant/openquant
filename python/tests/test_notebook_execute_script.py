from __future__ import annotations

import json
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_exec_module():
    import importlib.util

    script_path = REPO_ROOT / "notebooks" / "python" / "scripts" / "execute_notebook_cells.py"
    spec = importlib.util.spec_from_file_location("openquant_nb_exec", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_execute_notebook_writes_to_explicit_output_path():
    mod = _load_exec_module()
    nb = {
        "cells": [
            {"cell_type": "markdown", "metadata": {}, "source": ["# t\n"]},
            {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": ["print('ok')\n"]},
        ],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        src = td_path / "in.ipynb"
        out = td_path / "out" / "executed.ipynb"
        src.write_text(json.dumps(nb), encoding="utf-8")

        mod.execute_notebook(src, out)

        assert src.exists()
        assert out.exists()
        executed = json.loads(out.read_text(encoding="utf-8"))
        assert executed["cells"][1]["execution_count"] == 1
        assert executed["cells"][1]["outputs"][0]["output_type"] == "stream"

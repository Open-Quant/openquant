"""The notebook runner behind `just notebooks-run`: errors fail, outputs are normalised."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = REPO_ROOT / "notebooks" / "python" / "scripts"


def _load(name: str) -> ModuleType:
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location(f"openquant_nb_{name}", SCRIPTS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses look their module up here
    spec.loader.exec_module(module)
    return module


def _notebook(*sources: str) -> dict[str, object]:
    return {
        "cells": [
            {
                "cell_type": "code",
                "metadata": {},
                "execution_count": None,
                "outputs": [],
                "source": [s],
            }
            for s in sources
        ],
        "metadata": {"language_info": {"name": "python", "version": "3.13.9"}},
        "nbformat": 4,
        "nbformat_minor": 4,
    }


def test_cell_error_raises_and_keeps_the_traceback(tmp_path: Path) -> None:
    mod = _load("execute_notebook_cells")
    src = tmp_path / "bad.ipynb"
    src.write_text(
        json.dumps(_notebook("x = 1", "raise ValueError('boom')", "y = 2")), encoding="utf-8"
    )

    with pytest.raises(mod.NotebookExecutionError, match="boom"):
        mod.execute_notebook(src, tmp_path / "out.ipynb")

    partial = json.loads((tmp_path / "out.ipynb").read_text(encoding="utf-8"))
    assert partial["cells"][1]["outputs"][0]["output_type"] == "error"
    assert partial["cells"][2]["outputs"] == []


def test_rich_outputs_are_kept_and_metadata_is_normalised(tmp_path: Path) -> None:
    mod = _load("execute_notebook_cells")
    src = tmp_path / "rich.ipynb"
    src.write_text(
        json.dumps(_notebook("from IPython.display import HTML\nHTML('<b>hi</b>')")),
        encoding="utf-8",
    )

    mod.execute_notebook(src)

    nb = json.loads(src.read_text(encoding="utf-8"))
    out = nb["cells"][0]["outputs"][0]
    assert out["output_type"] == "execute_result"
    assert "text/html" in out["data"]
    assert nb["metadata"]["language_info"] == {"name": "python"}
    assert nb["cells"][0]["id"] == "cell-00"
    assert "execution" not in nb["cells"][0]["metadata"]


def test_fingerprint_ignores_float_noise_and_png_bytes_but_not_text() -> None:
    runner = _load("run_notebooks")

    def nb(value: str, png: str) -> str:
        cell = {
            "cell_type": "code",
            "source": ["x"],
            "outputs": [
                {"output_type": "stream", "name": "stdout", "text": [f"sharpe {value}\n"]},
                {
                    "output_type": "display_data",
                    "data": {"image/png": png, "text/plain": "<Image>"},
                },
            ],
        }
        return json.dumps({"cells": [cell]})

    base = runner.notebook_fingerprint(nb("0.123456789012", "AAAA"))
    assert runner.notebook_fingerprint(nb("0.123456789013", "BBBB")) == base
    assert runner.notebook_fingerprint(nb("0.2", "AAAA")) != base


def test_every_notebook_is_run_or_excluded_with_a_reason() -> None:
    runner = _load("run_notebooks")
    notebooks = runner.discover()
    assert len(notebooks) >= 8
    for name, reason in runner.EXCLUDED.items():
        assert (runner.NOTEBOOK_DIR / name).exists(), name
        assert reason.strip(), name


def test_figure_fingerprint_ignores_embedded_png_bytes_and_path_ids() -> None:
    runner = _load("run_notebooks")

    def svg(png: str, pid: str, x: str) -> str:
        return (
            f'<path id="{pid}" d="M 0 {x}"/>\n'
            f'<image xlink:href="data:image/png;base64,\n{png}" width="10"/>\n'
        )

    base = runner.figure_fingerprint(svg("iVBORw0KAAAA", "m0123456789", "1.00000001"))
    assert runner.figure_fingerprint(svg("iVBORw0KBBBB", "mabcdefabcd", "1.00000002")) == base
    assert runner.figure_fingerprint(svg("iVBORw0KAAAA", "m0123456789", "2.0")) != base

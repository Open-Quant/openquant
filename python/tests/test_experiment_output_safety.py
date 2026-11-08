"""Experiment output paths must not escape or replace earlier run artifacts."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def runner():
    spec = importlib.util.spec_from_file_location(
        "experiment_output_safety", ROOT / "experiments/run_pipeline.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _no_dataset(_config):
    pytest.fail("output validation must precede dataset construction")


@pytest.mark.parametrize(
    "name", ["../escape", "/absolute", r"..\escape", r"C:\escape", "", ".", "..", "bad\0name"]
)
@pytest.mark.parametrize("mode", ["single", "grid"])
def test_invalid_run_set_names_fail_before_work(runner, monkeypatch, tmp_path, name, mode):
    cfg = {"meta": {"name": name}}
    monkeypatch.setattr(
        runner, "_read_config", lambda path: cfg if path.name == "base" else {"runs": [{}]}
    )
    monkeypatch.setattr(runner, "_dataset_from_cfg", _no_dataset)
    with pytest.raises(ValueError, match="name"):
        if mode == "single":
            runner.run(Path("base"), tmp_path / "out")
        else:
            runner.run_grid(Path("base"), Path("grid"), tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_grid_rejects_unsafe_entry_name_before_any_work(runner, monkeypatch, tmp_path):
    cfg = {"meta": {"name": "safe"}}
    grid = {"runs": [{"name": "good"}, {"name": "../escape"}]}
    monkeypatch.setattr(runner, "_read_config", lambda path: cfg if path.name == "base" else grid)
    monkeypatch.setattr(runner, "_dataset_from_cfg", _no_dataset)
    with pytest.raises(ValueError, match="name"):
        runner.run_grid(Path("base"), Path("grid"), tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_grid_rejects_duplicate_destination_before_any_work(runner, monkeypatch, tmp_path):
    cfg = {"meta": {"name": "safe"}}
    grid = {"runs": [{"name": "same", "step_size": 0.1}, {"name": "same", "step_size": 0.1}]}
    monkeypatch.setattr(runner, "_read_config", lambda path: cfg if path.name == "base" else grid)
    monkeypatch.setattr(runner, "_dataset_from_cfg", _no_dataset)
    with pytest.raises(ValueError, match="duplicate"):
        runner.run_grid(Path("base"), Path("grid"), tmp_path / "out")
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("mode", ["single", "grid"])
@pytest.mark.parametrize("kind", ["directory", "file", "dangling_symlink"])
def test_existing_destination_is_preserved_before_work(runner, monkeypatch, tmp_path, mode, kind):
    cfg = {"meta": {"name": "safe"}}
    grid = {"runs": [{}]}
    config = cfg if mode == "single" else {"base_config": cfg, "grid_config": grid}
    digest = runner.openquant.research.research_run_manifest(config)["config_digest"]
    destination = tmp_path / (f"safe-{digest}" if mode == "single" else f"safe-grid-{digest}")
    if kind == "directory":
        destination.mkdir()
        (destination / "metrics.parquet").write_bytes(b"keep existing user data")
    elif kind == "file":
        destination.write_bytes(b"keep existing user data")
    else:
        destination.symlink_to(tmp_path / "absent", target_is_directory=True)
    monkeypatch.setattr(runner, "_read_config", lambda path: cfg if path.name == "base" else grid)
    monkeypatch.setattr(runner, "_dataset_from_cfg", _no_dataset)
    with pytest.raises(FileExistsError):
        if mode == "single":
            runner.run(Path("base"), tmp_path)
        else:
            runner.run_grid(Path("base"), Path("grid"), tmp_path)
    if kind == "directory":
        assert (destination / "metrics.parquet").read_bytes() == b"keep existing user data"
    elif kind == "file":
        assert destination.read_bytes() == b"keep existing user data"
    else:
        assert destination.is_symlink()
        assert not (tmp_path / "absent").exists()


@pytest.mark.parametrize("mode", ["single", "grid"])
def test_output_created_after_preflight_is_not_overwritten(runner, monkeypatch, tmp_path, mode):
    cfg = {"meta": {"name": "safe"}}
    grid = {"runs": [{}]}
    config = cfg if mode == "single" else {"base_config": cfg, "grid_config": grid}
    digest = runner.openquant.research.research_run_manifest(config)["config_digest"]
    destination = tmp_path / (f"safe-{digest}" if mode == "single" else f"safe-grid-{digest}")
    monkeypatch.setattr(runner, "_read_config", lambda path: cfg if path.name == "base" else grid)
    monkeypatch.setattr(runner, "_dataset_from_cfg", lambda config: (None, {}))

    def competing_run(*args, **kwargs):
        destination.mkdir()
        (destination / "metrics.parquet").write_bytes(b"concurrent artifact")
        return {}

    function = "run_flywheel_iteration" if mode == "single" else "run_flywheel_grid"
    monkeypatch.setattr(runner.openquant.research, function, competing_run)
    with pytest.raises(FileExistsError):
        if mode == "single":
            runner.run(Path("base"), tmp_path)
        else:
            runner.run_grid(Path("base"), Path("grid"), tmp_path)
    assert (destination / "metrics.parquet").read_bytes() == b"concurrent artifact"


@pytest.mark.parametrize(
    "entries",
    [
        "[[runs]]\n[[runs]]\n",
        '[[runs]]\nname="same"\nstep_size=0.1\n[[runs]]\nname="same"\nstep_size=0.2\n',
    ],
)
def test_distinct_grid_destinations_and_symlink_output_root_remain_supported(
    runner, tmp_path, entries
):
    grid = tmp_path / "grid.toml"
    grid.write_text(entries, encoding="utf-8")
    actual_root = tmp_path / "actual"
    actual_root.mkdir()
    output_root = tmp_path / "chosen-link"
    output_root.symlink_to(actual_root, target_is_directory=True)
    result = runner.run_grid(
        ROOT / "experiments/configs/futures_oil_baseline.toml", grid, output_root
    )
    directories = [path for path in result.iterdir() if path.is_dir()]
    assert len(directories) == 2
    assert all((path / "metrics.parquet").is_file() for path in directories)
    assert result.resolve().is_relative_to(actual_root.resolve())

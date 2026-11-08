"""Configured runner contracts: exact science, strict config, exclusive publication."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "experiments/run_offline_lab.py"
TINY_CONFIG = {"repetitions": 2, "days": 4, "seed": 31, "families": ["on_time"]}


def cli(*arguments, module=False, cwd=ROOT):
    invocation = ["-m", "experiments.run_offline_lab"] if module else [str(SCRIPT)]
    return subprocess.run(
        [sys.executable, *invocation, *map(str, arguments)],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )


@pytest.mark.parametrize("module", [False, True])
def test_cli_lists_allowlisted_labs_and_runtime_requirements(module):
    result = cli("list", module=module)
    assert result.returncode == 0, result.stderr
    catalog = json.loads(result.stdout)
    assert catalog["schema_version"] == 1
    entries = {entry["name"]: entry for entry in catalog["labs"]}
    assert entries["point_in_time"] == {
        "name": "point_in_time",
        "source_file": "point_in_time_lab.py",
        "dependency_source_files": [],
        "required_runtime_packages": ["numpy"],
        "optional_runtime_packages": [],
    }
    assert entries["selection_interval"]["required_runtime_packages"] == ["numpy", "scipy"]
    assert entries["covariance_estimation"]["optional_runtime_packages"] == ["scipy"]


def test_cli_defaults_is_only_saveable_config(tmp_path):
    result = cli("defaults", "--lab", "point_in_time", cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {
        "repetitions": 200,
        "days": 64,
        "seed": 97,
        "families": ["on_time", "delayed_release", "late_revision", "signal_then_revision"],
    }
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("module", [False, True])
def test_cli_tiny_run_preserves_config_and_exactly_replays(tmp_path, module):
    config, output = tmp_path / "config.json", tmp_path / "nested" / "report.json"
    raw = (json.dumps(TINY_CONFIG, indent=4) + "\n").encode()
    config.write_bytes(raw)
    result = cli(
        "run", "--lab", "point_in_time", "--config", config, "--out", output, module=module
    )
    assert result.returncode == 0, result.stderr
    receipt = json.loads(result.stdout)
    assert receipt["operation"] == "configured_offline_lab_run"
    assert receipt["lab"] == "point_in_time"
    assert receipt["identity_matched"] is True
    assert receipt["scientific_replay_performed"] is False
    assert receipt["config_sha256"] == hashlib.sha256(raw).hexdigest()
    assert receipt["report_sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    assert receipt["output"] == str(output.resolve())
    assert config.read_bytes() == raw
    report = json.loads(output.read_bytes())
    assert report["config"] == TINY_CONFIG
    assert report["provenance"]["source_sha256"] == receipt["source_sha256"]
    replay = subprocess.run(
        [
            sys.executable,
            str(ROOT / "experiments/replay_offline_report.py"),
            str(output),
            "--lab",
            "point_in_time",
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert replay.returncode == 0, replay.stderr
    verdict = json.loads(replay.stdout)
    assert verdict["matched"] is True
    assert verdict["saved_scientific_sha256"] == verdict["replayed_scientific_sha256"]
    assert sorted(p.name for p in output.parent.iterdir()) == ["report.json"]


@pytest.fixture
def runner():
    import importlib.util

    spec = importlib.util.spec_from_file_location("run_offline_lab", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def configured_paths(tmp_path):
    config, output = tmp_path / "config.json", tmp_path / "report.json"
    config.write_text(json.dumps(TINY_CONFIG), encoding="utf-8")
    return config, output


def install_run_hook(runner, monkeypatch, hook):
    """Exercise a real tiny study, changing only the external failure under test."""
    import functools

    module = runner.replay._load_lab("point_in_time")
    original = module.run_lab

    @functools.wraps(original)
    def intercepted(**kwargs):
        return hook(original, kwargs)

    monkeypatch.setattr(module, "run_lab", intercepted)
    monkeypatch.setattr(runner.replay, "_load_lab", lambda name: module)


def test_list_never_loads_a_lab(runner, monkeypatch):
    def forbidden(name):
        pytest.fail("listing imported a study")

    monkeypatch.setattr(runner.replay, "_load_lab", forbidden)
    result = runner.list_labs()
    assert {item["name"] for item in result["labs"]} == set(runner.replay.LABS)


def test_defaults_inspects_signature_without_running_study(runner, monkeypatch):
    def forbidden(original, kwargs):
        pytest.fail("defaults executed a study")

    install_run_hook(runner, monkeypatch, forbidden)
    assert runner.lab_defaults("point_in_time") == {
        "repetitions": 200,
        "days": 64,
        "seed": 97,
        "families": ["on_time", "delayed_release", "late_revision", "signal_then_revision"],
    }


def test_run_forwards_exact_decoded_kwargs_once(runner, monkeypatch, configured_paths):
    received = []

    def record(original, kwargs):
        received.append(kwargs)
        return original(**kwargs)

    install_run_hook(runner, monkeypatch, record)
    config, output = configured_paths
    runner.run_configured_lab("point_in_time", config, output)
    assert received == [TINY_CONFIG]
    assert json.loads(output.read_bytes())["config"] == TINY_CONFIG


@pytest.mark.parametrize(
    "raw",
    [
        b'{"days":4,"days":8}',
        b'{"nested":{"x":1,"x":2}}',
        b'{"days":NaN}',
        b'{"days":Infinity}',
        b'{"days":-Infinity}',
        b'{"nested":[1e999]}',
        b'{"nested":{"x":-1e999}}',
        b"[]",
        b"null",
        b"1",
        b'"string"',
        b"garbage",
        b"{",
        b"{} {}",
        b'{"x":"\xff"}',
        b'{"x":' + b"[" * 65 + b"0" + b"]" * 65 + b"}",
        b" " * (1024 * 1024) + b"{}",
    ],
)
def test_bad_configs_rejected_before_lab_load(runner, monkeypatch, tmp_path, raw):
    config, output = tmp_path / "config.json", tmp_path / "report.json"
    config.write_bytes(raw)

    def forbidden(name):
        pytest.fail("invalid config loaded a lab")

    monkeypatch.setattr(runner.replay, "_load_lab", forbidden)
    with pytest.raises(ValueError):
        runner.run_configured_lab("point_in_time", config, output)
    assert config.read_bytes() == raw
    assert sorted(p.name for p in tmp_path.iterdir()) == ["config.json"]


def test_config_accepts_exact_size_limit_and_preserves_json_types(runner, tmp_path):
    config = tmp_path / "config.json"
    raw = b'{"x":[true,null,1,1.0,"[\\"{\\"]" ]}'
    config.write_bytes(raw + b" " * (1024 * 1024 - len(raw)))
    result = runner.read_config(config)
    assert result == {"x": [True, None, 1, 1.0, '["{"]']}
    assert type(result["x"][2]) is int and type(result["x"][3]) is float


def test_unknown_keyword_fails_before_study(runner, monkeypatch, configured_paths):
    def forbidden(original, kwargs):
        pytest.fail("unknown keyword reached run_lab")

    install_run_hook(runner, monkeypatch, forbidden)
    config, output = configured_paths
    config.write_text(json.dumps({**TINY_CONFIG, "source": "os"}))
    with pytest.raises(TypeError, match="source"):
        runner.run_configured_lab("point_in_time", config, output)
    assert not output.exists()


@pytest.mark.parametrize("name", ["../../evil", "os", "point_in_time_lab.py", "", "https://x"])
def test_only_registered_names_can_select_code(runner, configured_paths, name):
    config, output = configured_paths
    with pytest.raises(ValueError, match="unknown lab"):
        runner.run_configured_lab(name, config, output)
    with pytest.raises(ValueError, match="unknown lab"):
        runner.lab_defaults(name)
    assert not output.exists()


def test_scientific_validation_failure_does_not_publish(runner, configured_paths):
    config, output = configured_paths
    raw = b'{"repetitions": false}'
    config.write_bytes(raw)
    with pytest.raises(ValueError, match="repetitions"):
        runner.run_configured_lab("point_in_time", config, output)
    assert not output.exists() and config.read_bytes() == raw
    assert sorted(p.name for p in config.parent.iterdir()) == ["config.json"]


@pytest.mark.parametrize("change", ["source", "runtime", "nonfinite"])
def test_invalid_report_rejected_before_publication(runner, monkeypatch, configured_paths, change):
    def corrupt(original, kwargs):
        report = original(**kwargs)
        if change == "source":
            report["provenance"]["source_sha256"] = "wrong"
        elif change == "runtime":
            report["provenance"]["numpy"] = "wrong"
        else:
            report["invalid"] = float("nan")
        return report

    install_run_hook(runner, monkeypatch, corrupt)
    config, output = configured_paths
    original_bytes = config.read_bytes()
    with pytest.raises(ValueError):
        runner.run_configured_lab("point_in_time", config, output)
    assert not output.exists() and config.read_bytes() == original_bytes
    assert sorted(p.name for p in config.parent.iterdir()) == ["config.json"]


@pytest.mark.parametrize("target_kind", ["file", "config", "hardlink", "symlink", "dangling"])
def test_existing_outputs_refused_before_loading(
    runner, monkeypatch, configured_paths, target_kind
):
    import os

    config, output = configured_paths
    if target_kind == "file":
        output.write_bytes(b"existing")
    elif target_kind == "config":
        output = config
    elif target_kind == "hardlink":
        os.link(config, output)
    elif target_kind == "symlink":
        output.symlink_to(config)
    else:
        output.symlink_to(config.parent / "missing")
    original = config.read_bytes()

    def forbidden(name):
        pytest.fail("existing output allowed study import")

    monkeypatch.setattr(runner.replay, "_load_lab", forbidden)
    with pytest.raises(FileExistsError):
        runner.run_configured_lab("point_in_time", config, output)
    assert config.read_bytes() == original
    if target_kind == "file":
        assert output.read_bytes() == b"existing"
    if target_kind in ("symlink", "dangling"):
        assert output.is_symlink()


@pytest.mark.parametrize("symlink", [False, True])
def test_concurrent_creator_is_not_overwritten(runner, monkeypatch, configured_paths, symlink):
    config, output = configured_paths
    original = config.read_bytes()
    real_link = runner.os.link

    def raced_link(temporary, destination):
        if symlink:
            output.symlink_to(config)
        else:
            output.write_bytes(b"other writer")
        return real_link(temporary, destination)

    monkeypatch.setattr(runner.os, "link", raced_link)
    with pytest.raises(FileExistsError):
        runner.run_configured_lab("point_in_time", config, output)
    assert config.read_bytes() == original
    assert output.is_symlink() if symlink else output.read_bytes() == b"other writer"
    assert sorted(p.name for p in config.parent.iterdir()) == ["config.json", "report.json"]


@pytest.mark.parametrize("operation", ["write", "flush", "fsync", "link"])
def test_write_failure_cleans_only_own_temporary(runner, monkeypatch, configured_paths, operation):
    config, output = configured_paths
    original = config.read_bytes()
    unrelated = config.parent / "keep.tmp"
    unrelated.write_bytes(b"unrelated")

    def fail(*args, **kwargs):
        raise OSError("simulated " + operation + " failure")

    if operation in ("fsync", "link"):
        monkeypatch.setattr(runner.os, operation, fail)
    else:
        real_temporary = runner.tempfile.NamedTemporaryFile

        def failing_temporary(*args, **kwargs):
            stream = real_temporary(*args, **kwargs)
            setattr(stream, operation, fail)
            return stream

        monkeypatch.setattr(runner.tempfile, "NamedTemporaryFile", failing_temporary)
    with pytest.raises(OSError, match="simulated"):
        runner.run_configured_lab("point_in_time", config, output)
    assert not output.exists() and config.read_bytes() == original
    assert unrelated.read_bytes() == b"unrelated"
    assert sorted(p.name for p in config.parent.iterdir()) == ["config.json", "keep.tmp"]


def test_cli_run_requires_explicit_config(tmp_path):
    output = tmp_path / "report.json"
    result = cli("run", "--lab", "point_in_time", "--out", output)
    assert result.returncode == 2
    assert "--config" in result.stderr
    assert not output.exists()


def test_missing_dependency_has_actionable_error(runner, monkeypatch):
    def missing(name):
        raise ModuleNotFoundError("No module named 'scipy'", name="scipy")

    monkeypatch.setattr(runner.replay, "_load_lab", missing)
    with pytest.raises(ValueError, match="scipy.*requirements-offline-research"):
        runner.lab_defaults("selection_interval")


@pytest.mark.parametrize("changed", ["lab", "dependency"])
def test_mid_run_source_change_cannot_stamp_new_code_identity(
    runner, monkeypatch, configured_paths, tmp_path, changed
):
    """Use untracked source fixtures, never rewrite a tracked scientific module."""
    fixture_dir = tmp_path / "sources"
    fixture_dir.mkdir()
    source, dependency = fixture_dir / "fixture.py", fixture_dir / "dependency.py"
    source.write_text("before lab\n")
    dependency.write_text("before dependency\n")
    monkeypatch.setattr(runner.replay, "__file__", str(fixture_dir / "replay_offline_report.py"))
    monkeypatch.setitem(runner.replay.LABS, "point_in_time", source.name)
    monkeypatch.setitem(runner.replay.DEPENDENCIES, "point_in_time", (dependency.name,))
    from types import SimpleNamespace

    def run_lab(**kwargs):
        (source if changed == "lab" else dependency).write_text("changed during execution\n")
        return {
            "schema_version": 1,
            "synthetic_only": True,
            "lab": "point_in_time",
            "config": kwargs,
            "provenance": {
                "numpy": runner.replay.np.__version__,
                "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "dependency_sha256": {
                    dependency.name: hashlib.sha256(dependency.read_bytes()).hexdigest()
                },
            },
        }

    monkeypatch.setattr(runner.replay, "_load_lab", lambda name: SimpleNamespace(run_lab=run_lab))
    config, output = configured_paths
    original = config.read_bytes()
    with pytest.raises(ValueError, match="source.*changed"):
        runner.run_configured_lab("point_in_time", config, output)
    assert not output.exists() and config.read_bytes() == original


@pytest.mark.parametrize(
    "name,kwargs,performance_keys,provenance_keys",
    [
        (
            "selection_validation",
            {
                "repetitions": 2,
                "train_size": 8,
                "holdout_size": 9,
                "n_candidates": 2,
                "families": ["iid_null"],
            },
            {"wall_seconds", "candidate_observations", "candidate_observations_per_second"},
            {"git_sha", "source_sha256", "python", "numpy", "platform"},
        ),
        (
            "preprocessing_leakage",
            {
                "repetitions": 2,
                "train_size": 8,
                "holdout_size": 8,
                "n_features": 2,
                "families": ["stationary_null"],
            },
            {"wall_seconds"},
            {"git_sha", "source_sha256", "python", "numpy", "platform"},
        ),
        (
            "covariance_estimation",
            {
                "repetitions": 2,
                "n_assets": 2,
                "train_size": 4,
                "families": ["isotropic"],
                "shrinkages": [0.5],
            },
            {"wall_seconds"},
            {"git_sha", "source_sha256", "python", "numpy", "scipy"},
        ),
        (
            "adaptive_sign_audit",
            {
                "repetitions": 2,
                "sample_size": 4,
                "n_candidates": 2,
                "rounds": 1,
                "families": ["iid_null"],
            },
            {"wall_seconds"},
            {"git_sha", "source_sha256", "python", "numpy", "dependency_sha256"},
        ),
        (
            "cluster_sign",
            {"repetitions": 2, "n_blocks": 2, "block_size": 2, "families": ["independent_blocks"]},
            {"wall_seconds"},
            {"git_sha", "source_sha256", "python", "numpy", "dependency_sha256"},
        ),
    ],
)
def test_legacy_cli_metadata_preserves_original_science(
    runner, tmp_path, name, kwargs, performance_keys, provenance_keys
):
    module = runner.replay._load_lab(name)
    expected = json.loads(json.dumps(module.run_lab(**kwargs), allow_nan=False))
    assert "provenance" not in expected
    config, output = tmp_path / "config.json", tmp_path / "report.json"
    config.write_text(json.dumps(kwargs))
    runner.run_configured_lab(name, config, output)
    report = json.loads(output.read_bytes())
    assert {k: v for k, v in report.items() if k not in {"provenance", "performance"}} == expected
    assert set(report["provenance"]) == provenance_keys
    assert set(report["performance"]) == performance_keys
    if name == "selection_validation":
        assert report["performance"]["candidate_observations"] == 68
    assert runner.replay.replay_report(report, lab=name)["matched"] is True


def test_legacy_adapter_never_repairs_incorrect_provenance(runner, monkeypatch, tmp_path):
    import functools

    module = runner.replay._load_lab("selection_validation")
    original = module.run_lab

    @functools.wraps(original)
    def corrupted(**kwargs):
        result = original(**kwargs)
        result["provenance"] = {"numpy": "wrong"}
        return result

    monkeypatch.setattr(module, "run_lab", corrupted)
    monkeypatch.setattr(runner.replay, "_load_lab", lambda name: module)
    config, output = tmp_path / "config.json", tmp_path / "report.json"
    config.write_text(
        json.dumps(
            {
                "repetitions": 2,
                "train_size": 8,
                "holdout_size": 8,
                "n_candidates": 2,
                "families": ["iid_null"],
            }
        )
    )
    with pytest.raises(ValueError, match="mismatch"):
        runner.run_configured_lab("selection_validation", config, output)
    assert not output.exists()


def test_unexpected_missing_provenance_is_not_invented(runner, monkeypatch, configured_paths):
    def missing(original, kwargs):
        report = original(**kwargs)
        del report["provenance"]
        return report

    install_run_hook(runner, monkeypatch, missing)
    config, output = configured_paths
    with pytest.raises(ValueError, match="provenance"):
        runner.run_configured_lab("point_in_time", config, output)
    assert not output.exists()


def test_numerical_guard_errors_are_concise_cli_failures(
    runner, monkeypatch, configured_paths, capsys
):
    def overflow(original, kwargs):
        raise OverflowError("numerical range exceeded")

    install_run_hook(runner, monkeypatch, overflow)
    config, output = configured_paths
    with pytest.raises(SystemExit) as error:
        runner.main(
            ["run", "--lab", "point_in_time", "--config", str(config), "--out", str(output)]
        )
    assert error.value.code == 2
    stderr = capsys.readouterr().err
    assert "numerical range exceeded" in stderr and "Traceback" not in stderr
    assert not output.exists()


def test_post_publication_cleanup_failure_leaves_complete_report(
    runner, monkeypatch, configured_paths
):
    config, output = configured_paths
    original = config.read_bytes()

    def cannot_unlink(path, *args, **kwargs):
        raise OSError("simulated unlink failure")

    monkeypatch.setattr(runner.Path, "unlink", cannot_unlink)
    with pytest.raises(OSError, match="unlink failure"):
        runner.run_configured_lab("point_in_time", config, output)
    assert config.read_bytes() == original
    report = json.loads(output.read_bytes())
    assert runner.replay.replay_report(report, lab="point_in_time")["matched"] is True
    leftovers = [p for p in config.parent.iterdir() if p not in (config, output)]
    assert len(leftovers) == 1
    assert leftovers[0].read_bytes() == output.read_bytes()


def test_explicit_empty_config_uses_unchanged_lab_defaults(runner, tmp_path):
    config, output = tmp_path / "config.json", tmp_path / "report.json"
    config.write_bytes(b"{}\n")
    receipt = runner.run_configured_lab("capacity_scaling", config, output)
    report = json.loads(output.read_bytes())
    assert runner.replay.replay_report(report, lab="capacity_scaling")["matched"] is True
    assert receipt["config_sha256"] == hashlib.sha256(b"{}\n").hexdigest()
    assert report["config"]["scales"] == [1000.0, 10000.0, 100000.0, 1000000.0, 10000000.0]


def test_nesting_boundary_counts_containers_not_string_brackets(runner, tmp_path):
    config = tmp_path / "config.json"
    accepted = b'{"x":' + b"[" * 63 + b'"[[["' + b"]" * 63 + b"}"
    config.write_bytes(accepted)
    assert isinstance(runner.read_config(config)["x"], list)
    config.write_bytes(b'{"x":' + b"[" * 64 + b"0" + b"]" * 64 + b"}")
    with pytest.raises(ValueError, match="nesting"):
        runner.read_config(config)


def test_fresh_process_listing_imports_no_scientific_module():
    code = """
import json
import sys
from pathlib import Path
from experiments import run_offline_lab as runner
result = runner.list_labs()
scientific_files = {str((Path(runner.replay.__file__).parent / name).resolve())
                    for name in runner.replay.LABS.values()}
loaded = {str(Path(module.__file__).resolve())
          for module in list(sys.modules.values()) if getattr(module, '__file__', None)}
assert not loaded.intersection(scientific_files)
print(json.dumps(result))
"""
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["labs"]

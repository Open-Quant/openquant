"""Replay contracts: detect changed evidence without executing report-supplied code."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

EXPERIMENTS = Path(__file__).resolve().parents[2] / "experiments"


def load_module(name):
    spec = importlib.util.spec_from_file_location(name, EXPERIMENTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def report():
    lab = load_module("validation_lab")
    result = lab.run_lab(
        repetitions=3, train_size=8, holdout_size=9, n_candidates=2, families=["iid_null"]
    )
    result["provenance"] = lab._provenance()
    return result


def test_exact_replay_and_timing_independence(report):
    replay = load_module("replay_offline_report")
    verdict = replay.replay_report(report, lab="selection_validation")
    assert verdict["matched"]
    assert verdict["saved_scientific_sha256"] == verdict["replayed_scientific_sha256"]
    report["performance"] = {"wall_seconds": 999}
    report["provenance"]["platform"] = "another host"
    assert replay.replay_report(report, lab="selection_validation")["matched"]


@pytest.mark.parametrize("field", ["holdout_sharpe", "selected_candidate"])
def test_modified_replication_fails(report, field):
    replay = load_module("replay_offline_report")
    report["scenarios"][0]["replications"][0][field] += 1
    result = replay.replay_report(report, lab="selection_validation")
    assert not result["matched"]
    assert result["saved_scientific_sha256"] != result["replayed_scientific_sha256"]
    assert field in result["first_difference"]


def test_changed_dataset_hash_or_summary_fails(report):
    replay = load_module("replay_offline_report")
    for key in ("dataset_sha256", "summary"):
        changed = copy.deepcopy(report)
        changed["scenarios"][0][key] = "modified"
        assert not replay.replay_report(changed, lab="selection_validation")["matched"]


@pytest.mark.parametrize("field", ["source_sha256", "numpy"])
def test_mismatched_or_absent_provenance_fails_before_simulating(report, field):
    replay = load_module("replay_offline_report")
    report["provenance"][field] = "wrong"
    with pytest.raises(ValueError, match="mismatch"):
        replay.replay_report(report, lab="selection_validation")
    del report["provenance"][field]
    with pytest.raises(ValueError, match="mismatch"):
        replay.replay_report(report, lab="selection_validation")


@pytest.mark.parametrize("lab", ["../../evil", "os", "validation_lab.py", ""])
def test_report_cannot_select_arbitrary_code(report, lab):
    replay = load_module("replay_offline_report")
    with pytest.raises(ValueError, match="unknown lab"):
        replay.replay_report(report, lab=lab)


@pytest.mark.parametrize(
    "text",
    ['{"config": {}, "config": {}}', '{"a": NaN}', '{"a": Infinity}', "[]", '{"a": [1e999]}'],
)
def test_ambiguous_or_nonstandard_json_rejected(tmp_path, text):
    replay = load_module("replay_offline_report")
    path = tmp_path / "report.json"
    path.write_text(text)
    with pytest.raises(ValueError):
        replay.read_report(path)


def test_structural_comparison_rejects_boolean_numeric_equivalence():
    replay = load_module("replay_offline_report")
    assert replay.first_difference({"a": True}, {"a": 1}) == "$.a: value types differ"
    assert replay.first_difference([1], [1, 2]) == "$: list lengths differ"
    assert replay.first_difference({"a": 1}, {}) == "$: object keys differ"
    assert replay.first_difference(float("nan"), float("nan"))


def test_cli_returns_nonzero_on_modified_science_and_preserves_input(report, tmp_path):
    path = tmp_path / "report.json"
    report["scenarios"][0]["replications"][0]["train_sharpe"] += 0.5
    original = json.dumps(report)
    path.write_text(original)
    result = subprocess.run(
        [
            sys.executable,
            str(EXPERIMENTS / "replay_offline_report.py"),
            str(path),
            "--lab",
            "selection_validation",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1
    assert json.loads(result.stdout)["matched"] is False
    assert path.read_text() == original


def test_cli_refuses_to_overwrite_input(report, tmp_path):
    path = tmp_path / "report.json"
    original = json.dumps(report)
    path.write_text(original)
    result = subprocess.run(
        [
            sys.executable,
            str(EXPERIMENTS / "replay_offline_report.py"),
            str(path),
            "--lab",
            "selection_validation",
            "--out",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert path.read_text() == original


def test_unrecognized_config_argument_is_not_silently_ignored(report):
    replay = load_module("replay_offline_report")
    report["config"]["code"] = "print('should never execute')"
    with pytest.raises(TypeError):
        replay.replay_report(report, lab="selection_validation")


def test_boolean_schema_version_rejected(report):
    replay = load_module("replay_offline_report")
    report["schema_version"] = True
    with pytest.raises(ValueError, match="schema"):
        replay.replay_report(report, lab="selection_validation")


def test_cli_refuses_hardlinked_output_before_replay(report, tmp_path):
    path, alias = tmp_path / "report.json", tmp_path / "alias.json"
    original = json.dumps(report)
    path.write_text(original)
    os.link(path, alias)
    result = subprocess.run(
        [
            sys.executable,
            str(EXPERIMENTS / "replay_offline_report.py"),
            str(path),
            "--lab",
            "selection_validation",
            "--out",
            str(alias),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "must not overwrite" in result.stderr
    assert path.read_text() == alias.read_text() == original


def test_cli_writes_valid_verdict_without_temporary_files(report, tmp_path):
    path, output = tmp_path / "report.json", tmp_path / "verdict.json"
    path.write_text(json.dumps(report))
    result = subprocess.run(
        [
            sys.executable,
            str(EXPERIMENTS / "replay_offline_report.py"),
            str(path),
            "--lab",
            "selection_validation",
            "--out",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert json.loads(output.read_text())["matched"]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["report.json", "verdict.json"]


@pytest.mark.parametrize(
    "name,config",
    [
        (
            "adaptive_holdout",
            {
                "repetitions": 2,
                "sample_size": 8,
                "n_candidates": 3,
                "rounds": 2,
                "families": ["iid_null"],
            },
        ),
        (
            "preprocessing_leakage",
            {
                "repetitions": 2,
                "train_size": 8,
                "holdout_size": 8,
                "n_features": 3,
                "families": ["stationary_signal"],
            },
        ),
    ],
)
def test_new_labs_exact_replay_and_dependency_validation(name, config):
    replay = load_module("replay_offline_report")
    lab = replay._load_lab(name)
    result = lab.run_lab(**config)
    if "provenance" not in result:
        result["provenance"] = {
            "numpy": lab.np.__version__,
            "source_sha256": hashlib.sha256(
                (EXPERIMENTS / replay.LABS[name]).read_bytes()
            ).hexdigest(),
        }
    assert replay.replay_report(result, lab=name)["matched"]
    if name == "adaptive_holdout":
        result["provenance"]["dependency_sha256"]["validation_lab.py"] = "tampered"
        with pytest.raises(ValueError, match="dependency"):
            replay.replay_report(result, lab=name)


@pytest.mark.parametrize(
    "name,args",
    [
        (
            "cluster_sign",
            ["--n-blocks", "4", "--block-size", "4", "--families", "independent_blocks"],
        ),
        (
            "bounded_finite_family",
            ["--sample-size", "8", "--candidates", "3", "--draws", "8", "--families", "iid_null"],
        ),
        (
            "adaptive_sign_audit",
            [
                "--sample-size",
                "8",
                "--n-candidates",
                "3",
                "--rounds",
                "2",
                "--families",
                "iid_null",
            ],
        ),
        (
            "adaptive_holdout",
            ["--sample-size", "8", "--candidates", "3", "--rounds", "2", "--families", "iid_null"],
        ),
        (
            "preprocessing_leakage",
            [
                "--train-size",
                "8",
                "--holdout-size",
                "8",
                "--n-features",
                "3",
                "--families",
                "stationary_signal",
            ],
        ),
    ],
)
def test_saved_cli_reports_replay(name, args, tmp_path):
    replay = load_module("replay_offline_report")
    path = tmp_path / "report.json"
    generated = subprocess.run(
        [
            sys.executable,
            str(EXPERIMENTS / replay.LABS[name]),
            "--repetitions",
            "2",
            *args,
            "--out",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert generated.returncode == 0, generated.stderr
    checked = subprocess.run(
        [sys.executable, str(EXPERIMENTS / "replay_offline_report.py"), str(path), "--lab", name],
        capture_output=True,
        text=True,
        check=False,
    )
    assert checked.returncode == 0, checked.stderr + checked.stdout
    assert json.loads(checked.stdout)["matched"]


def test_optional_interval_runtime_version_is_checked_before_replay(monkeypatch):
    pytest.importorskip("scipy", reason="optional selection-interval research dependency")
    replay = load_module("replay_offline_report")
    interval = load_module("selection_interval_lab")
    report = interval.run_lab(repetitions=2, sample_size=8, n_candidates=2, families=["iid_null"])
    assert replay.replay_report(report, lab="selection_interval")["matched"]
    report["provenance"]["scipy"] = "wrong"
    monkeypatch.setattr(replay, "_load_lab", lambda name: pytest.fail("must check version first"))
    with pytest.raises(ValueError, match="scipy version mismatch"):
        replay.replay_report(report, lab="selection_interval")


def test_numpy_only_replay_does_not_import_optional_scipy(report, monkeypatch):
    replay = load_module("replay_offline_report")
    monkeypatch.setattr(
        replay.importlib, "import_module", lambda name: pytest.fail("unexpected optional import")
    )
    assert replay.replay_report(report, lab="selection_validation")["matched"]


def test_missing_optional_package_has_clear_error(monkeypatch):
    pytest.importorskip("scipy", reason="optional selection-interval research dependency")
    replay = load_module("replay_offline_report")
    interval = load_module("selection_interval_lab")
    report = interval.run_lab(repetitions=2, sample_size=8, n_candidates=2, families=["iid_null"])

    def unavailable(name):
        raise ImportError("test missing package")

    monkeypatch.setattr(replay.importlib, "import_module", unavailable)
    with pytest.raises(ValueError, match="scipy is required"):
        replay.replay_report(report, lab="selection_interval")


@pytest.mark.parametrize("lab", ["selection_interval", "selection_budget"])
def test_optional_research_report_fresh_process_replay(lab, tmp_path):
    pytest.importorskip("scipy", reason="optional selection-interval research dependency")
    replay = load_module("replay_offline_report")
    module = load_module(replay.LABS[lab].removesuffix(".py"))
    config = (
        dict(repetitions=2, sample_size=8, n_candidates=2, families=["iid_null"])
        if lab == "selection_interval"
        else dict(row_budgets=[8], candidate_counts=[2], probabilities=[0.5, 0.65])
    )
    report = module.run_lab(**config)
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report))
    checked = subprocess.run(
        [sys.executable, str(EXPERIMENTS / "replay_offline_report.py"), str(path), "--lab", lab],
        capture_output=True,
        text=True,
        check=False,
    )
    assert checked.returncode == 0, checked.stderr + checked.stdout
    verdict = json.loads(checked.stdout)
    assert verdict["matched"]
    assert verdict["runtime_versions"]["scipy"] == report["provenance"]["scipy"]


def test_sequential_report_replays_without_scipy(monkeypatch):
    replay = load_module("replay_offline_report")
    module = load_module("sequential_sign_lab")
    report = module.run_lab(horizon=16, repetitions=4, families=["iid_fair"])
    monkeypatch.setattr(
        replay.importlib, "import_module", lambda name: pytest.fail("unexpected optional import")
    )
    assert replay.replay_report(report, lab="sequential_sign")["matched"]
    report["families"]["iid_fair"]["methods"]["fixed_mixture"]["observed_crossing_rate"] = 0.123
    assert not replay.replay_report(report, lab="sequential_sign")["matched"]


def test_covariance_optional_runtime_presence_and_absence(monkeypatch):
    replay = load_module("replay_offline_report")
    covariance = load_module("covariance_estimation_lab")
    report = covariance.run_lab(repetitions=2, n_assets=2, train_size=8, families=["isotropic"])
    report["provenance"] = covariance._provenance()
    assert replay.replay_report(report, lab="covariance_estimation")["matched"]
    # Model a source runtime with no SciPy: moments remain, quantiles disappear.
    monkeypatch.setattr(covariance, "scipy", None)
    monkeypatch.setattr(covariance, "f", None)
    monkeypatch.setattr(covariance, "chi2", None)
    report = covariance.run_lab(**report["config"])
    report["provenance"] = covariance._provenance()

    def missing(name):
        raise ImportError("simulated absent optional package")

    monkeypatch.setattr(replay.importlib, "import_module", missing)
    monkeypatch.setattr(replay, "_load_lab", lambda lab: covariance)
    assert replay.replay_report(report, lab="covariance_estimation")["matched"]
    report["provenance"].pop("scipy")
    with pytest.raises(ValueError, match="scipy version mismatch"):
        replay.replay_report(report, lab="covariance_estimation")


def test_multiple_sequential_replay_checks_frozen_dependency(monkeypatch):
    replay = load_module("replay_offline_report")
    module = load_module("multiple_sequential_lab")
    report = module.run_lab(horizon=8, repetitions=3, n_candidates=2, families=["iid_null"])
    assert replay.replay_report(report, lab="multiple_sequential")["matched"]
    report["provenance"]["dependency_sha256"]["sequential_sign_lab.py"] = "wrong"
    monkeypatch.setattr(replay, "_load_lab", lambda lab: pytest.fail("must check dependency first"))
    with pytest.raises(ValueError, match="dependency source hash mismatch"):
        replay.replay_report(report, lab="multiple_sequential")


def test_covariance_saved_cli_report_replays(tmp_path):
    path = tmp_path / "report.json"
    generated = subprocess.run(
        [
            sys.executable,
            str(EXPERIMENTS / "covariance_estimation_lab.py"),
            "--repetitions",
            "2",
            "--n-assets",
            "2",
            "--train-size",
            "8",
            "--families",
            "isotropic",
            "--out",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert generated.returncode == 0, generated.stderr
    checked = subprocess.run(
        [
            sys.executable,
            str(EXPERIMENTS / "replay_offline_report.py"),
            str(path),
            "--lab",
            "covariance_estimation",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert checked.returncode == 0, checked.stderr + checked.stdout
    assert json.loads(checked.stdout)["matched"]


def test_identity_only_never_executes_or_claims_scientific_match(report, monkeypatch):
    replay = load_module("replay_offline_report")
    monkeypatch.setattr(
        replay, "_load_lab", lambda name: pytest.fail("identity check executed lab")
    )
    # Scientific content may be altered: this mode must not certify it.
    report["unverified_scientific_field"] = "tampered"
    verdict = replay.check_report_identity(report, lab="selection_validation")
    assert verdict["identity_matched"]
    assert verdict["scientific_replay_performed"] is False
    assert "matched" not in verdict
    report["provenance"]["source_sha256"] = "wrong"
    with pytest.raises(ValueError, match="source hash mismatch"):
        replay.check_report_identity(report, lab="selection_validation")


def test_cli_identity_check_does_not_run_invalid_config(report, tmp_path):
    report["config"]["not_a_lab_argument"] = 1
    path = tmp_path / "identity-only.json"
    path.write_text(json.dumps(report))
    checked = subprocess.run(
        [
            sys.executable,
            str(EXPERIMENTS / "replay_offline_report.py"),
            str(path),
            "--lab",
            "selection_validation",
            "--check-identity",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert checked.returncode == 0, checked.stderr
    verdict = json.loads(checked.stdout)
    assert verdict["scientific_replay_performed"] is False
    assert verdict["config"]["not_a_lab_argument"] == 1


def test_payoff_target_replay_keeps_cost_and_gross_sign_targets_distinct():
    replay = load_module("replay_offline_report")
    module = load_module("payoff_target_lab")
    report = module.run_lab(horizons=[8], repetitions=3)
    assert replay.replay_report(report, lab="payoff_target")["matched"]
    report["groups"][0]["scenarios"][0]["economics"]["expected_net_per_row"] = 1000.0
    assert not replay.replay_report(report, lab="payoff_target")["matched"]


def test_lossless_gzip_report_input(report, tmp_path):
    import gzip

    replay = load_module("replay_offline_report")
    path = tmp_path / "compressed-evidence.bin"
    path.write_bytes(gzip.compress(json.dumps(report).encode(), mtime=0))
    restored = replay.read_report(path)
    assert restored == report
    assert replay.replay_report(restored, lab="selection_validation")["matched"]


@pytest.mark.parametrize("payload", [b'{"a":1,"a":2}', b'{"x":1e999}', b"[]"])
def test_gzip_preserves_strict_json_contract(payload, tmp_path):
    import gzip

    replay = load_module("replay_offline_report")
    path = tmp_path / "report.json.gz"
    path.write_bytes(gzip.compress(payload, mtime=0))
    with pytest.raises(ValueError):
        replay.read_report(path)


def test_truncated_gzip_fails_cleanly(tmp_path):
    import gzip

    replay = load_module("replay_offline_report")
    path = tmp_path / "report.json.gz"
    path.write_bytes(gzip.compress(b'{"x":1}', mtime=0)[:-4])
    with pytest.raises(ValueError, match="gzip"):
        replay.read_report(path)


@pytest.mark.parametrize("tampered", [False, True])
def test_cli_runtime_diagnostics_do_not_change_scientific_verdict(report, tmp_path, tampered):
    if tampered:
        report["scenarios"][0]["replications"][0]["holdout_sharpe"] += 1
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report))
    command = [
        sys.executable,
        str(EXPERIMENTS / "replay_offline_report.py"),
        str(path),
        "--lab",
        "selection_validation",
    ]
    plain = subprocess.run(command, capture_output=True, text=True, check=False)
    diagnostic = subprocess.run(
        command + ["--runtime-context"], capture_output=True, text=True, check=False
    )
    assert plain.returncode == diagnostic.returncode == int(tampered)
    original = json.loads(plain.stdout)
    detailed = json.loads(diagnostic.stdout)
    context = detailed.pop("execution_runtime")
    assert context["scope"] == "current_execution_only"
    assert context["original_report_runtime"] == "unknown_not_reconstructed"
    assert detailed == original
    assert json.loads(path.read_text()) == report


def test_identity_runtime_diagnostics_do_not_execute_configuration(report, tmp_path):
    report["config"]["not_a_lab_argument"] = 1
    path = tmp_path / "identity.json"
    path.write_text(json.dumps(report))
    checked = subprocess.run(
        [
            sys.executable,
            str(EXPERIMENTS / "replay_offline_report.py"),
            str(path),
            "--lab",
            "selection_validation",
            "--check-identity",
            "--runtime-context",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert checked.returncode == 0, checked.stderr
    result = json.loads(checked.stdout)
    assert result["scientific_replay_performed"] is False
    assert result["execution_runtime"]["scope"] == "current_execution_only"


def test_runtime_diagnostics_support_module_invocation(report, tmp_path):
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report))
    checked = subprocess.run(
        [
            sys.executable,
            "-m",
            "experiments.replay_offline_report",
            str(path),
            "--lab",
            "selection_validation",
            "--runtime-context",
        ],
        cwd=EXPERIMENTS.parent,
        capture_output=True,
        text=True,
        check=False,
    )
    assert checked.returncode == 0, checked.stderr
    assert json.loads(checked.stdout)["matched"] is True

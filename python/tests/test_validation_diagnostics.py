"""Independent reference and registry contracts for the research-only adapter."""

from __future__ import annotations

import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
MODULE = ROOT / "experiments" / "compare_validation_diagnostics.py"


def load_adapter():
    assert MODULE.is_file(), "the diagnostic comparison adapter is not implemented"
    spec = importlib.util.spec_from_file_location("compare_validation_diagnostics", MODULE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def reference_psr(values, benchmark=0.0):
    # Independent standard-library formula, not any production helper.
    n = len(values)
    mean = math.fsum(values) / n
    dev = [x - mean for x in values]
    m2 = math.fsum(x * x for x in dev) / n
    m3 = math.fsum(x**3 for x in dev) / n
    m4 = math.fsum(x**4 for x in dev) / n
    sr = mean / math.sqrt(m2 * n / (n - 1))
    skew, kurt = m3 / m2**1.5, m4 / m2**2
    return NormalDist().cdf(
        (sr - benchmark) * math.sqrt(n - 1) / math.sqrt(1 - skew * sr + (kurt - 1) * sr * sr / 4)
    )


def reference_dsr(values, scores):
    n = len(scores)
    mean = math.fsum(scores) / n
    std = math.sqrt(math.fsum((s - mean) ** 2 for s in scores) / n)
    gamma = 0.5772156649015329
    normal = NormalDist()
    sr0 = std * (
        (1 - gamma) * normal.inv_cdf(1 - 1 / n) + gamma * normal.inv_cdf(1 - 1 / (n * math.e))
    )
    return reference_psr(values, sr0)


def fixture():
    # Training scores: 1/sqrt(5/3), 0, negative of first.
    # Holdout has opposite signs for first/last: choice must remain column 0.
    return np.array(
        [
            [-0.5, -2, 0.5],
            [0.5, -1, -0.5],
            [1.5, 1, -1.5],
            [2.5, 2, -2.5],
            [-2.5, -2, 0.5],
            [-1.5, -1, 1.5],
            [-0.5, 1, 2.5],
            [0.5, 2, 3.5],
        ]
    )


def test_adapter_matches_independent_psr_and_dsr_reference():
    adapter = load_adapter()
    matrix = fixture()
    out = adapter.compare_candidates(matrix, train_size=4)
    scores = [1 / math.sqrt(5 / 3), 0, -1 / math.sqrt(5 / 3)]
    assert out["selected_candidate"] == 0
    assert out["training_psr"] == pytest.approx(reference_psr(matrix[:4, 0]), abs=1e-9)
    assert out["training_dsr"] == pytest.approx(reference_dsr(matrix[:4, 0], scores), abs=1e-9)
    assert out["holdout_psr"] == pytest.approx(reference_psr(matrix[4:, 0]), abs=1e-9)
    assert out["contaminated_holdout_psr"] == pytest.approx(reference_psr(matrix[4:, 2]), abs=1e-9)


@pytest.mark.parametrize("skew_sign", [1.0, -1.0])
def test_asymmetric_returns_match_independent_psr_and_dsr(skew_sign):
    # Deliberately nonzero third moments: symmetric fixtures cannot detect a missing
    # or reversed skewness term in the PSR/DSR denominator. The mean shifts preserve
    # dispersion while making the training winner and contaminated winner explicit.
    base = np.array([-2.0, -1.0, 0.0, 0.5, 1.0, 5.0])
    selected = skew_sign * (base - math.fsum(base) / len(base)) + 0.3
    training = np.column_stack([selected, selected - 0.25, selected - 0.5])
    holdout = np.column_stack([-selected + 0.4, -selected + 0.5, -selected + 0.6])
    scores = []
    for column in training.T:
        mean = math.fsum(column) / len(column)
        variance = math.fsum((value - mean) ** 2 for value in column) / (len(column) - 1)
        scores.append(mean / math.sqrt(variance))
    out = load_adapter().compare_candidates(np.vstack([training, holdout]), train_size=6)
    assert out["selected_candidate"] == 0
    assert out["contaminated_candidate"] == 2
    assert out["training_psr"] == pytest.approx(reference_psr(training[:, 0]), abs=1e-9)
    assert out["training_dsr"] == pytest.approx(reference_dsr(training[:, 0], scores), abs=1e-9)
    assert out["holdout_psr"] == pytest.approx(reference_psr(holdout[:, 0]), abs=1e-9)
    assert out["contaminated_holdout_psr"] == pytest.approx(reference_psr(holdout[:, 2]), abs=1e-9)


def test_holdout_changes_cannot_change_training_diagnostics():
    adapter = load_adapter()
    a = fixture()
    b = a.copy()
    b[4:, :] = -a[4:, :]
    first = adapter.compare_candidates(a, train_size=4)
    second = adapter.compare_candidates(b, train_size=4)
    for key in ["selected_candidate", "training_psr", "training_dsr", "trial_sharpe_std"]:
        assert first[key] == second[key]
    assert first["holdout_psr"] != second["holdout_psr"]


def test_gap_is_not_used_by_diagnostics():
    adapter = load_adapter()
    a = fixture()
    with_gap = np.vstack([a[:4], [[9e99, -9e99, 9e99]], a[4:]])
    one = adapter.compare_candidates(a, train_size=4)
    two = adapter.compare_candidates(with_gap, train_size=4, gap=1)
    for key in ["training_psr", "training_dsr", "holdout_psr"]:
        assert one[key] == two[key]


def test_single_candidate_has_unavailable_dsr_not_zero():
    adapter = load_adapter()
    out = adapter.compare_candidates(fixture()[:, :1], train_size=4)
    assert out["training_dsr"] is None
    assert out["exact_unique_training_dsr"] is None
    assert out["dsr_status"] == "requires_at_least_two_trials"


def test_exact_duplicates_are_audited_without_claiming_effective_trial_count():
    adapter = load_adapter()
    matrix = fixture()[:, :2]
    out = adapter.compare_candidates(np.repeat(matrix, 2, axis=1), train_size=4)
    distinct = adapter.compare_candidates(matrix, train_size=4)
    assert out["n_candidates"] == 4
    assert out["n_exact_unique_training_returns"] == 2
    assert out["exact_unique_training_dsr"] == pytest.approx(distinct["training_dsr"], abs=1e-9)
    assert out["training_dsr"] < out["exact_unique_training_dsr"]


def test_registry_persists_counts_and_matches_direct_dsr(tmp_path):
    adapter = load_adapter()
    matrix = np.repeat(fixture()[:, :2], 2, axis=1)
    path = tmp_path / "registry.json"
    result = adapter.audit_registry(matrix[:4], path=path, selected_candidate=0)
    assert path.exists()
    assert len(json.loads(path.read_text())["trials"]) == 4
    assert result["recorded_trials"] == 4
    assert result["reopened_trials"] == 4
    assert result["trials_after_rerecord"] == 4
    assert result["registry_dsr"] == pytest.approx(result["direct_dsr"], abs=1e-12)
    assert result["n_exact_unique_returns"] == 2
    assert result["registry_sharpe_max_abs_error"] < 1e-12
    with pytest.raises(ValueError):
        adapter.audit_registry(matrix[:4], path=path, selected_candidate=0)


@pytest.mark.parametrize("scale", [1e-200, 1e200])
def test_adapter_handles_scale_without_moment_overflow(scale):
    adapter = load_adapter()
    a = adapter.compare_candidates(fixture(), train_size=4)
    b = adapter.compare_candidates(fixture() * scale, train_size=4)
    for metric in ["training_psr", "training_dsr", "holdout_psr"]:
        assert a[metric] == pytest.approx(b[metric], abs=1e-9)


def test_probability_summary_keeps_unavailable_distinct_and_zero_rate_uncertainty():
    adapter = load_adapter()
    result = adapter.summarize_probabilities([0.1, 0.2, None], threshold=0.95)
    assert result["n_available"] == 2 and result["n_unavailable"] == 1
    assert result["score_mean"] == pytest.approx(0.15)
    assert result["score_monte_carlo_standard_error"] == pytest.approx(0.05)
    assert result["threshold_exceedance_fraction"] == 0
    assert result["wilson_interval_95"][0] == pytest.approx(0)
    assert 0 < result["wilson_interval_95"][1] < 1
    missing = adapter.summarize_probabilities([None, None], threshold=0.95)
    assert missing["threshold_exceedance_fraction"] is None
    assert missing["wilson_interval_95"] is None


@pytest.mark.parametrize(
    "values,threshold",
    [
        ([float("nan")], 0.95),
        ([1.1], 0.95),
        ([-0.1], 0.95),
        ([0.5], 0),
        ([0.5], 1),
        ([0.5], float("inf")),
    ],
)
def test_invalid_probability_inputs_fail_closed(values, threshold):
    adapter = load_adapter()
    with pytest.raises(ValueError):
        adapter.summarize_probabilities(values, threshold=threshold)


def test_comparison_reuses_same_control_datasets_and_records_registry_audits(tmp_path):
    adapter = load_adapter()
    options = dict(
        repetitions=3,
        train_size=12,
        holdout_size=12,
        n_candidates=4,
        seed=9,
        families=["iid_null", "duplicated_null"],
    )
    original = adapter.lab.run_lab(**options)
    result = adapter.run_comparison(**options, registry_repetitions=1)
    for original_scenario, scenario in zip(original["scenarios"], result["scenarios"]):
        assert original_scenario["dataset_sha256"] == scenario["dataset_sha256"]
        assert len(scenario["registry_audits"]) == 1
        assert scenario["registry_audits"][0]["recorded_trials"] == 4
        assert scenario["diagnostic_summary"]["holdout_psr"]["n_available"] == 3
    json.dumps(result, allow_nan=False)


def test_cli_writes_comparison_with_native_source_provenance(tmp_path):
    load_adapter()
    output = tmp_path / "comparison.json"
    result = subprocess.run(
        [
            sys.executable,
            str(MODULE),
            "--repetitions",
            "2",
            "--train-size",
            "12",
            "--holdout-size",
            "12",
            "--candidates",
            "3",
            "--families",
            "iid_null",
            "--out",
            str(output),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    data = json.loads(output.read_text())
    assert data["diagnostics"]["threshold"] == 0.95
    assert len(data["diagnostics"]["provenance"]["evaluation_sha256"]) == 64
    assert len(data["diagnostics"]["provenance"]["native_extension_sha256"]) == 64
    assert data["performance"]["wall_seconds"] > 0


def test_registry_audit_rejects_complex_returns_without_writing(tmp_path):
    adapter = load_adapter()
    values = fixture()[:4].astype(complex) + 1j
    path = tmp_path / "complex.json"
    with pytest.raises((TypeError, ValueError)):
        adapter.audit_registry(values, path=path, selected_candidate=0)
    assert not path.exists()


def test_wilson_interval_matches_independent_literal():
    adapter = load_adapter()
    result = adapter.summarize_probabilities([1.0] * 5 + [0.0] * 95, threshold=0.95)
    assert result["threshold_exceedance_fraction"] == 0.05
    assert result["wilson_interval_95"] == pytest.approx(
        [0.02154367915436796, 0.11175046923191913], abs=1e-12
    )

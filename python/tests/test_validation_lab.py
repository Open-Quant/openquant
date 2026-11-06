"""Synthetic-lab contract tests; hand-derived expectations, no extension required."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

MODULE = Path(__file__).resolve().parents[2] / "experiments" / "validation_lab.py"


def load_lab():
    assert MODULE.is_file(), "the offline selection-validation lab is not implemented"
    spec = importlib.util.spec_from_file_location("validation_lab", MODULE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_sharpe_uses_sample_deviation_and_no_annualization():
    lab = load_lab()
    np.testing.assert_allclose(lab.score_candidates([[1, -2], [2, 0], [3, 2]]), [2, 0])


def test_selection_never_uses_holdout_and_reports_contaminated_comparator():
    lab = load_lab()
    # Training SRs: 2 and 0. Holdout SRs: -2 and 2.
    returns = np.array([[1, -2], [2, 0], [3, 2], [-3, 1], [-2, 2], [-1, 3]])
    result = lab.evaluate_split(returns, train_size=3)
    assert result["selected_candidate"] == 0
    assert result["train_sharpe"] == pytest.approx(2)
    assert result["holdout_sharpe"] == pytest.approx(-2)
    assert result["contaminated_candidate"] == 1
    assert result["contaminated_holdout_sharpe"] == pytest.approx(2)
    assert result["selection_optimism"] == pytest.approx(4)
    returns[3:, 1] = [-300, -200, -100]
    assert lab.evaluate_split(returns, train_size=3)["selected_candidate"] == 0


def test_gap_is_excluded_from_both_selection_and_evaluation():
    lab = load_lab()
    result = lab.evaluate_split(
        [[1, -2], [2, 0], [3, 2], [1e6, -1e6], [-3, 1], [-2, 2], [-1, 3]], train_size=3, gap=1
    )
    assert result["train_sharpe"] == pytest.approx(2)
    assert result["holdout_sharpe"] == pytest.approx(-2)
    assert result["train_range"] == [0, 3]
    assert result["holdout_range"] == [4, 7]


def test_exact_ties_select_first_without_using_future():
    lab = load_lab()
    assert lab.select_on_training([[1, 2], [2, 4], [3, 6]]) == 0


@pytest.mark.parametrize(
    "values",
    [
        [],
        [1, 2, 3],
        [[1], [1], [1]],
        [[1], [2]],
        [[1], [2], [float("nan")]],
        [[1], [2], [float("inf")]],
        [[1, 2], [2]],
        [[1 + 2j], [2 + 3j], [3 + 4j]],
    ],
)
def test_invalid_or_undefined_scores_fail_closed(values):
    lab = load_lab()
    with pytest.raises((ValueError, TypeError)):
        lab.score_candidates(values)


@pytest.mark.parametrize(
    "train_size,gap", [(2, 0), (4, 0), (3, -1), (3, 1), (True, 0), (3.5, 0), (3, False)]
)
def test_invalid_split_configuration_is_rejected(train_size, gap):
    lab = load_lab()
    with pytest.raises((ValueError, TypeError)):
        lab.evaluate_split(
            [[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]], train_size=train_size, gap=gap
        )


def test_duplicate_family_is_exact_and_reproducible():
    lab = load_lab()
    a = lab.simulate_family("duplicated_null", 50, 7, seed=11)
    b = lab.simulate_family("duplicated_null", 50, 7, seed=11)
    np.testing.assert_array_equal(a, b)
    assert a.shape == (50, 7)
    np.testing.assert_array_equal(a[:, 0], a[:, 1])
    np.testing.assert_array_equal(a[:, 2], a[:, 3])
    assert not np.array_equal(a[:, 0], a[:, 2])


def test_positive_drift_changes_only_known_signal_candidate():
    lab = load_lab()
    null = lab.simulate_family("iid_null", 50, 8, seed=11)
    signal = lab.simulate_family("positive_drift", 50, 8, seed=11)
    np.testing.assert_array_equal(signal[:, 1:], null[:, 1:])
    np.testing.assert_allclose(signal[:, 0] - null[:, 0], 0.001)


def test_null_families_have_intended_dependence_structure():
    lab = load_lab()
    iid = lab.simulate_family("iid_null", 30000, 4, seed=21)
    corr = lab.simulate_family("correlated_null", 30000, 4, seed=21)
    ar = lab.simulate_family("serial_null", 30000, 4, seed=21)
    heavy = lab.simulate_family("heavy_tailed_null", 30000, 4, seed=21)
    for values in (iid, corr, ar):
        assert abs(float(values.mean())) < 0.0005
        assert 0.0095 < float(values.std(ddof=1)) < 0.0105
    assert abs(float(np.corrcoef(iid.T)[0, 1])) < 0.04
    assert 0.72 < float(np.corrcoef(corr.T)[0, 1]) < 0.78
    assert 0.57 < float(np.corrcoef(ar[:-1, 0], ar[1:, 0])[0, 1]) < 0.63
    # df=3 has no finite fourth moment; extreme draws should be more common than Gaussian.
    assert np.mean(np.abs(heavy) > 0.05) > 0.001


def test_summary_standard_error_is_over_independent_replications():
    lab = load_lab()
    summary = lab.summarize_values([1, 2, 3])
    assert summary["mean"] == pytest.approx(2)
    assert summary["monte_carlo_standard_error"] == pytest.approx(1 / math.sqrt(3))


def test_report_is_reproducible_finite_and_family_order_independent():
    lab = load_lab()
    kwargs = dict(repetitions=4, train_size=20, holdout_size=15, n_candidates=5, seed=12)
    first = lab.run_lab(families=["iid_null", "duplicated_null"], **kwargs)
    second = lab.run_lab(families=["duplicated_null", "iid_null"], **kwargs)
    one = {x["family"]: x for x in first["scenarios"]}
    two = {x["family"]: x for x in second["scenarios"]}
    assert one == two
    json.dumps(first, allow_nan=False)
    assert first["sharpe_units"] == "per_observation"
    assert first["synthetic_only"] is True
    assert len(one["iid_null"]["dataset_sha256"]) == 64
    assert all(
        r["contaminated_holdout_sharpe"] >= r["holdout_sharpe"]
        for r in one["iid_null"]["replications"]
    )


def test_cli_emits_valid_report_without_openquant_extension(tmp_path):
    load_lab()
    output = tmp_path / "nested" / "report.json"
    proc = subprocess.run(
        [
            sys.executable,
            str(MODULE),
            "--repetitions",
            "3",
            "--train-size",
            "15",
            "--holdout-size",
            "12",
            "--candidates",
            "4",
            "--seed",
            "123",
            "--out",
            str(output),
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    data = json.loads(output.read_text())
    assert len(data["scenarios"]) == 6
    assert data["provenance"]["source_sha256"] == hashlib.sha256(MODULE.read_bytes()).hexdigest()
    assert data["performance"]["wall_seconds"] > 0
    assert data["performance"]["candidate_observations"] == 6 * 3 * 27 * 4


@pytest.mark.parametrize("scale", [1e-300, 1e300])
def test_sharpe_is_stable_under_extreme_positive_scaling(scale):
    lab = load_lab()
    np.testing.assert_allclose(
        lab.score_candidates(np.array([[1.0, -2.0], [2.0, 0.0], [3.0, 2.0]]) * scale),
        [2.0, 0.0],
        atol=1e-14,
    )


def test_evaluation_does_not_mutate_inputs():
    lab = load_lab()
    returns = np.array([[1, -2], [2, 0], [3, 2], [-3, 1], [-2, 2], [-1, 3]], dtype=float)
    original = returns.copy()
    lab.evaluate_split(returns, train_size=3)
    np.testing.assert_array_equal(returns, original)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"repetitions": 1},
        {"train_size": 2},
        {"holdout_size": 2},
        {"train_size": 3},
        {"holdout_size": 3},
        {"n_candidates": 0},
        {"gap": -1},
        {"seed": -1},
        {"seed": True},
        {"families": []},
        {"families": ["unknown"]},
        {"families": ["iid_null", "iid_null"]},
    ],
)
def test_invalid_lab_configuration_rejected(kwargs):
    lab = load_lab()
    with pytest.raises((ValueError, TypeError)):
        lab.run_lab(**kwargs)


@pytest.mark.parametrize("values", [[], [1], [1, float("inf")], [[1, 2], [3, 4]]])
def test_invalid_summary_inputs_rejected(values):
    lab = load_lab()
    with pytest.raises(ValueError):
        lab.summarize_values(values)


def test_dataset_hash_changes_when_seed_changes():
    lab = load_lab()
    kwargs = dict(
        repetitions=2, train_size=10, holdout_size=10, n_candidates=2, families=["iid_null"]
    )
    first = lab.run_lab(seed=1, **kwargs)["scenarios"][0]["dataset_sha256"]
    second = lab.run_lab(seed=2, **kwargs)["scenarios"][0]["dataset_sha256"]
    assert first != second


def test_cli_rejects_invalid_config_without_writing_report(tmp_path):
    load_lab()
    output = tmp_path / "invalid.json"
    proc = subprocess.run(
        [sys.executable, str(MODULE), "--repetitions", "1", "--out", str(output)],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 2
    assert not output.exists()

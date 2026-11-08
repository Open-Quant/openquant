"""Independent PCA/ridge algebra, information-boundary and report contracts."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

MODULE = Path(__file__).resolve().parents[2] / "experiments" / "preprocessing_leakage_lab.py"
spec = importlib.util.spec_from_file_location("preprocessing_leakage_lab", MODULE)
assert spec is not None and spec.loader is not None
lab = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lab)


def test_one_dimensional_ridge_matches_hand_calculation():
    # Centered x=(-1,0,1), y=(-2,0,2): slope=4/(2+2)=1, intercept=6.
    model = lab.fit_pca_ridge([[1], [2], [3]], [6, 8, 10], ridge=2)
    np.testing.assert_allclose(model["coefficient"], [1])
    assert model["intercept"] == pytest.approx(6)
    np.testing.assert_allclose(lab.predict(model, [[0], [4]]), [6, 10])


def test_global_pca_can_remove_signal_without_any_future_labels():
    tx = np.array([[-2, 0], [2, 0], [0, -1], [0, 1]], dtype=float)
    y = tx[:, 0]
    # The future features make feature 1 the principal direction.
    future = np.array([[0, -10], [0, 10], [0, -20], [0, 20]])
    honest = lab.fit_pca_ridge(tx, y, ridge=1)
    global_fit = lab.fit_pca_ridge(tx, y, ridge=1, preprocessing_x=np.vstack([tx, future]))
    np.testing.assert_allclose(honest["coefficient"], [8 / 9, 0])
    np.testing.assert_allclose(global_fit["coefficient"], [0, 0], atol=1e-14)
    assert honest["pca_fit_rows"] == 4
    assert global_fit["pca_fit_rows"] == 8


def test_future_feature_perturbation_does_not_change_training_only_fit():
    x, y, _ = lab.generate("stationary_signal", 20, 20, 4, 97)
    before = lab.fit_pca_ridge(x[:20], y[:20])
    global_before = lab.fit_pca_ridge(x[:20], y[:20], preprocessing_x=x)
    x[20:, 1] *= 100
    after = lab.fit_pca_ridge(x[:20], y[:20])
    global_after = lab.fit_pca_ridge(x[:20], y[:20], preprocessing_x=x)
    np.testing.assert_array_equal(lab.predict(before, x[:20]), lab.predict(after, x[:20]))
    assert (
        np.max(np.abs(lab.predict(global_before, x[:20]) - lab.predict(global_after, x[:20]))) > 0.1
    )


def test_full_component_pca_is_ordinary_isotropic_ridge():
    rng = np.random.default_rng(789)
    x, y, future = (
        rng.normal(size=(30, 4)),
        rng.normal(size=30),
        rng.normal(size=(20, 4)) * [1, 10, 2, 3],
    )
    xc, yc = x - x.mean(axis=0), y - y.mean()
    expected = np.linalg.solve(xc.T @ xc + 3 * np.eye(4), xc.T @ yc)
    for population in (None, np.vstack([x, future])):
        model = lab.fit_pca_ridge(x, y, n_components=4, ridge=3, preprocessing_x=population)
        np.testing.assert_allclose(model["coefficient"], expected, rtol=1e-12, atol=1e-12)
        assert model["intercept"] == pytest.approx(y.mean() - x.mean(axis=0) @ expected)


@pytest.mark.parametrize("ridge", [0, -1, float("nan"), float("inf"), True])
def test_invalid_penalty_fails_closed(ridge):
    with pytest.raises((TypeError, ValueError)):
        lab.fit_pca_ridge([[1], [2], [3]], [1, 2, 3], ridge=ridge)


@pytest.mark.parametrize(
    "x,y",
    [
        ([[1], [2]], [1, 2]),
        ([[1], [2], [3]], [1, 2]),
        ([[1], [2], [float("nan")]], [1, 2, 3]),
        ([[1j], [2j], [3j]], [1, 2, 3]),
        ([[1], [2], [3]], [1, float("inf"), 3]),
    ],
)
def test_invalid_arrays_fail_closed(x, y):
    with pytest.raises((TypeError, ValueError)):
        lab.fit_pca_ridge(x, y)


@pytest.mark.parametrize("components", [0, True, 1.5, 3])
def test_invalid_component_count(components):
    with pytest.raises((TypeError, ValueError)):
        lab.fit_pca_ridge([[1, 2], [2, 3], [3, 4]], [1, 2, 3], n_components=components)


def test_extreme_finite_arithmetic_fails_closed():
    with pytest.raises(ValueError, match="representable"):
        lab.fit_pca_ridge([[1e308], [1e308], [-1e308]], [1, 2, 3])


def test_families_order_does_not_change_paths_and_reports_replay_exactly():
    options = dict(repetitions=5, train_size=12, holdout_size=10, n_features=3, seed=7)
    a = lab.run_lab(**options)
    b = lab.run_lab(**options, families=reversed(lab.FAMILIES))
    assert a["scenarios"] == b["scenarios"][::-1]
    assert a == lab.run_lab(**a["config"])
    json.dumps(a, allow_nan=False)


def test_paired_summary_uses_within_replication_differences():
    report = lab.run_lab(repetitions=5, train_size=12, holdout_size=10, n_features=3)
    for scenario in report["scenarios"]:
        delta = np.array(
            [r["one_component"]["future_minus_training_mse"] for r in scenario["replications"]]
        )
        summary = scenario["summary"]["one_component"]["future_minus_training_mse"]
        assert summary["mean"] == pytest.approx(delta.mean())
        assert summary["monte_carlo_standard_error"] == pytest.approx(
            delta.std(ddof=1) / np.sqrt(5)
        )
        assert scenario["summary"]["all_components"]["max_prediction_difference"] < 1e-10


def test_both_directions_are_preserved_in_declared_shift_controls():
    report = lab.run_lab(repetitions=40, seed=19)
    scores = {
        s["family"]: s["summary"]["one_component"]["future_minus_training_mse"]["mean"]
        for s in report["scenarios"]
    }
    assert scores["signal_variance_rises"] < -5
    assert scores["nuisance_variance_rises"] > 1


def test_cli_records_source_and_environment_provenance(tmp_path):
    output = tmp_path / "report.json"
    subprocess.run(
        [sys.executable, str(MODULE), "--repetitions", "2", "--out", str(output)],
        check=True,
        capture_output=True,
    )
    report = json.loads(output.read_text())
    assert len(report["provenance"]["source_sha256"]) == 64
    assert report["provenance"]["numpy"] == np.__version__
    assert report["performance"]["wall_seconds"] > 0
    replay = subprocess.run(
        [
            sys.executable,
            str(MODULE.parent / "replay_offline_report.py"),
            str(output),
            "--lab",
            "preprocessing_leakage",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(replay.stdout)["matched"] is True


def test_constant_predictors_return_training_mean_without_spurious_fit():
    model = lab.fit_pca_ridge(np.ones((5, 2)), [1, 2, 3, 4, 5])
    np.testing.assert_array_equal(model["coefficient"], [0, 0])
    np.testing.assert_array_equal(lab.predict(model, [[1, 1], [4, 7]]), [3, 3])


@pytest.mark.parametrize("population", [np.ones((2, 2)), np.ones((5, 3))])
def test_incompatible_preprocessing_population_is_rejected(population):
    with pytest.raises(ValueError, match="PCA data"):
        lab.fit_pca_ridge(np.ones((5, 2)), [1, 2, 3, 4, 5], preprocessing_x=population)

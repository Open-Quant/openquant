"""Risk-lab contracts: chronological fitting, known risk, and self-financing drift."""

from __future__ import annotations

import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

MODULE = Path(__file__).resolve().parents[2] / "experiments" / "risk_model_lab.py"


def load_lab():
    assert MODULE.is_file(), "the rolling risk-model utility lab is not implemented"
    spec = importlib.util.spec_from_file_location("risk_model_lab", MODULE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_covariance_uses_sample_denominator_and_fixed_diagonal_shrink():
    lab = load_lab()
    models = lab.estimate_covariances([[0.01, 0.02], [0.02, 0.04], [0.03, 0.06]])
    np.testing.assert_allclose(models["sample"], [[0.0001, 0.0002], [0.0002, 0.0004]])
    np.testing.assert_allclose(models["diagonal_shrink_10"], [[0.0001, 0.00018], [0.00018, 0.0004]])
    assert lab.covariance_diagnostics(models["sample"])["condition_number"] is None
    assert lab.covariance_diagnostics(models["sample"])["numerical_rank"] == 1
    assert lab.covariance_diagnostics(models["diagonal_shrink_10"])["condition_number"] > 1


def test_existing_allocators_match_analytic_diagonal_case_and_hrp_shrink():
    from openquant import hrp

    lab = load_lab()
    cov = np.diag([0.0001, 0.0004])
    for method in ("inverse_variance", "min_vol_sample", "hrp_sample"):
        weights = lab.allocate(method, {"sample": cov, "diagonal_shrink_10": cov})
        np.testing.assert_allclose(weights, [0.8, 0.2], atol=1e-8)
    values = np.random.default_rng(2).normal(0, 0.01, (50, 4))
    expected, _ = hrp.allocate_hrp(
        ["0", "1", "2", "3"], asset_returns=values.tolist(), use_shrinkage=True
    )
    actual = lab.allocate("hrp_shrunk", lab.estimate_covariances(values))
    np.testing.assert_allclose(actual, expected, atol=1e-12)


def test_buy_and_hold_drift_and_initial_cost_are_hand_computed():
    lab = load_lab()
    result = lab.hold_block([[0.1, 0.0], [0.0, 0.1]], [0.5, 0.5], cost_fraction=0.01)
    np.testing.assert_allclose(result["gross_returns"], [0.05, 1 / 21])
    np.testing.assert_allclose(result["net_returns"], [0.0395, 1 / 21])
    np.testing.assert_allclose(result["ending_weights"], [0.5, 0.5])
    assert np.prod(1 + np.array(result["net_returns"])) == pytest.approx(1.089)


def test_complex_weights_are_rejected_without_lossy_coercion():
    with pytest.raises(ValueError):
        load_lab().hold_block([[0.01, 0.02]], np.array([0.5 + 2j, 0.5]))


def test_rolling_fit_never_reads_future_and_turnover_uses_drifted_weights():
    lab = load_lab()
    values = np.array([[0.01, -0.01], [-0.01, 0.01], [0.02, 0.01], [0.1, 0], [0, 0.1], [0, 0]])
    report = lab.evaluate_path(values, train_size=3, rebalance_every=1, cost_bps=10)
    changed = values.copy()
    changed[3:] = [[0, 0.2], [0.1, 0], [0, 0]]
    other = lab.evaluate_path(changed, train_size=3, rebalance_every=1, cost_bps=10)
    for method in lab.METHODS:
        first = report[method]["rebalances"][0]
        assert first["train_range"] == [0, 3]
        assert first["holdout_range"] == [3, 4]
        np.testing.assert_array_equal(first["weights"], other[method]["rebalances"][0]["weights"])
    equal = report["equal_weight"]["rebalances"]
    assert equal[0]["traded_fraction"] == pytest.approx(1)
    assert equal[1]["traded_fraction"] == pytest.approx(1 / 21)
    assert equal[1]["target_weight_change_l1"] == 0
    assert equal[1]["cost_fraction"] == pytest.approx(0.001 / 21)
    assert equal[-1]["holdout_range"] == [5, 6]


def test_known_covariance_is_diagnostic_only_and_uses_drifting_weights():
    lab = load_lab()
    values = np.random.default_rng(7).normal(0, 0.01, (20, 2))
    truth = np.repeat(np.diag([0.0001, 0.0004])[None, :, :], 20, axis=0)
    first = lab.evaluate_path(values, train_size=12, rebalance_every=3, true_covariances=truth)
    other = lab.evaluate_path(values, train_size=12, rebalance_every=3, true_covariances=truth * 3)
    for method in lab.METHODS:
        a, b = first[method], other[method]
        assert a["gross_returns"] == b["gross_returns"]
        assert a["mean_conditional_variance"] * 3 == pytest.approx(b["mean_conditional_variance"])


@pytest.mark.parametrize("boundary", [12, 16, 20])
def test_every_rebalance_is_invariant_to_its_future_suffix(boundary):
    lab = load_lab()
    values = np.random.default_rng(18).normal(0, 0.01, (24, 3))
    changed = values.copy()
    changed[boundary:] *= -3
    first = lab.evaluate_path(values, train_size=12, rebalance_every=4)
    second = lab.evaluate_path(changed, train_size=12, rebalance_every=4)
    block = (boundary - 12) // 4
    for method in lab.METHODS:
        np.testing.assert_array_equal(
            first[method]["rebalances"][block]["weights"],
            second[method]["rebalances"][block]["weights"],
        )


def test_failed_paths_are_unavailable_instead_of_zero_performance():
    lab = load_lab()
    report = lab.evaluate_path(np.zeros((6, 2)), train_size=2)
    for method in lab.METHODS:
        assert report[method]["status"] == "unavailable"
        assert "realized_variance" not in report[method]
        assert report[method]["completed_blocks"] == 0
    summary = lab._summary([1.0, None, 3.0])
    assert summary["mean"] == 2
    assert summary["n_available"] == 2
    assert summary["n_unavailable"] == 1


@pytest.mark.parametrize(
    "truth",
    [
        np.zeros((5, 2, 2)),
        np.full((6, 2, 2), np.nan),
        np.tile([[1, 2], [0, 1]], (6, 1, 1)),
        np.tile([[-1, 0], [0, 1]], (6, 1, 1)),
    ],
)
def test_invalid_truth_cannot_contaminate_risk_measurements(truth):
    values = np.random.default_rng(7).normal(0, 0.01, (6, 2))
    with pytest.raises(ValueError):
        load_lab().evaluate_path(values, train_size=3, true_covariances=truth)


def test_generators_have_known_covariance_and_regime_change():
    lab = load_lab()
    values, covs = lab.simulate_family("correlated", 30000, 4, seed=7)
    np.testing.assert_allclose(np.cov(values, rowvar=False), covs[0], rtol=0.06, atol=1e-6)
    _, shift = lab.simulate_family("regime_shift", 30, 4, seed=7)
    np.testing.assert_array_equal(shift[0], shift[14])
    assert not np.array_equal(shift[14], shift[15])
    values2, _ = lab.simulate_family("correlated", 30000, 4, seed=7)
    np.testing.assert_array_equal(values, values2)


def test_report_has_paired_replication_precision_and_order_independent_hashes():
    lab = load_lab()
    options = dict(
        repetitions=3, train_size=15, holdout_size=6, n_assets=3, rebalance_every=4, seed=3
    )
    first = lab.run_lab(families=["diagonal", "regime_shift"], **options)
    second = lab.run_lab(families=["regime_shift", "diagonal"], **options)
    assert {s["family"]: s for s in first["scenarios"]} == {
        s["family"]: s for s in second["scenarios"]
    }
    scenario = first["scenarios"][0]
    assert len(scenario["dataset_sha256"]) == 64
    delta = [
        r["methods"]["hrp_sample"]["realized_variance"]
        - r["methods"]["equal_weight"]["realized_variance"]
        for r in scenario["replications"]
    ]
    summary = scenario["paired_vs_equal_weight"]["hrp_sample"]["realized_variance"]
    assert summary["mean"] == pytest.approx(np.mean(delta))
    assert summary["monte_carlo_standard_error"] == pytest.approx(
        np.std(delta, ddof=1) / math.sqrt(3)
    )
    json.dumps(first, allow_nan=False)


@pytest.mark.parametrize(
    "values",
    [
        [],
        [1, 2],
        [[0, 0], [0, 0]],
        [[0, 1], [1, np.nan]],
        [[0, 1], [1, np.inf]],
        [[0, 1], [-1, 2]],
        [[1 + 1j, 0], [0, 1]],
    ],
)
def test_invalid_returns_fail_closed(values):
    with pytest.raises((ValueError, TypeError)):
        load_lab().evaluate_path(values, train_size=2)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"train_size": True},
        {"train_size": 1},
        {"rebalance_every": 0},
        {"cost_bps": -1},
        {"cost_bps": np.inf},
        {"cost_bps": 5000},
    ],
)
def test_invalid_evaluation_configuration_fails_closed(kwargs):
    values = np.random.default_rng(7).normal(0, 0.01, (20, 3))
    with pytest.raises((ValueError, TypeError)):
        load_lab().evaluate_path(values, **{"train_size": 12, **kwargs})


def test_cli_writes_strict_json_with_native_provenance(tmp_path):
    load_lab()
    out = tmp_path / "new" / "risk.json"
    process = subprocess.run(
        [
            sys.executable,
            str(MODULE),
            "--repetitions",
            "2",
            "--train-size",
            "12",
            "--holdout-size",
            "5",
            "--assets",
            "3",
            "--families",
            "diagonal",
            "--out",
            str(out),
        ],
        text=True,
        capture_output=True,
    )
    assert process.returncode == 0, process.stderr
    report = json.loads(out.read_text())
    assert report["synthetic_only"] is True
    assert len(report["provenance"]["native_extension_sha256"]) == 64
    assert report["config"]["cost_bps"] == 10

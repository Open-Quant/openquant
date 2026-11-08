"""Analytic singular-simplex checks and verified optimizer shortcut contracts."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("scipy", reason="optional constraint-solver research dependency")
spec = importlib.util.spec_from_file_location(
    "constraint_solver_lab",
    Path(__file__).resolve().parents[2] / "experiments/constraint_solver_lab.py",
)
assert spec is not None and spec.loader is not None
lab = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lab)


def test_exact_three_vertex_control_and_interior_future_optimum():
    case = lab.ambiguity_case("symmetric_null_assets")
    expected = [(0.5, 0.5, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)]
    actual = [r["weights"] for r in case["zero_risk_vertices"]]
    assert len(actual) == 3
    for w in expected:
        assert any(np.allclose(w, v, atol=1e-12) for v in actual)
    np.testing.assert_allclose(case["minimum_norm_tie_break"]["weights"], [0.25] * 4, atol=1e-12)
    assert case["minimum_norm_tie_break"]["future_variance"] == pytest.approx(1.375)
    np.testing.assert_allclose(
        case["future_minimum_on_zero_set_diagnostic_only"]["weights"],
        [16 / 37, 16 / 37, 4 / 37, 1 / 37],
        atol=1e-12,
    )
    assert case["future_minimum_on_zero_set_diagnostic_only"]["future_variance"] == pytest.approx(
        16 / 37
    )
    assert case["maximum_future_variance_on_zero_set"] == pytest.approx(16.0)
    assert min(r["future_variance"] for r in case["zero_risk_vertices"]) == pytest.approx(0.5)


def test_asymmetric_minimum_norm_is_not_equal_weight_or_finite_ridge():
    case = lab.ambiguity_case("asymmetric_factor")
    np.testing.assert_allclose(
        case["minimum_norm_tie_break"]["weights"], np.array([3, 6, 5, 5]) / 19, atol=1e-12
    )
    largest = case["ridge_changes_primary_objective"][-1]["original_objective_score"]
    assert largest["training_variance"] > 0
    assert not np.allclose(largest["weights"], case["minimum_norm_tie_break"]["weights"])


@pytest.mark.parametrize("case", list(lab.CASES))
def test_all_candidates_feasible_and_convex_future_bounds(case):
    report = lab.ambiguity_case(case)
    F = np.array(report["factors"])
    Q = np.array(report["future_covariance"])
    vertices = np.array([r["weights"] for r in report["zero_risk_vertices"]])
    rng = np.random.default_rng(7)
    for _ in range(50):
        w = rng.dirichlet(np.ones(len(vertices))) @ vertices
        np.testing.assert_allclose(F @ w, 0, atol=1e-10)
        assert (
            report["future_minimum_on_zero_set_diagnostic_only"]["future_variance"] - 1e-9
            <= w @ Q @ w
            <= report["maximum_future_variance_on_zero_set"] + 1e-9
        )
    for record in report["multistart_primary"]:
        assert record["fit"]["available"]
        assert record["score"]["training_variance"] < 1e-10


@pytest.mark.parametrize("scale", [1e-150, 1.0, 1e150])
def test_feasible_inverse_shortcut_exact_and_no_optimizer(monkeypatch, scale):
    cov = scale * np.diag([1.0, 2.0, 5.0])
    inverse = lab.robustness.inverse_gmv(cov)
    monkeypatch.setattr(
        lab.robustness,
        "long_only_gmv",
        lambda a: pytest.fail("must not optimize a verified feasible inverse"),
    )
    result = lab.fast_long_only(cov, inverse)
    assert result["used_fast_path"]
    assert result["weights"] == inverse["weights"]


@pytest.mark.parametrize(
    "candidate",
    [
        {"available": True, "weights": [1.0, 0.0]},
        {"available": True, "weights": [np.nan, 0.0]},
        {"available": False},
        {"available": True, "weights": [-0.1, 1.1]},
    ],
)
def test_bad_candidate_falls_back(candidate):
    result = lab.fast_long_only(np.eye(2), candidate)
    assert not result["used_fast_path"]
    np.testing.assert_allclose(result["weights"], [0.5, 0.5], atol=1e-12)


def test_signed_inverse_and_singular_do_not_use_shortcut():
    for cov in [np.array([[1.0, 1.5], [1.5, 4.0]]), np.array([[1.0, 1.0], [1.0, 1.0]])]:
        result = lab.fast_long_only(cov)
        assert result["available"] and not result["used_fast_path"]


@pytest.mark.parametrize("bad", [[[1 + 1j, 0]], [[np.inf, 0]], [[0, 0]], [[1, 1]], []])
def test_invalid_factor_design(bad):
    with pytest.raises(ValueError):
        lab.equality_system(bad)


def test_scientific_replay_and_counterbalanced_benchmark():
    first = lab.run_lab(benchmark_repetitions=2)
    second = lab.run_lab(**first["config"])
    assert first["cases"] == second["cases"]
    assert first["benchmark_records"] == second["benchmark_records"]
    assert len(first["benchmark_records"]) == 72
    assert first["performance"]["strict_seconds"] > 0
    for row in first["benchmark_records"]:
        if not row["used_fast_path"]:
            assert row["strict_weights"] == row["fast_weights"]


def test_constrained_optimum_is_not_mislabeled_as_unconstrained_inverse():
    cov = np.array([[1.0, 1.5], [1.5, 4.0]])
    result = lab.fast_long_only(cov, {"available": True, "weights": [1.0, 0.0]})
    assert not result["used_fast_path"]


def test_primary_rejects_indefinite_and_handles_solver_exception(monkeypatch):
    with pytest.raises(ValueError):
        lab.primary_from_start(np.array([[1.0, 2.0], [2.0, 1.0]]), [0.5, 0.5])

    def fail(*a, **k):
        raise RuntimeError("test")

    monkeypatch.setattr(lab, "minimize", fail)
    assert not lab.primary_from_start(np.eye(2), [0.5, 0.5])["available"]

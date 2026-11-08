"""Independent analytical and operational checks for covariance robustness."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("scipy", reason="optional covariance robustness research dependency")
spec = importlib.util.spec_from_file_location(
    "covariance_robustness_lab",
    Path(__file__).resolve().parents[2] / "experiments/covariance_robustness_lab.py",
)
assert spec is not None and spec.loader is not None
lab = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lab)


@pytest.mark.parametrize("p", [2, 4, 8])
@pytest.mark.parametrize("regime", lab.REGIMES)
def test_population_oracles_and_kkt(p, regime):
    _, future = lab.covariances(regime, p)
    truth = lab.population_oracle(regime, p)
    if regime == "hedge_blocks":
        np.testing.assert_allclose(
            truth["unrestricted_weights"], np.tile([1.25, -0.25], p // 2) / (p // 2)
        )
        assert truth["unrestricted_variance"] == pytest.approx(0.875 / (p // 2))
        assert truth["long_only_variance"] == pytest.approx(1 / (p // 2))
    else:
        target = (1 / np.diag(future)) / sum(1 / np.diag(future))
        np.testing.assert_allclose(truth["long_only_weights"], target)
    fit = lab.long_only_gmv(future)
    assert fit["available"]
    assert np.array(fit["weights"]) @ future @ np.array(fit["weights"]) == pytest.approx(
        truth["long_only_variance"], abs=1e-10
    )


@pytest.mark.parametrize("scale", [1e-200, 1.0, 1e200])
def test_scale_invariance_and_singular_simplex(scale):
    cov = np.array([[1.0, 1.5], [1.5, 4.0]])
    fit = lab.long_only_gmv(scale * cov)
    assert fit["available"]
    np.testing.assert_allclose(fit["weights"], [1, 0], atol=1e-7)
    zero = lab.long_only_gmv(np.zeros((3, 3)))
    assert zero["available"]
    np.testing.assert_allclose(zero["weights"], [1 / 3] * 3)


@pytest.mark.parametrize("bad", [[np.nan, 0], [np.inf, 0], [1, 2, 3], [1 + 1j, 0], [1e308, 1e308]])
def test_malformed_success_is_unavailable(monkeypatch, bad):
    monkeypatch.setattr(
        lab,
        "minimize",
        lambda *a, **k: SimpleNamespace(
            success=True, status=0, message="test", nit=1, x=np.array(bad)
        ),
    )
    result = lab.long_only_gmv(np.eye(2))
    assert not result["available"]
    assert result["weights"] is None


@pytest.mark.parametrize(
    "success,weights", [(False, [0.5, 0.5]), (True, [1.0, 0.0]), (True, [-0.1, 1.1])]
)
def test_solver_success_is_insufficient(monkeypatch, success, weights):
    monkeypatch.setattr(
        lab,
        "minimize",
        lambda *a, **k: SimpleNamespace(
            success=success, status=0, message="test", nit=1, x=np.array(weights)
        ),
    )
    assert not lab.long_only_gmv(np.eye(2))["available"]


def test_solver_exception_recorded(monkeypatch):
    def fail(*a, **k):
        raise RuntimeError("numerical test")

    monkeypatch.setattr(lab, "minimize", fail)
    assert not lab.long_only_gmv(np.eye(2))["available"]


def test_training_only_shift_and_distinct_shrinkage():
    x = np.random.default_rng(42).normal(size=(20, 4)) * [1, 2, 3, 4]
    first = lab.evaluate(x, np.eye(4))
    second = lab.evaluate(x, np.diag([4, 3, 2, 1]))
    for name in lab.METHODS:
        assert first[name]["weights"] == second[name]["weights"]
    assert not np.allclose(
        first["weight_blend_50"]["weights"], first["diagonal_shrink_50"]["weights"]
    )


def test_singular_sample_does_not_suppress_other_estimators():
    x = np.random.default_rng(7).normal(size=(4, 8))
    fits = lab.fit_models(x)["methods"]
    assert not fits["sample_gmv"]["available"]
    assert not fits["weight_blend_50"]["available"]
    for name in ["equal_weight", "long_only", "diagonal_shrink_50"]:
        assert fits[name]["available"]


@pytest.mark.parametrize("distribution", lab.DISTRIBUTIONS)
def test_covariance_normalization_empirical(distribution):
    cov = np.array([[1.0, 0.4], [0.4, 2.0]])
    x = lab.draw_training(np.random.default_rng(18), distribution, 100000, cov)
    np.testing.assert_allclose(np.cov(x, rowvar=False), cov, rtol=0.04, atol=0.025)


def test_replay_scientific_contract_and_no_claimed_mean_mcse():
    first = lab.run_lab(repetitions=2, n_assets=2, train_size=4)
    second = lab.run_lab(**first["config"])
    assert first["scenarios"] == second["scenarios"]
    assert first["provenance"] == second["provenance"]
    assert first["performance"]["fit_seconds"]["long_only"] > 0
    for scenario in first["scenarios"]:
        for method in scenario["summary"].values():
            assert method["future_variance"]["mean_mcse"] is None


@pytest.mark.parametrize(
    "value",
    [np.array([[1, 2], [2, 1]]), np.array([[1, 2], [0, 1]]), np.array([[1, np.nan], [np.nan, 1]])],
)
def test_invalid_objective(value):
    with pytest.raises(ValueError):
        lab.long_only_gmv(value)


def test_diagonal_shift_reuses_identical_training_draws():
    report = lab.run_lab(repetitions=2, n_assets=2, train_size=4)
    for distribution in lab.DISTRIBUTIONS:
        records = {
            s["regime"]: s["records"]
            for s in report["scenarios"]
            if s["distribution"] == distribution
        }
        for stationary, shifted in zip(
            records["diagonal"], records["diagonal_reversal"], strict=True
        ):
            assert stationary["training_sha256"] == shifted["training_sha256"]
            for method in lab.METHODS:
                assert (
                    stationary["methods"][method]["weights"]
                    == shifted["methods"][method]["weights"]
                )

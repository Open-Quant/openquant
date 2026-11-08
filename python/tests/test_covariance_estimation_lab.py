"""Independent quadratic geometry and exact-law boundary controls; no native API."""

from __future__ import annotations

import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

MODULE = Path(__file__).resolve().parents[2] / "experiments" / "covariance_estimation_lab.py"
spec = importlib.util.spec_from_file_location("covariance_estimation_lab", MODULE)
assert spec is not None and spec.loader is not None
lab = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lab)


def test_hand_diagonal_sample_covariance_and_weights():
    x = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 2.0], [0.0, -2.0]])
    fit = lab.fit_sample_gmv(x)
    np.testing.assert_allclose(fit["sample_covariance"], np.diag([2 / 3, 8 / 3]))
    np.testing.assert_allclose(fit["weights"], [0.8, 0.2])
    r = lab.evaluate(x, np.eye(2))["methods"]
    assert r["0.0"]["population_variance"] == pytest.approx(0.68)
    assert r["0.0"]["forecast_variance"] == pytest.approx(8 / 15)
    assert r["0.5"]["population_variance"] == pytest.approx(0.545)
    assert r["1.0"]["population_variance"] == pytest.approx(0.5)


def test_signed_weights_are_not_clipped_or_renormalized_to_longonly():
    x = math.sqrt(1.5) * np.array([[1, 2], [-1, -2], [0, math.sqrt(5)], [0, -math.sqrt(5)]])
    row = lab.evaluate(x, np.eye(2))["methods"]["0.0"]
    np.testing.assert_allclose(row["weights"], [7 / 6, -1 / 6])
    assert row["gross_leverage"] == pytest.approx(4 / 3)
    assert row["negative_weight_count"] == 1


def test_population_risk_geometry_and_forecast_blend_identity():
    x = np.random.default_rng(7).normal(size=(20, 8))
    result = lab.evaluate(x, np.eye(8))["methods"]
    for lam in lab.SHRINKAGES:
        row = result[str(lam)]
        w = np.asarray(row["weights"])
        equal = np.full(8, 1 / 8)
        assert row["population_variance"] == pytest.approx(1 / 8 + sum((w - equal) ** 2))
        assert row["population_variance"] == pytest.approx(row["squared_weight_norm"])
        assert row["forecast_variance"] == pytest.approx(
            (1 - lam**2) * result["0.0"]["forecast_variance"]
            + lam**2 * result["1.0"]["forecast_variance"]
        )
    assert result["0.0"]["forecast_variance"] <= result["1.0"]["forecast_variance"]
    assert result["0.0"]["population_variance"] >= result["1.0"]["population_variance"]


def test_population_covariance_does_not_change_estimator():
    x = np.random.default_rng(9).normal(size=(20, 8))
    a = lab.evaluate(x, np.eye(8))
    b = lab.evaluate(x, np.diag(np.arange(1, 9)))
    for name in a["methods"]:
        assert a["methods"][name]["weights"] == b["methods"][name]["weights"]
    assert a["methods"]["0.0"]["population_variance"] != b["methods"]["0.0"]["population_variance"]


def test_centering_translation_and_common_scale_invariance():
    x = np.random.default_rng(10).normal(size=(20, 8))
    fit = lab.fit_sample_gmv(x)
    np.testing.assert_allclose(lab.fit_sample_gmv(x + 3)["weights"], fit["weights"], atol=1e-14)
    scaled = lab.fit_sample_gmv(x * 7)
    np.testing.assert_allclose(scaled["weights"], fit["weights"], atol=1e-14)
    np.testing.assert_allclose(scaled["sample_covariance"], 49 * fit["sample_covariance"])


def test_singular_inverse_unavailable_but_equalweight_remains_available():
    x = np.random.default_rng(1).normal(size=(8, 8))
    result = lab.evaluate(x, np.eye(8))
    assert result["sample_fit_failure"]
    assert not result["methods"]["0.0"]["available"]
    assert not result["methods"]["0.75"]["available"]
    assert result["methods"]["1.0"]["available"]
    assert result["methods"]["1.0"]["population_variance"] == pytest.approx(1 / 8)
    constant = lab.evaluate(np.ones((20, 8)), np.eye(8))
    assert not constant["methods"]["0.0"]["available"]
    assert constant["methods"]["1.0"]["available"]


def test_exact_mean_and_moment_existence():
    for t, status, variance in [
        (8, "unavailable", False),
        (9, "infinite", False),
        (10, "finite", False),
        (11, "finite", False),
        (12, "finite", True),
    ]:
        reference = lab.exact_reference(t, 8, np.eye(8), 0)
        assert reference["population_variance_mean_status"] == status
        assert reference["population_variance_has_finite_variance"] is variance
        equal = lab.exact_reference(t, 8, np.eye(8), 1)
        assert equal["expected_population_variance"] == pytest.approx(1 / 8)
        assert equal["population_variance_has_finite_variance"]
    assert lab.exact_reference(20, 8, np.eye(8), 0)[
        "expected_population_variance"
    ] == pytest.approx((1 / 8) * 18 / 11)
    assert lab.exact_reference(20, 8, np.eye(8), 0)["expected_forecast_variance"] == pytest.approx(
        (1 / 8) * 12 / 19
    )


def test_hand_F_and_chisquare_quantiles():
    if lab.scipy is None:
        pytest.skip("optional reference quantiles require scipy")
    # F(2,4) CDF = 1 - (2/(2+x))**2, so risk-ratio median is sqrt(2).
    r = lab.exact_reference(6, 3, np.eye(3), 0)
    assert r["population_risk_ratio_quantiles"]["0.5"] == pytest.approx(math.sqrt(2))
    # chi-square(2) median = 2 log(2); sample covariance df = T-1 =3.
    r = lab.exact_reference(4, 2, np.eye(2), 0)
    assert r["sample_gmv_forecast_ratio_quantiles"]["0.5"] == pytest.approx(2 * math.log(2) / 3)


def test_heterogeneous_oracle_and_weight_shrink_mean():
    sigma = np.diag([1.0, 4.0])
    population = lab.population_reference(sigma)
    np.testing.assert_allclose(population["oracle_weights"], [0.8, 0.2])
    assert population["oracle_variance"] == pytest.approx(0.8)
    assert population["equal_variance"] == pytest.approx(1.25)
    r = lab.exact_reference(10, 2, sigma, 0.5)
    assert r["expected_population_variance"] == pytest.approx(
        0.8 + 0.25 * 0.8 / 7 + 0.25 * (1.25 - 0.8)
    )
    assert r["population_risk_ratio_quantiles"] is None


def test_nonexistent_moments_never_get_finite_mean_mcse():
    r = lab.run_lab(repetitions=4, n_assets=8, train_size=9, families=["isotropic"])
    metrics = r["scenarios"][0]["summary"]["0.0"]["metrics"]
    assert metrics["population_variance"]["empirical_mean"] is None
    assert metrics["population_variance"]["monte_carlo_standard_error"] is None
    assert metrics["population_variance"]["quantiles"] is not None
    assert metrics["forecast_variance"]["monte_carlo_standard_error"] is not None
    r = lab.run_lab(repetitions=4, n_assets=8, train_size=10, families=["isotropic"])
    metrics = r["scenarios"][0]["summary"]["0.0"]["metrics"]
    assert metrics["population_variance"]["empirical_mean"] is not None
    assert metrics["population_variance"]["monte_carlo_standard_error"] is None


def test_report_replay_family_and_shrinkage_order_invariance():
    a = lab.run_lab(repetitions=3, n_assets=3, train_size=8)
    b = lab.run_lab(
        **{
            **a["config"],
            "families": list(reversed(lab.FAMILIES)),
            "shrinkages": list(reversed(lab.SHRINKAGES)),
        }
    )
    assert a["scenarios"] == b["scenarios"][::-1]
    assert a == lab.run_lab(**a["config"])
    json.dumps(a, allow_nan=False)


def test_optional_scipy_absence_keeps_moments_and_point_control(monkeypatch):
    monkeypatch.setattr(lab, "scipy", None)
    monkeypatch.setattr(lab, "chi2", None)
    monkeypatch.setattr(lab, "f", None)
    r = lab.exact_reference(10, 2, np.eye(2), 0)
    assert not r["scipy_quantiles_available"]
    assert r["population_risk_ratio_quantiles"] is None
    assert r["expected_population_variance"] == pytest.approx(4 / 7)
    assert lab.exact_reference(10, 2, np.eye(2), 1)["population_risk_ratio_quantiles"]["0.5"] == 1


@pytest.mark.parametrize(
    "config",
    [
        {"repetitions": 1},
        {"n_assets": 1},
        {"train_size": True},
        {"seed": -1},
        {"shrinkages": []},
        {"shrinkages": [0, 0]},
        {"shrinkages": [True]},
        {"shrinkages": [float("nan")]},
        {"families": []},
    ],
)
def test_invalid_config(config):
    with pytest.raises((ValueError, TypeError)):
        lab.run_lab(**config)


@pytest.mark.parametrize(
    "covariance",
    [
        np.zeros((2, 2)),
        np.array([[1, 2], [2, 1]]),
        np.array([[1, 2], [0, 1]]),
        np.ones((2, 3)),
        np.eye(1),
        np.eye(2) * np.nan,
    ],
)
def test_invalid_covariance(covariance):
    with pytest.raises((ValueError, TypeError)):
        lab.population_reference(covariance)


def test_tiny_finite_covariance_is_normalized_without_weight_corruption():
    r = lab.population_reference(np.eye(2) * 1e-308)
    np.testing.assert_allclose(r["oracle_weights"], [0.5, 0.5])
    assert r["oracle_variance"] > 0


def test_cli_report(tmp_path):
    out = tmp_path / "covariance.json"
    subprocess.run(
        [sys.executable, str(MODULE), "--repetitions", "2", "--out", str(out)],
        check=True,
        capture_output=True,
    )
    report = json.loads(out.read_text())
    assert report["lab"] == "covariance_estimation"
    assert len(report["provenance"]["source_sha256"]) == 64
    assert report["performance"]["wall_seconds"] > 0


def test_constant_summary_has_exact_mean_and_zero_mcse_without_changing_moment_contract():
    value = 0.7322587533203613
    summary = lab._summary([value] * 5000)
    assert summary["empirical_mean"] == value
    assert summary["monte_carlo_standard_error"] == 0
    undefined = lab._summary([value] * 5000, mean_exists=False, variance_exists=False)
    assert undefined["empirical_mean"] is None
    assert undefined["monte_carlo_standard_error"] is None

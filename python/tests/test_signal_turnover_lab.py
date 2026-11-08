"""Analytic, chronology and independent-path contracts for the signal lab."""

from __future__ import annotations

import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

MODULE = Path(__file__).resolve().parents[2] / "experiments" / "signal_turnover_lab.py"


def load_lab():
    assert MODULE.is_file(), "the signal/turnover lab is not implemented"
    spec = importlib.util.spec_from_file_location("signal_turnover_lab", MODULE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def inputs(**updates):
    values = dict(rho=0.5, eta=0.1, k=1.0, beta=0.001, sigma=0.02)
    values.update(updates)
    return values


@pytest.mark.parametrize("rho", [-0.9, -0.5, 0.0, 0.5, 0.9])
@pytest.mark.parametrize("k", [-2.0, 0.0, 1.0])
def test_hand_derived_full_adjustment_moments(rho, k):
    m = load_lab().stationary_moments(**inputs(rho=rho, eta=1, k=k))
    assert m["position_signal_product"] == k
    assert m["position_second_moment"] == k * k
    assert m["conditional_position_variance"] == 0
    assert m["squared_turnover"] == pytest.approx(2 * k * k * (1 - rho))
    assert m["absolute_turnover"] == pytest.approx(2 * abs(k) * math.sqrt((1 - rho) / math.pi))
    assert m["gross_pnl"] == pytest.approx(0.001 * k)
    assert m["forecast_mse"] == 0.02**2
    assert m["forecast_correlation"] == pytest.approx(0.001 / math.hypot(0.001, 0.02))


@pytest.mark.parametrize("rho", [-0.8, 0.0, 0.9])
@pytest.mark.parametrize("eta", [0.01, 0.1, 0.5, 1.0])
def test_independent_discrete_lyapunov_reference(rho, eta):
    # Y_t = A Y_(t-1) + B z_t, Y=(signal, position). Solve vec(P)=vec(A P A'+BB').
    k, a = -1.3, 1 - eta
    b = math.sqrt(1 - rho * rho) * np.array([1, eta * k])
    transition = np.array([[rho, 0], [eta * k * rho, a]])
    covariance = np.linalg.solve(
        np.eye(4) - np.kron(transition, transition), np.outer(b, b).reshape(-1)
    ).reshape(2, 2)
    lag = transition @ covariance
    delta_variance = 2 * (covariance[1, 1] - lag[1, 1])
    m = load_lab().stationary_moments(**inputs(rho=rho, eta=eta, k=k))
    assert covariance[0, 0] == pytest.approx(1)
    assert m["position_signal_product"] == pytest.approx(covariance[1, 0], rel=2e-13)
    assert m["position_second_moment"] == pytest.approx(covariance[1, 1], rel=2e-13)
    assert m["squared_turnover"] == pytest.approx(delta_variance, rel=2e-12)
    conditional = covariance[1, 1] - covariance[1, 0] ** 2
    assert m["conditional_position_variance"] == pytest.approx(conditional, abs=2e-14)
    # The declared initialization has covariance P, and one state update preserves it.
    root = np.array(
        [[1, 0], [m["position_signal_product"], math.sqrt(m["conditional_position_variance"])]]
    )
    initialized = root @ root.T
    assert initialized == pytest.approx(covariance, abs=2e-14)
    assert transition @ initialized @ transition.T + np.outer(b, b) == pytest.approx(initialized)


def test_cash_zero_risk_zero_return_and_persistence_boundary():
    lab = load_lab()
    m = lab.stationary_moments(**inputs(eta=0))
    for key in (
        "position_signal_product",
        "position_second_moment",
        "conditional_position_variance",
        "squared_turnover",
        "absolute_turnover",
        "gross_pnl",
    ):
        assert m[key] == 0
    assert lab.stationary_moments(**inputs(beta=0, sigma=0))["forecast_correlation"] is None
    assert lab.stationary_moments(**inputs(beta=-2, sigma=0))["forecast_correlation"] == -1
    for rho in (math.nextafter(-1.0, 0), math.nextafter(1.0, 0)):
        m = lab.stationary_moments(**inputs(rho=rho, eta=math.nextafter(0.0, 1.0), k=0))
        assert m["position_second_moment"] == 0
        m = lab.stationary_moments(**inputs(rho=rho, eta=0.1))
        assert m["squared_turnover"] > 0
        assert m["conditional_position_variance"] > 0


def path(**updates):
    values = dict(
        signal_innovations=[2.0, -1.0, 0.0],
        return_innovations=[3.0, -2.0, 1.0],
        initial_signal=2.0,
        initial_residual=0.0,
        rho=0.0,
        eta=0.5,
        k=1.0,
        beta=0.1,
        sigma=0.2,
        cost_rate=0.1,
        cost_model="proportional",
        keep_records=True,
    )
    values.update(updates)
    return values


def test_hand_path_has_correct_decision_return_and_cost_chronology():
    r = load_lab().simulate(**path())
    records = r["records"]
    assert r["initial_position"] == 1
    assert [x["signal"] for x in records] == [2, -1, 0]
    assert [x["position"] for x in records] == [1.5, 0.25, 0.125]
    assert [x["trade"] for x in records] == [0.5, -1.25, -0.125]
    assert [x["next_return"] for x in records] == pytest.approx([0.8, -0.5, 0.2])
    assert [x["gross_pnl"] for x in records] == pytest.approx([1.2, -0.125, 0.025])
    assert [x["cost"] for x in records] == pytest.approx([0.05, 0.125, 0.0125])
    assert r["metrics"]["net_pnl"] == pytest.approx((1.1 - 0.1875) / 3)
    q = load_lab().simulate(**path(cost_model="quadratic"))
    assert [x["cost"] for x in q["records"]] == pytest.approx([0.0125, 0.078125, 0.00078125])


def test_future_suffix_and_unseen_return_innovations_cannot_change_positions():
    lab = load_lab()
    original = lab.simulate(**path())
    suffix = lab.simulate(**path(signal_innovations=[2, -1, 99], return_innovations=[3, -2, -99]))
    assert original["records"][:2] == suffix["records"][:2]
    other_returns = lab.simulate(**path(return_innovations=[-50, 99, 0]))
    assert [x["position"] for x in original["records"]] == [
        x["position"] for x in other_returns["records"]
    ]


@pytest.mark.parametrize("model", ["proportional", "quadratic"])
def test_zero_cost_and_cash_controls(model):
    lab = load_lab()
    zero = lab.simulate(**path(cost_model=model, cost_rate=0))
    assert zero["metrics"]["cost"] == 0
    assert zero["metrics"]["net_pnl"] == zero["metrics"]["gross_pnl"]
    cash = lab.simulate(**path(cost_model=model, eta=0, initial_residual=19))
    assert cash["initial_position"] == 0
    for key in (
        "position_second_moment",
        "absolute_turnover",
        "squared_turnover",
        "gross_pnl",
        "cost",
        "net_pnl",
    ):
        assert cash["metrics"][key] == 0


@pytest.mark.parametrize(
    "name,value",
    [
        ("rho", -1),
        ("rho", 1),
        ("eta", -0.01),
        ("eta", 1.01),
        ("sigma", -1),
        ("beta", float("nan")),
        ("k", float("inf")),
    ],
)
def test_invalid_numeric_domains(name, value):
    with pytest.raises(ValueError):
        load_lab().stationary_moments(**inputs(**{name: value}))


@pytest.mark.parametrize("value", [True, np.bool_(False), "0.5", [0.5], 1j, None])
def test_scalar_types_are_not_silently_coerced(value):
    with pytest.raises((TypeError, ValueError)):
        load_lab().stationary_moments(**inputs(rho=value))


@pytest.mark.parametrize(
    "updates",
    [
        dict(signal_innovations=[]),
        dict(signal_innovations=[1, 2]),
        dict(signal_innovations=[True, 1, 2]),
        dict(return_innovations=[1, float("inf"), 2]),
        dict(cost_rate=-1),
        dict(cost_model="unknown"),
        dict(initial_signal=True),
        dict(signal_innovations=[[1, 2, 3]]),
    ],
)
def test_path_validation(updates):
    with pytest.raises((TypeError, ValueError)):
        load_lab().simulate(**path(**updates))


def test_unrepresentable_moments_or_path_arithmetic_raise_not_economic_output():
    lab = load_lab()
    for updates in (dict(k=1e308), dict(sigma=1e308), dict(k=1e-300), dict(beta=1e308, k=2, eta=1)):
        with pytest.raises(ArithmeticError, match="binary64|finite|arithmetic"):
            lab.stationary_moments(**inputs(**updates))
    with pytest.raises(ArithmeticError, match="binary64|finite|arithmetic"):
        lab.simulate(**path(initial_signal=1e308, signal_innovations=[1e308] * 3, rho=0.9))
    with pytest.raises(ArithmeticError):
        lab.simulate(**path(cost_rate=1e308))


def small_grid():
    return dict(
        seed=123,
        repetitions=4,
        horizon=8,
        rhos=[0, 0.5],
        etas=[0, 0.1, 1],
        betas=[0, 0.001],
        cost_rates=[0, 0.001],
    )


def scientific(report):
    return {key: value for key, value in report.items() if key not in ("performance", "provenance")}


def test_serialized_config_replay_and_path_based_standard_errors():
    lab = load_lab()
    report = lab.run_lab(**small_grid())
    again = lab.run_lab(**json.loads(json.dumps(report["config"])))
    assert scientific(report) == scientific(again)
    assert report["lab"] == "signal_turnover"
    assert len(report["cells"]) == 48
    assert len(report["primitive_paths"]) == 4
    assert len({x["sha256"] for x in report["primitive_paths"]}) == 4
    for cell in report["cells"]:
        assert cell["replication_ids"] == list(range(4))
        for metric, values in cell["per_replication"].items():
            summary = cell["summary"][metric]
            assert summary["mean"] == pytest.approx(np.mean(values))
            assert summary["mc_se"] == pytest.approx(np.std(values, ddof=1) / 2)
            assert summary["deviation"] == pytest.approx(summary["mean"] - summary["analytic"])
        if cell["beta"] == 0:
            assert cell["summary"]["gross_pnl"]["analytic"] == 0
            assert cell["summary"]["net_pnl"]["analytic"] <= 0
        if cell["cost_rate"] == 0:
            assert cell["per_replication"]["gross_pnl"] == cell["per_replication"]["net_pnl"]
    json.dumps(report, allow_nan=False)


def test_shared_primitives_policies_and_costs_do_not_change_forecasts():
    report = load_lab().run_lab(**small_grid())
    for rho in (0, 0.5):
        cells = [x for x in report["cells"] if x["rho"] == rho]
        assert all(
            x["per_replication"]["signal_second_moment"]
            == cells[0]["per_replication"]["signal_second_moment"]
            for x in cells
        )
        assert all(
            x["per_replication"]["forecast_mse"] == cells[0]["per_replication"]["forecast_mse"]
            for x in cells
        )
    forecast = [x["summary"]["forecast_mse"]["analytic"] for x in report["cells"]]
    assert len(set(forecast)) == 1


@pytest.mark.parametrize(
    "updates",
    [
        dict(repetitions=1),
        dict(repetitions=True),
        dict(horizon=0),
        dict(seed=-1),
        dict(etas=[0, 0]),
        dict(rhos=[1]),
        dict(betas=[]),
        dict(cost_rates=[-1]),
        dict(cost_models=["proportional", "proportional"]),
        dict(horizon=1000000),
        dict(seed="1"),
    ],
)
def test_study_configuration_validation(updates):
    with pytest.raises((TypeError, ValueError)):
        load_lab().run_lab(**{**small_grid(), **updates})


@pytest.mark.parametrize("symlink", [False, True])
def test_cli_refuses_overwrite_before_running_study(tmp_path, symlink):
    load_lab()
    target = tmp_path / "target.json"
    target.write_text("preserve me")
    output = tmp_path / "output.json" if symlink else target
    if symlink:
        output.symlink_to(target)
    result = subprocess.run(
        [sys.executable, str(MODULE), "--out", str(output)], text=True, capture_output=True
    )
    assert result.returncode != 0
    assert "refusing to overwrite" in result.stderr
    assert target.read_text() == "preserve me"


def test_simulation_rejects_unrepresentable_adjustment_and_underflow():
    lab = load_lab()
    with pytest.raises(ArithmeticError):
        lab.simulate(**path(eta=1e-20))
    with pytest.raises(ArithmeticError):
        lab.simulate(**path(k=1e-160))


def test_retained_rows_match_independently_reconstructed_replication_primitives():
    import hashlib

    lab = load_lab()
    settings = small_grid()
    report = lab.run_lab(**settings)
    rep = 2
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([settings["seed"], rep])))
    initial = rng.standard_normal(2)
    draws = rng.standard_normal((settings["horizon"], 2))
    raw = np.concatenate([initial, draws.reshape(-1)]).astype("<f8").tobytes()
    assert report["primitive_paths"][rep]["sha256"] == hashlib.sha256(raw).hexdigest()
    for cell in report["cells"]:
        single = lab.simulate(
            draws[:, 0],
            draws[:, 1],
            initial_signal=initial[0],
            initial_residual=initial[1],
            rho=cell["rho"],
            eta=cell["eta"],
            beta=cell["beta"],
            k=report["config"]["k"],
            sigma=report["config"]["sigma"],
            cost_rate=cell["cost_rate"],
            cost_model=cell["cost_model"],
        )
        for key in single["metrics"]:
            assert single["metrics"][key] == cell["per_replication"][key][rep]


def test_nonzero_signal_return_correlation_cannot_silently_underflow():
    # MSE=1e200 and gross ~1.8e-301 are representable; correlation ~1e-400 is not.
    with pytest.raises(ArithmeticError, match="forecast_correlation"):
        load_lab().stationary_moments(**inputs(beta=1e-300, sigma=1e100))


def test_correlation_is_signed_signal_return_correlation_not_forecast_return():
    lab = load_lab()
    positive = lab.stationary_moments(**inputs(beta=0.001))
    negative = lab.stationary_moments(**inputs(beta=-0.001))
    null = lab.stationary_moments(**inputs(beta=0))
    assert negative["forecast_correlation"] == -positive["forecast_correlation"]
    assert null["forecast_correlation"] == 0

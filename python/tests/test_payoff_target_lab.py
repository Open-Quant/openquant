"""Independent bounded-stopping and target-mismatch checks."""

import importlib.util
import itertools
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    "payoff_target_lab", Path(__file__).resolve().parents[2] / "experiments/payoff_target_lab.py"
)
assert spec is not None and spec.loader is not None
lab = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lab)


@pytest.mark.parametrize("p", [0.0, 0.4, 0.6, 0.85, 1.0])
@pytest.mark.parametrize("horizon", [1, 4, 8])
def test_terminal_distribution_and_wald_exhaustive(p, horizon):
    boundary = lab.sequential.crossing_boundaries(horizon, alpha=0.5)["fixed_mixture"]
    expected = {}
    expected_payoff = expected_tau = 0.0
    win, loss, cost = 1.0, 2.0, 0.25
    for path in itertools.product((0, 1), repeat=horizon):
        weight = p ** sum(path) * (1 - p) ** (horizon - sum(path))
        successes = 0
        for t, x in enumerate(path, 1):
            successes += x
            if successes >= boundary[t] or t == horizon:
                crossed = successes >= boundary[t]
                key = (t, successes, bool(crossed))
                expected[key] = expected.get(key, 0.0) + weight
                expected_tau += weight * t
                expected_payoff += weight * (successes * win - (t - successes) * loss - t * cost)
                break
    terminal = lab.terminal_distribution(boundary, p)
    actual = {(r["time"], r["successes"], r["crossed"]): r["mass"] for r in terminal}
    for key, weight in expected.items():
        assert actual.get(key, 0.0) == pytest.approx(weight, abs=2e-15)
    result = lab.terminal_payoffs(terminal, win=win, loss=loss, cost=cost)
    assert result["expected_net_payoff"] == pytest.approx(expected_payoff, abs=2e-13)
    assert result["expected_rows"] == pytest.approx(expected_tau)
    assert expected_payoff == pytest.approx(
        expected_tau * (p * win - (1 - p) * loss - cost), abs=2e-13
    )


def test_no_crossing_paths_are_charged_all_rows():
    boundary = np.arange(9) + 1
    law = lab.terminal_distribution(boundary, 0.6)
    assert all(r["time"] == 8 and not r["crossed"] for r in law)
    result = lab.terminal_payoffs(law, win=1, loss=1, cost=0.25)
    assert result["crossing_probability"] == 0
    assert result["conditional_crossed_mean_net_descriptive"] is None
    assert result["expected_net_payoff"] == pytest.approx(8 * (0.2 - 0.25))


def test_cost_changes_net_sign_target_but_not_gross():
    tie = lab.economics(0.85, 0.2, 1, 0.2)
    negative = lab.economics(0.85, 0.2, 1, 0.25)
    assert tie["net_zero_probability"] == 0.85
    assert tie["net_positive_probability"] == 0
    assert negative["gross_positive_probability"] == 0.85
    assert negative["net_positive_probability"] == negative["net_zero_probability"] == 0
    assert negative["expected_net_per_row"] == pytest.approx(-0.23)


def test_maps_share_evidence_and_cost_difference_exact():
    report = lab.run_lab(horizons=[16], repetitions=12)
    for group in report["groups"]:
        tau = np.array(group["observed_rows"])
        scenarios = group["scenarios"]
        for i in range(0, len(scenarios), 2):
            gross = np.array(scenarios[i]["stopped"]["sample"]["values"])
            net = np.array(scenarios[i + 1]["stopped"]["sample"]["values"])
            np.testing.assert_allclose(gross - net, 0.25 * tau, atol=2e-15)
            assert (
                scenarios[i]["economics"]["gross_positive_probability"]
                == scenarios[i + 1]["economics"]["gross_positive_probability"]
            )
        assert len(group["gross_sign_first_crossing_rows"]) == 12
    assert lab.run_lab(**report["config"])["groups"] == report["groups"]


def test_gross_hitrate_same_expected_payoff_opposite_sign():
    good = lab.economics(0.6, 1, 1, 0)
    bad = lab.economics(0.6, 1, 2, 0)
    assert good["gross_positive_probability"] == bad["gross_positive_probability"]
    assert good["expected_gross_per_row"] > 0 > bad["expected_gross_per_row"]


@pytest.mark.parametrize(
    "win,loss,cost", [(0, 1, 0), (1, 0, 0), (1, 1, -1), (np.nan, 1, 0), (True, 1, 0), (1, 1, 1j)]
)
def test_invalid_economics(win, loss, cost):
    with pytest.raises((ValueError, TypeError)):
        lab.economics(0.6, win, loss, cost)


def test_payoff_overflow_rejected():
    with pytest.raises(ValueError):
        lab.run_lab(
            horizons=[64],
            repetitions=1,
            maps=[dict(id="huge", p=0.6, win=1e308, loss=1e308, cost=0)],
        )


def test_sample_summary_large_finite_values():
    summary = lab._sample_summary([1e200, -1e200])
    assert summary["mean"] == 0
    assert summary["mcse"] == pytest.approx(1e200)

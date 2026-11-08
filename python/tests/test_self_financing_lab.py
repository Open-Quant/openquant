"""Failures target fee-base errors, lost cash, and unannounced normalization."""

import importlib.util
from pathlib import Path

import pytest

PATH = Path(__file__).resolve().parents[2] / "experiments/self_financing_lab.py"


def load_lab():
    assert PATH.is_file(), "self-financing fee lab has not been implemented"
    spec = importlib.util.spec_from_file_location("self_financing_lab", PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("fee", [0.0, 0.001, 0.1, 0.9])
def test_initial_purchase_charges_actual_postfee_notional(fee):
    lab = load_lab()
    result = lab.rebalance([0.0, 0.0], [0.25, 0.75], fee)
    assert result["wealth_fraction"] == pytest.approx(1 / (1 + fee), rel=0, abs=3e-15)
    assert result["fee_fraction"] == pytest.approx(fee / (1 + fee), rel=0, abs=3e-15)


@pytest.mark.parametrize("fee", [0.0, 0.001, 0.1, 0.9])
def test_complete_switch_charges_sale_plus_smaller_purchase(fee):
    lab = load_lab()
    result = lab.rebalance([1.0, 0.0], [0.0, 1.0], fee)
    assert result["wealth_fraction"] == pytest.approx((1 - fee) / (1 + fee), rel=0, abs=3e-15)


def test_unchanged_fully_invested_holdings_are_not_traded():
    lab = load_lab()
    result = lab.rebalance([0.25, 0.75], [0.25, 0.75], 0.9)
    assert result["wealth_fraction"] == 1
    assert result["fee_fraction"] == 0
    assert result["trade_fractions"] == [0.0, 0.0]


@pytest.mark.parametrize(
    "w,v,fee",
    [
        ([-0.1, 1.1], [0.5, 0.5], 0.1),
        ([0, 0], [0.4, 0.4], 0.1),
        ([0, 0], [0.5, 0.5], 1),
        ([0, 0], [0.5, 0.5], -0.1),
        ([0, 0], [0.5, 0.5], True),
        ([0, float("nan")], [0.5, 0.5], 0.1),
        ([0, 0], [0.5, 0.5], float("inf")),
        ([1, 1], [0.5, 0.5], 0.1),
        ([0], [0.5, 0.5], 0.1),
        ([], [], 0.1),
    ],
)
def test_invalid_contract_rejects(w, v, fee):
    lab = load_lab()
    with pytest.raises((ValueError, TypeError)):
        lab.rebalance(w, v, fee)


def test_proxy_error_has_both_signs_at_same_turnover():
    lab = load_lab()
    outward = lab.rebalance([1, 0], [0.9, 0.1], 0.1)
    inward = lab.rebalance([0.9, 0.1], [1, 0], 0.1)
    assert outward["wealth_fraction"] == pytest.approx(45 / 46)
    assert inward["wealth_fraction"] == pytest.approx(54 / 55)
    assert outward["wealth_fraction"] < 0.98 < inward["wealth_fraction"]


def test_permutation_preserves_wealth_and_cash_funding():
    lab = load_lab()
    a = lab.rebalance([0.1, 0.3, 0.2], [0.25, 0.25, 0.5], 0.01)
    b = lab.rebalance([0.2, 0.1, 0.3], [0.5, 0.25, 0.25], 0.01)
    assert a["wealth_fraction"] == pytest.approx(b["wealth_fraction"], rel=0, abs=2e-15)
    assert sum(a["trade_fractions"]) + a["fee_fraction"] == pytest.approx(0.4, rel=0, abs=2e-15)
    assert abs(a["self_financing_residual"]) < 2e-15


def test_cash_ledger_never_trades_or_observes_future_for_decisions():
    lab = load_lab()
    assert hasattr(lab, "simulate"), "portfolio ledger is not implemented"
    r = lab.simulate([[0.1, -0.2], [-0.1, 0.3]], policy="cash", fee=0.1)
    assert r["terminal_wealth"] == 1
    assert r["total_fees"] == 0
    assert r["rebalance_attempt_count"] == 0


@pytest.mark.parametrize("fee", [0, 0.0001, 0.001, 0.01])
def test_two_period_rebalance_threshold_has_independent_formula(fee):
    lab = load_lab()
    assert hasattr(lab, "simulate"), "portfolio ledger is not implemented"
    a = 0.002
    path = [[a, 0], [-a / (1 + a), 0]]
    hold = lab.simulate(path, policy="buy_hold", fee=fee)
    daily = lab.simulate(path, policy="daily", fee=fee)
    relative = (a + 2) ** 2 / (4 * (1 + a)) * (1 - fee * a / (a + 2))
    assert hold["terminal_wealth"] == pytest.approx(1 / (1 + fee), rel=0, abs=3e-15)
    assert daily["terminal_wealth"] / hold["terminal_wealth"] == pytest.approx(
        relative, rel=0, abs=3e-15
    )
    assert (daily["terminal_wealth"] > hold["terminal_wealth"]) == (fee < a / (a + 2))


def test_future_suffix_cannot_change_earlier_ledger():
    lab = load_lab()
    assert hasattr(lab, "simulate"), "portfolio ledger is not implemented"
    a = lab.simulate([[0.02, -0.02]] * 10, policy="drift_02", fee=0.001, keep_records=True)
    b = lab.simulate(
        [[0.02, -0.02]] * 5 + [[-0.03, 0.01]] * 5, policy="drift_02", fee=0.001, keep_records=True
    )
    assert a["records"][:5] == b["records"][:5]


@pytest.mark.parametrize("path", [[[-1, 0]], [[float("nan"), 0]], [[True, 0]], [], [[1], [2, 3]]])
def test_invalid_returns_fail_without_clipping(path):
    lab = load_lab()
    assert hasattr(lab, "simulate"), "portfolio ledger is not implemented"
    with pytest.raises((ValueError, TypeError)):
        lab.simulate(path, policy="buy_hold", fee=0.001)


def test_report_is_replayable_and_retains_paired_paths():
    lab = load_lab()
    assert hasattr(lab, "run_lab"), "prespecified report runner is not implemented"
    a = lab.run_lab(horizon=8, repetitions=2, fee_rates=[0, 0.001], seed=19)
    b = lab.run_lab(**a["config"])
    assert a["reports"] == b["reports"]
    assert a["paths"] == b["paths"]
    assert len(a["reports"]) == 6 * 2 * 6 * 2
    assert all(x["status"] == "ok" for x in a["reports"])
    assert len({x["path_sha256"] for x in a["reports"]}) <= 6


@pytest.mark.parametrize(
    "kwargs",
    [
        {"horizon": 0},
        {"repetitions": True},
        {"seed": -1},
        {"fee_rates": []},
        {"repetitions": 0},
        {"fee_rates": [0.01, 0.01]},
    ],
)
def test_invalid_run_budget_and_duplicate_grid_reject(kwargs):
    lab = load_lab()
    assert hasattr(lab, "run_lab"), "prespecified report runner is not implemented"
    with pytest.raises((ValueError, TypeError)):
        lab.run_lab(**kwargs)


def test_near_total_loss_does_not_invent_cash_or_double_surviving_wealth():
    import math

    lab = load_lab()
    loss = math.nextafter(-1.0, 0.0)
    result = lab.simulate([[loss] * 3, [0.0] * 3], policy="daily", fee=0, keep_records=True)
    assert result["terminal_wealth"] == pytest.approx(1 + loss, rel=5e-15, abs=0)
    assert sum(result["records"][0]["ending_weights"]) == pytest.approx(1, rel=0, abs=1e-15)


def test_mixed_boolean_weight_cannot_silently_become_a_position():
    lab = load_lab()
    with pytest.raises(TypeError):
        lab.rebalance([0, False], [0.5, 0.5], 0.1)


def test_roundoff_in_risky_weights_is_not_cash_at_the_next_total_loss():
    import math

    lab = load_lab()
    first = [-0.810836844161442, -0.13317924646728885]
    loss = math.nextafter(-1.0, 0.0)
    result = lab.simulate([first, [loss, loss]], policy="buy_hold", fee=0, keep_records=True)
    expected = math.fsum([(1 + first[0]) / 2, (1 + first[1]) / 2]) * (1 + loss)
    assert result["terminal_wealth"] == pytest.approx(expected, rel=5e-15, abs=0)
    assert result["records"][0]["cash_weight"] == 0


def test_offline_replay_accepts_report_and_rejects_changed_scientific_results():
    import copy

    lab = load_lab()
    path = PATH.parent / "replay_offline_report.py"
    spec = importlib.util.spec_from_file_location("fee_replay", path)
    assert spec is not None and spec.loader is not None
    replay = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(replay)
    assert "self_financing" in replay.LABS, "new lab is not registered for exact replay"
    report = lab.run_lab(horizon=4, repetitions=1, fee_rates=[0.001])
    assert replay.replay_report(report, lab="self_financing")["matched"]
    changed = copy.deepcopy(report)
    changed["reports"][0]["terminal_wealth"] += 0.01
    assert not replay.replay_report(changed, lab="self_financing")["matched"]


def test_stress_fee_normalizes_binary_coefficients_before_ill_conditioned_root():
    import math
    from fractions import Fraction

    lab = load_lab()
    c = math.nextafter(1.0, 0.0)
    epsilon = 2.0**-53
    # Exact simplex target is [1/(1+eps), eps/(1+eps)], not [1,eps].
    cc = Fraction(c)
    ee = Fraction(epsilon)
    expected = (1 - cc) / (1 + cc * (ee - 1) / (1 + ee))
    result = lab.rebalance([1, 0], [1, epsilon], c)
    assert result["wealth_fraction"] == pytest.approx(float(expected), rel=1e-15)
    assert result["normalization"] == "exact_rational_stress"

"""Independent holdings oracles catch omitted/double exit fees and fee-base errors."""

import importlib.util
import math
from fractions import Fraction as F
from pathlib import Path

import pytest

PATH = Path(__file__).resolve().parents[2] / "experiments/terminal_liquidation_lab.py"


def load_lab():
    assert PATH.exists(), "terminal liquidation lab is not implemented"
    spec = importlib.util.spec_from_file_location("terminal_liquidation_lab", PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_flat_round_trip_charges_actual_entry_and_sale_notionals():
    r = load_lab().simulate([[0, 0]], policy="buy_hold", entry_fee=0.1, trade_fee=0.7, exit_fee=0.2)
    assert r["mark_to_market_wealth"] == pytest.approx(10 / 11)
    assert r["liquidated_wealth"] == pytest.approx(8 / 11)
    assert r["entry_fees"] == pytest.approx(1 / 11)
    assert r["exit_fee_amount"] == pytest.approx(2 / 11)
    assert r["ongoing_fees"] == 0
    assert r["total_fees"] == pytest.approx(3 / 11)


def test_terminal_cash_is_not_charged_and_zero_inventory_stays_cash():
    lab = load_lab()
    r = lab.liquidate([0.2, 0.3], cash=0.5, exit_fee=0.1)
    assert r["liquidated_wealth"] == pytest.approx(0.95)
    assert r["exit_traded_notional"] == 0.5
    assert abs(r["terminal_accounting_residual"]) < 1e-15
    r = lab.simulate([[0.1, -0.5]], policy="cash", entry_fee=0.2, trade_fee=0.3, exit_fee=0.9)
    assert r["liquidated_wealth"] == r["mark_to_market_wealth"] == 1
    assert r["total_fees"] == r["exit_traded_notional"] == 0


def test_rational_two_asset_daily_oracle_accounts_for_fee_funded_sale():
    # Initial holdings each 5/11; after returns [1/4, 0], holdings [25/44, 5/11].
    # Equal-dollar rebalance sells h1-x, buys x-h2: total traded h1-h2.
    # Thus post-trade wealth = h1+h2-c*(h1-h2), not a prefee target debit.
    entry, trade, exit_fee = F(1, 10), F(1, 5), F(3, 10)
    h1, h2 = F(5, 11) * F(5, 4), F(5, 11)
    ongoing = trade * (h1 - h2)
    x = (h1 + h2 - ongoing) / 2
    terminal = x * F(4, 5) + x * F(11, 10)
    r = load_lab().simulate(
        [[0.25, 0], [-0.2, 0.1]],
        policy="daily",
        entry_fee=float(entry),
        trade_fee=float(trade),
        exit_fee=float(exit_fee),
        keep_records=True,
    )
    assert r["mark_to_market_wealth"] == pytest.approx(float(terminal), abs=2e-15)
    assert r["liquidated_wealth"] == pytest.approx(float(terminal * (1 - exit_fee)))
    assert r["ongoing_fees"] == pytest.approx(float(ongoing))
    assert r["total_fees"] == pytest.approx(
        float(entry / (1 + entry) + ongoing + exit_fee * terminal)
    )
    assert len(r["records"]) == 2  # no fictitious final rebalance before liquidation


def test_liquidation_changes_cash_comparison_but_not_fully_invested_policy_order():
    lab = load_lab()
    r = lab.simulate([[0.02]], policy="buy_hold", entry_fee=0.001, trade_fee=0, exit_fee=0.03)
    assert r["mark_to_market_excess_vs_cash"] > 0
    assert r["liquidated_excess_vs_cash"] < 0
    assert r["cash_comparison_reversal"]
    path = [[0.1, 0], [-0.05, 0.03]]
    rows = [
        lab.simulate(path, policy=p, entry_fee=0.01, trade_fee=0.02, exit_fee=0.03)
        for p in ("daily", "buy_hold")
    ]
    assert rows[0]["liquidated_wealth"] / rows[1]["liquidated_wealth"] == pytest.approx(
        rows[0]["mark_to_market_wealth"] / rows[1]["mark_to_market_wealth"]
    )


def test_exit_choice_cannot_change_preterminal_ledger():
    lab = load_lab()
    kwargs = dict(policy="daily", entry_fee=0.01, trade_fee=0.02, keep_records=True)
    a = lab.simulate([[0.02, -0.01]] * 4, exit_fee=0, **kwargs)
    b = lab.simulate([[0.02, -0.01]] * 4, exit_fee=0.5, **kwargs)
    assert a["records"] == b["records"]
    assert a["mark_to_market_wealth"] == b["mark_to_market_wealth"]


def test_near_one_exit_preserves_surviving_cash_without_cancellation():
    fee = math.nextafter(1, 0)
    r = load_lab().liquidate([1e-100], cash=0, exit_fee=fee)
    assert r["liquidated_wealth"] == pytest.approx(1e-100 * (1 - fee), rel=1e-15, abs=0)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(exit_fee=True),
        dict(entry_fee=-0.1),
        dict(trade_fee=1),
        dict(exit_fee=float("nan")),
        dict(policy="optimized"),
    ],
)
def test_invalid_simulation_contract_rejects(kwargs):
    params = dict(policy="buy_hold", entry_fee=0, trade_fee=0, exit_fee=0)
    params.update(kwargs)
    with pytest.raises((ValueError, TypeError)):
        load_lab().simulate([[0.01]], **params)


@pytest.mark.parametrize("values", [[], [[-1]], [[True, 0]], [[float("inf")]], [[1], [1, 2]]])
def test_invalid_path_rejects(values):
    with pytest.raises((ValueError, TypeError)):
        load_lab().simulate(values, policy="cash", entry_fee=0, trade_fee=0, exit_fee=0)


@pytest.mark.parametrize(
    "holdings,cash", [([-1], 1), ([True], 0), ([0.1], -1), ([0.1], True), ([float("nan")], 0)]
)
def test_invalid_inventory_rejects(holdings, cash):
    with pytest.raises((ValueError, TypeError)):
        load_lab().liquidate(holdings, cash=cash, exit_fee=0.1)


def test_report_replays_and_covers_prespecified_scenarios():
    lab = load_lab()
    a = lab.run_lab(horizon=4, repetitions=2, exit_fee_rates=[0, 0.03], seed=11)
    b = lab.run_lab(**a["config"])
    assert a["reports"] == b["reports"]
    assert a["paths"] == b["paths"]
    assert a["lab"] == "terminal_liquidation"
    assert len(a["reports"]) == len(a["paths"]) * 3 * 2
    assert all(row["status"] == "ok" for row in a["reports"])
    assert "self_financing_lab.py" in a["provenance"]["dependency_sha256"]


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(horizon=0),
        dict(repetitions=True),
        dict(seed=-1),
        dict(exit_fee_rates=[]),
        dict(exit_fee_rates=[0, 0]),
        dict(horizon=10001),
    ],
)
def test_report_rejects_invalid_budget(kwargs):
    with pytest.raises((ValueError, TypeError)):
        load_lab().run_lab(**kwargs)

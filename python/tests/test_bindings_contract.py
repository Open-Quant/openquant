import csv
from pathlib import Path

import pytest

import openquant


def _load_fixture_prices(max_rows: int = 64, symbols=("EEM", "EWG", "TIP")):
    fixture_path = (
        Path(__file__).resolve().parents[2]
        / "tests"
        / "fixtures"
        / "portfolio_optimization"
        / "stock_prices.csv"
    )
    rows = []
    with fixture_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            rows.append([float(row[s]) for s in symbols])
            if i + 1 >= max_rows:
                break
    return rows


def test_risk_metrics_smoke():
    returns = [-0.02, 0.01, 0.015, -0.01, 0.005]
    var_95 = openquant.risk.calculate_value_at_risk(returns, 0.05)
    es_95 = openquant.risk.calculate_expected_shortfall(returns, 0.05)
    cdar_95 = openquant.risk.calculate_conditional_drawdown_risk(returns, 0.05)

    assert var_95 == pytest.approx(-0.01)
    assert es_95 == pytest.approx(-0.02)
    assert isinstance(cdar_95, float)


def test_filters_indices_and_timestamps():
    close = [100.0, 101.2, 100.8, 102.4, 101.9, 103.1]
    ts = [
        "2024-01-01 09:30:00",
        "2024-01-01 09:31:00",
        "2024-01-01 09:32:00",
        "2024-01-01 09:33:00",
        "2024-01-01 09:34:00",
        "2024-01-01 09:35:00",
    ]

    idx = openquant.filters.cusum_filter_indices(close, 0.001)
    ts_out = openquant.filters.cusum_filter_timestamps(close, ts, 0.001)
    z_idx = openquant.filters.z_score_filter_indices(close, 3, 3, 1.0)
    z_ts_out = openquant.filters.z_score_filter_timestamps(close, ts, 3, 3, 1.0)

    assert all(0 <= i < len(close) for i in idx)
    assert len(ts_out) == len(idx)
    assert all(isinstance(x, str) for x in ts_out)
    assert all(0 <= i < len(close) for i in z_idx)
    assert len(z_ts_out) == len(z_idx)


def test_sampling_contracts():
    label_endtime = [(0, 2), (1, 3), (2, 4)]
    bar_index = [0, 1, 2, 3, 4]
    ind_mat = openquant.sampling.get_ind_matrix(label_endtime, bar_index)
    uniq = openquant.sampling.get_ind_mat_average_uniqueness(ind_mat)
    samples = openquant.sampling.seq_bootstrap(ind_mat, sample_length=4, warmup_samples=[0, 1])

    assert len(ind_mat) == len(bar_index)
    assert len(ind_mat[0]) == len(label_endtime)
    assert 0.0 <= uniq <= 1.0
    assert len(samples) == 4
    assert all(0 <= i < len(label_endtime) for i in samples)


def test_bet_sizing_contracts():
    signal = openquant.bet_sizing.get_signal([0.55, 0.6, 0.4], 2, [1.0, -1.0, 1.0])
    disc = openquant.bet_sizing.discrete_signal(signal, 0.1)
    b = openquant.bet_sizing.bet_size(10.0, 0.25, "sigmoid")

    assert len(signal) == 3
    assert len(disc) == 3
    assert isinstance(b, float)

    with pytest.raises(ValueError):
        openquant.bet_sizing.bet_size(10.0, 0.25, "not-a-function")


def test_portfolio_fixture_smoke():
    prices = _load_fixture_prices()
    ivp = openquant.portfolio.allocate_inverse_variance(prices)
    mv = openquant.portfolio.allocate_min_vol(prices)
    msr = openquant.portfolio.allocate_max_sharpe(prices, risk_free_rate=0.0)

    for out in (ivp, mv, msr):
        weights, risk, port_ret, sharpe = out
        assert len(weights) == 3
        assert sum(weights) == pytest.approx(1.0, abs=1e-6)
        assert isinstance(risk, float)
        assert isinstance(port_ret, float)
        assert isinstance(sharpe, float)


def test_portfolio_rejects_ragged_matrix():
    with pytest.raises(ValueError):
        openquant.portfolio.allocate_min_vol([[1.0, 2.0], [3.0]])

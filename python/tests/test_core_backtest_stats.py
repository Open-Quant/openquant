import math

import pytest

from _core_fixtures import load_csv_columns, load_timestamps

from openquant import backtest_stats

DATES = [f"2000-01-{day:02d} 00:00:00" for day in range(1, 11)]


def _load_log_returns():
    path = "backtest_statistics/dollar_bar_sample.csv"
    (close,) = load_csv_columns(path, ["close"])
    timestamps = load_timestamps(path)
    returns = [math.log(close[i]) - math.log(close[i - 1]) for i in range(1, len(close))]
    return timestamps[1:], returns


def test_timing_of_flattening_and_flips():
    # Mirrors crates/openquant/tests/backtest_statistics.rs::test_timing_of_flattening_and_flips
    positions = [1.0, 1.5, 0.5, 0.0, -0.5, -1.0, 0.5, 1.5, 1.5, 1.5]
    out = backtest_stats.timing_of_flattening_and_flips(DATES, positions)
    # flattenings at index 3 and 9, flip at index 6
    assert out == [DATES[3], DATES[6], DATES[9]]


def test_average_holding_period():
    # Mirrors crates/openquant/tests/backtest_statistics.rs::test_average_holding_period
    hold = [0.0, 1.0, 1.0, -1.0, -1.0, 0.0, 0.0, 2.0, 2.0, 0.0]
    assert abs(backtest_stats.average_holding_period(DATES, hold) - 2.0) < 1e-4

    never_closed = [0.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0]
    assert backtest_stats.average_holding_period(DATES, never_closed) is None


def test_bets_concentration():
    # Mirrors crates/openquant/tests/backtest_statistics.rs::test_bets_concentration
    _, returns = _load_log_returns()
    positive = backtest_stats.bets_concentration(returns)
    negative = backtest_stats.bets_concentration([-r for r in returns])
    assert abs(positive - negative) < 1e-5
    assert abs(positive - 2.0111445) < 1e-3


def test_all_bets_concentration():
    # Mirrors crates/openquant/tests/backtest_statistics.rs::test_all_bets_concentration
    timestamps, returns = _load_log_returns()
    positive, negative, time_conc = backtest_stats.all_bets_concentration(timestamps, returns)
    assert abs(positive - 0.0014938) < 1e-5
    assert abs(negative - 0.0016261) < 1e-5
    assert abs(time_conc - 0.0195998) < 1e-5


def test_drawdown_and_time_under_water():
    # Mirrors crates/openquant/tests/backtest_statistics.rs::test_drawdown_and_time_under_water
    dollars = [100.0, 110.0, 90.0, 100.0, 120.0, 130.0, 100.0, 120.0, 140.0, 130.0]
    drawdown, time_under_water = backtest_stats.drawdown_and_time_under_water(
        DATES, dollars, dollars=True
    )
    assert drawdown == [20.0, 30.0, 10.0]
    assert len(time_under_water) == len(drawdown)


def test_sharpe_and_information_ratio():
    # Mirrors crates/openquant/tests/backtest_statistics.rs::test_sharpe_information_ratios
    returns = [0.03, 0.02, 0.01, -0.01, 0.02, 0.01, 0.0, -0.01, 0.01]
    assert abs(backtest_stats.sharpe_ratio(returns, 12.0, 0.005) - 0.987483) < 1e-2
    assert abs(backtest_stats.information_ratio(returns, 0.006, 12.0) - 0.733559) < 1e-2


def test_probabilistic_and_deflated_sharpe_ratio():
    # Mirrors crates/openquant/tests/backtest_statistics.rs::test_probabilistic_deflated_sr
    psr = backtest_stats.probabilistic_sharpe_ratio(1.14, 1.0, 250, 0.0, 3.0)
    assert abs(psr - 0.95727) < 1e-3

    dsr = backtest_stats.deflated_sharpe_ratio(1.14, [3.5, 1.01, 1.02], 250, 0.0, 3.0)
    assert abs(dsr - 0.95836) < 1e-3
    benchmark = backtest_stats.deflated_sharpe_ratio(
        1.14, [0.4, 100.0], 250, 0.0, 3.0, estimates_param=True, benchmark_out=True
    )
    assert abs(benchmark - 1.012241) < 1e-3
    from_params = backtest_stats.deflated_sharpe_ratio(
        1.14, [0.4, 100.0], 250, 0.0, 3.0, estimates_param=True
    )
    assert abs(from_params - 0.941740) < 1e-3


def test_minimum_track_record_length():
    # Mirrors crates/openquant/tests/backtest_statistics.rs::test_minimum_track_record_length
    min_trl = backtest_stats.minimum_track_record_length(1.14, 1.0, 0.0, 3.0, 0.05)
    assert abs(min_trl - 228.73497) < 1e-1


def test_backtest_stats_rejects_bad_timestamp_inputs():
    with pytest.raises(ValueError, match="length mismatch"):
        backtest_stats.average_holding_period(DATES, [1.0])
    with pytest.raises(ValueError, match="invalid datetime"):
        backtest_stats.drawdown_and_time_under_water(["2000-01-01"], [1.0])


def test_deflated_sharpe_ratio_without_estimates_raises_value_error():
    with pytest.raises(ValueError):
        backtest_stats.deflated_sharpe_ratio(1.14, [], 250, 0.0, 3.0)

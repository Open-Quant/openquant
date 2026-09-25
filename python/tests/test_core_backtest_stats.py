import math

import pytest
from _core_fixtures import load_csv_columns, load_json, load_timestamps
from openquant import backtest_stats

DATES = [f"2000-01-{day:02d} 00:00:00" for day in range(1, 11)]

# Expected values from tests/fixtures/backtest_statistics/generate.py (AFML ch. 14 and
# Bailey & Lopez de Prado, computed in numpy/scipy/pandas independently of this library).
REFERENCE = load_json("backtest_statistics/reference.json")


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
    # Hand-derived: long 1 from day 1 flips at day 3 (held 2 days, weight 1), short 1 from
    # day 3 is flattened at day 5 (2 days, weight 1), long 2 from day 7 is flattened at day 9
    # (2 days, weight 2). Weighted mean (2*1 + 2*1 + 2*2) / 4 = 2.
    assert abs(backtest_stats.average_holding_period(DATES, hold) - 2.0) < 1e-12

    never_closed = [0.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0]
    assert backtest_stats.average_holding_period(DATES, never_closed) is None


def test_bets_concentration():
    # Mirrors crates/openquant/tests/backtest_statistics.rs::test_bets_concentration
    _, returns = _load_log_returns()
    positive = backtest_stats.bets_concentration(returns)
    negative = backtest_stats.bets_concentration([-r for r in returns])
    # Weights r / sum(r) are unchanged by negating every return.
    assert abs(positive - negative) < 1e-12
    assert abs(positive - REFERENCE["bets_concentration"]) < 1e-9


def test_all_bets_concentration():
    # Mirrors crates/openquant/tests/backtest_statistics.rs::test_all_bets_concentration
    timestamps, returns = _load_log_returns()
    positive, negative, time_conc = backtest_stats.all_bets_concentration(timestamps, returns)
    expected = REFERENCE["all_bets_concentration"]
    assert abs(positive - expected["positive"]) < 1e-12
    assert abs(negative - expected["negative"]) < 1e-12
    assert abs(time_conc - expected["time"]) < 1e-12


def test_drawdown_and_time_under_water():
    # Mirrors crates/openquant/tests/backtest_statistics.rs::test_drawdown_and_time_under_water
    dollars = [100.0, 110.0, 90.0, 100.0, 120.0, 130.0, 100.0, 120.0, 140.0, 130.0]
    # Hand-derived (AFML snippet 14.4): high-water marks 100, 110, 120, 130, 140. The 110
    # mark falls to 90 (20), 120 is never under water, 130 falls to 100 (30), 140 to 130 (10).
    drawdown, time_under_water = backtest_stats.drawdown_and_time_under_water(
        DATES, dollars, dollars=True
    )
    assert drawdown == [20.0, 30.0, 10.0]
    assert len(time_under_water) == len(drawdown)


def test_sharpe_and_information_ratio():
    # Mirrors crates/openquant/tests/backtest_statistics.rs::test_sharpe_information_ratios
    returns = [0.03, 0.02, 0.01, -0.01, 0.02, 0.01, 0.0, -0.01, 0.01]
    sharpe = backtest_stats.sharpe_ratio(returns, 12.0, 0.005)
    assert abs(sharpe - REFERENCE["sharpe_ratio"]) < 1e-12
    info = backtest_stats.information_ratio(returns, 0.006, 12.0)
    assert abs(info - REFERENCE["information_ratio"]) < 1e-12


def test_probabilistic_and_deflated_sharpe_ratio():
    # Mirrors crates/openquant/tests/backtest_statistics.rs::test_probabilistic_deflated_sr
    psr = backtest_stats.probabilistic_sharpe_ratio(1.14, 1.0, 250, 0.0, 3.0)
    assert abs(psr - REFERENCE["psr"]) < 1e-9

    dsr = backtest_stats.deflated_sharpe_ratio(1.14, [3.5, 1.01, 1.02], 250, 0.0, 3.0)
    assert abs(dsr - REFERENCE["dsr"]) < 1e-9
    benchmark = backtest_stats.deflated_sharpe_ratio(
        1.14, [0.4, 100.0], 250, 0.0, 3.0, estimates_param=True, benchmark_out=True
    )
    assert abs(benchmark - REFERENCE["dsr_benchmark_from_params"]) < 1e-9
    from_params = backtest_stats.deflated_sharpe_ratio(
        1.14, [0.4, 100.0], 250, 0.0, 3.0, estimates_param=True
    )
    assert abs(from_params - REFERENCE["dsr_from_params"]) < 1e-9


def test_minimum_track_record_length():
    # Mirrors crates/openquant/tests/backtest_statistics.rs::test_minimum_track_record_length
    min_trl = backtest_stats.minimum_track_record_length(1.14, 1.0, 0.0, 3.0, 0.05)
    assert abs(min_trl - REFERENCE["min_trl"]) < 1e-7


def test_backtest_stats_rejects_bad_timestamp_inputs():
    with pytest.raises(ValueError, match="length mismatch"):
        backtest_stats.average_holding_period(DATES, [1.0])
    with pytest.raises(ValueError, match="invalid datetime"):
        backtest_stats.drawdown_and_time_under_water(["2000-01-01"], [1.0])


def test_deflated_sharpe_ratio_without_estimates_raises_value_error():
    with pytest.raises(ValueError):
        backtest_stats.deflated_sharpe_ratio(1.14, [], 250, 0.0, 3.0)

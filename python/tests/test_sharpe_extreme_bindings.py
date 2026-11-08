"""Public Python paths retain finite results from extreme native calculations."""

import math

import openquant
import pytest


def test_large_trial_count_benchmark_is_finite():
    assert openquant.evaluation.expected_max_sharpe(10**20, 1.0) == pytest.approx(
        9.323612882361873, abs=1e-11
    )
    assert openquant.evaluation.expected_max_sharpe(10**20, 0.0) == 0.0


def test_raw_native_empirical_dsr_recovers_dispersion():
    result = openquant.backtest_stats.deflated_sharpe_ratio(
        0.0, [-1e200, 1e200], 3, 0.0, 3.0, False, True
    )
    assert math.isfinite(result)
    assert result == pytest.approx(5.197553442805939e199, rel=1e-12)


def test_raw_native_psr_preserves_extreme_finite_limit():
    result = openquant.backtest_stats.probabilistic_sharpe_ratio(1e200, 0.0, 3, 0.0, 3.0)
    assert result == pytest.approx(0.9772498680518208, abs=2e-12)

"""Integer-scalar compatibility for the evaluation entry points (issue #278)."""

import math
from statistics import NormalDist

import numpy as np
import pytest
from openquant import evaluation as ev

RETURNS = [-0.03, 0.01, 0.02, -0.01, 0.04, 0.02]


@pytest.mark.parametrize("integer", [np.int8, np.int16, np.int32, np.int64, np.uint64])
def test_numpy_trial_counts_match_the_independent_formula(integer):
    # AFML §14.7 expected maximum, evaluated using the standard-library normal.
    gamma = 0.5772156649015329
    normal = NormalDist()
    benchmark = 0.05 * (
        (1 - gamma) * normal.inv_cdf(1 - 1 / 100) + gamma * normal.inv_cdf(1 - 1 / (100 * math.e))
    )
    assert ev.expected_max_sharpe(integer(100), 0.05) == pytest.approx(benchmark, abs=1e-9)
    assert ev.deflated_sharpe_ratio(
        RETURNS, n_trials=integer(100), sharpe_std=0.05
    ) == ev.deflated_sharpe_ratio(RETURNS, n_trials=100, sharpe_std=0.05)
    assert ev.efficiency_stats(
        RETURNS, n_trials=integer(100), sharpe_std=0.05
    ) == ev.efficiency_stats(RETURNS, n_trials=100, sharpe_std=0.05)


@pytest.mark.parametrize("value", [np.int64(0), np.int64(1), np.int64(-1), np.uint64(1)])
def test_numpy_trial_counts_keep_the_lower_bound(value):
    with pytest.raises(ValueError, match="n_trials must be at least 2"):
        ev.expected_max_sharpe(value, 0.05)


@pytest.mark.parametrize("value", [True, False, np.bool_(True), 2.0, np.float64(2), "2", None])
def test_trial_counts_reject_noninteger_scalars(value):
    with pytest.raises(TypeError, match="n_trials must be an int"):
        ev.expected_max_sharpe(value, 0.05)

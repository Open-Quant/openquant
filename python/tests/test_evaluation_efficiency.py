"""`efficiency_stats`, `meta_label_metrics` log loss, `strategy_risk_sensitivity`, and the
non-Gaussian and trial-count behaviour of PSR and DSR (beads OQ-nbr.7).

Expected values come from the formulas (Bailey and López de Prado 2012, 2014; AFML §14.7,
§15.3) evaluated with the standard library, not from openquant.
"""

import math
import random
from statistics import NormalDist

import pytest
from openquant import evaluation as ev
from openquant import strategy_risk

Z = NormalDist()


def _psr_with_moments(sr, n, skew, kurt, benchmark=0.0):
    return Z.cdf(
        (sr - benchmark) * math.sqrt(n - 1) / math.sqrt(1 - skew * sr + (kurt - 1) / 4 * sr**2)
    )


def _series(seed=3, n=500, mu=0.0008, sd=0.01):
    rng = random.Random(seed)
    return [rng.gauss(mu, sd) for _ in range(n)]


# --- PSR and DSR: non-Gaussian returns and trial-count deflation -------------------------


def test_negative_skew_and_fat_tails_lower_the_psr():
    # Mostly small gains and rare large losses: positive Sharpe, negative skew, fat tails.
    returns = [0.002] * 190 + [-0.025] * 10
    m = ev.return_moments(returns)
    assert m.sharpe > 0 and m.skewness < -2 and m.kurtosis > 6
    psr = ev.probabilistic_sharpe_ratio(returns)
    assert psr == pytest.approx(
        _psr_with_moments(m.sharpe, m.n_obs, m.skewness, m.kurtosis), abs=1e-9
    )
    # The same Sharpe ratio and length under normal moments would be more convincing.
    assert psr < _psr_with_moments(m.sharpe, m.n_obs, 0.0, 3.0)


def test_positive_skew_raises_the_psr():
    # The mirror image of a short-volatility profile: rare large gains, frequent small losses.
    returns = [-0.001] * 190 + [0.03] * 10
    m = ev.return_moments(returns)
    assert m.sharpe > 0 and m.skewness > 2
    assert ev.probabilistic_sharpe_ratio(returns) > _psr_with_moments(m.sharpe, m.n_obs, 0.0, 3.0)


def test_symmetric_fat_tails_lower_the_psr():
    rng = random.Random(11)
    # Student-t-like tails from a scale mixture of normals; mean added after.
    fat = [rng.gauss(0.0, 0.03 if rng.random() < 0.1 else 0.007) + 0.001 for _ in range(2000)]
    m = ev.return_moments(fat)
    assert m.kurtosis > 5 and abs(m.skewness) < 0.5
    gaussian = _psr_with_moments(m.sharpe, m.n_obs, m.skewness, 3.0)
    assert ev.probabilistic_sharpe_ratio(fat) < gaussian


def test_dsr_falls_as_the_trial_count_grows():
    returns = _series()
    psr0 = ev.probabilistic_sharpe_ratio(returns)
    dsrs = [
        ev.deflated_sharpe_ratio(returns, n_trials=n, sharpe_std=0.03)
        for n in (2, 5, 10, 100, 1000, 10_000)
    ]
    sr0s = [ev.expected_max_sharpe(n, 0.03) for n in (2, 5, 10, 100, 1000, 10_000)]
    assert all(a > b for a, b in zip(dsrs, dsrs[1:]))
    assert all(a < b for a, b in zip(sr0s, sr0s[1:]))
    assert dsrs[0] < psr0
    # Each DSR is the PSR at that trial count's expected maximum Sharpe ratio.
    for dsr, sr0 in zip(dsrs, sr0s):
        assert dsr == pytest.approx(ev.probabilistic_sharpe_ratio(returns, sr0), abs=1e-12)


def test_dsr_falls_as_the_trial_dispersion_grows():
    returns = _series()
    dsrs = [
        ev.deflated_sharpe_ratio(returns, n_trials=50, sharpe_std=s) for s in (0.01, 0.02, 0.05)
    ]
    assert dsrs[0] > dsrs[1] > dsrs[2]


def test_dsr_without_dispersion_is_the_psr():
    returns = _series()
    assert ev.deflated_sharpe_ratio(returns, n_trials=1000, sharpe_std=0.0) == pytest.approx(
        ev.probabilistic_sharpe_ratio(returns), abs=1e-15
    )


# --- efficiency_stats ---------------------------------------------------------------------


def test_efficiency_stats_matches_the_individual_functions():
    returns = _series()
    trials = [0.01, -0.02, 0.04, 0.0, 0.03]
    out = ev.efficiency_stats(returns, periods_per_year=252, trial_sharpes=trials)
    m = ev.return_moments(returns)
    assert out["n_obs"] == m.n_obs and out["sharpe"] == m.sharpe
    assert (out["skewness"], out["kurtosis"]) == (m.skewness, m.kurtosis)
    assert out["annualized_sharpe"] == pytest.approx(m.sharpe * math.sqrt(252), rel=1e-15)
    assert out["psr"] == ev.probabilistic_sharpe_ratio(returns)
    assert out["dsr"] == ev.deflated_sharpe_ratio(returns, trials)
    pop_sd = math.sqrt(sum((t - sum(trials) / 5) ** 2 for t in trials) / 5)
    assert out["expected_max_sharpe"] == pytest.approx(ev.expected_max_sharpe(5, pop_sd), abs=1e-15)
    # With a zero benchmark the information ratio is the annualised Sharpe ratio.
    assert out["information_ratio"] == pytest.approx(out["annualized_sharpe"], rel=1e-12)


def test_efficiency_stats_information_ratio():
    returns = _series()
    bench = _series(seed=4, mu=0.0003)
    excess = [r - b for r, b in zip(returns, bench)]
    mu = sum(excess) / len(excess)
    sd = math.sqrt(sum((x - mu) ** 2 for x in excess) / (len(excess) - 1))
    out = ev.efficiency_stats(returns, periods_per_year=12, benchmark_returns=bench)
    assert out["information_ratio"] == pytest.approx(mu / sd * math.sqrt(12), rel=1e-12)
    # A constant benchmark agrees with the native information ratio.
    const = ev.efficiency_stats(returns, benchmark_returns=0.0002)
    assert const["information_ratio"] == pytest.approx(
        ev._bs.information_ratio(returns, 0.0002, 252.0), rel=1e-12
    )
    # Tracking the benchmark exactly leaves no tracking error: undefined, reported as NaN.
    # Dyadic values, so every difference is exactly 0.125.
    tracked = ev.efficiency_stats(
        [0.25, 0.5, 0.75, 1.0], benchmark_returns=[0.125, 0.375, 0.625, 0.875]
    )
    assert math.isnan(tracked["information_ratio"])


def test_efficiency_stats_without_trials_and_validation():
    returns = _series()
    out = ev.efficiency_stats(returns, n_trials=20, sharpe_std=0.02)
    assert out["dsr"] == ev.deflated_sharpe_ratio(returns, n_trials=20, sharpe_std=0.02)
    assert out["expected_max_sharpe"] == ev.expected_max_sharpe(20, 0.02)
    bare = ev.efficiency_stats(returns)
    assert bare["dsr"] is None and bare["expected_max_sharpe"] is None
    with pytest.raises(ValueError, match="periods_per_year"):
        ev.efficiency_stats(returns, periods_per_year=0)
    with pytest.raises(ValueError, match="differ in length"):
        ev.efficiency_stats(returns, benchmark_returns=[0.0] * 3)
    with pytest.raises(ValueError, match="not both"):
        ev.efficiency_stats(returns, trial_sharpes=[0.1, 0.2], n_trials=2, sharpe_std=0.1)
    with pytest.raises(ValueError, match="both n_trials and sharpe_std"):
        ev.efficiency_stats(returns, n_trials=20)


# --- meta_label_metrics log loss -----------------------------------------------------------


def test_meta_label_log_loss():
    labels = [1, 0, 1, 0]
    probs = [0.8, 0.3, 0.6, 0.1]
    hand = -(math.log(0.8) + math.log(0.7) + math.log(0.6) + math.log(0.9)) / 4
    assert ev.meta_label_metrics(labels, probs)["log_loss"] == pytest.approx(hand, rel=1e-15)
    # Independent of the decision threshold.
    assert ev.meta_label_metrics(labels, probs, threshold=0.9)["log_loss"] == pytest.approx(hand)
    # Hard decisions are clipped: one mistake in four costs -log(1e-15) / 4.
    hard = ev.meta_label_metrics(labels, [1, 0, 0, 0])["log_loss"]
    assert hard == pytest.approx(-math.log(1e-15) / 4, rel=1e-9)
    assert ev.meta_label_metrics(labels, [1, 0, 1, 0])["log_loss"] < 1e-14


# --- strategy_risk_sensitivity -------------------------------------------------------------


@pytest.mark.parametrize(
    ("p", "n", "up", "down"),
    [(0.70, 260.0, 0.005, -0.01), (0.55, 52.0, 0.02, -0.02), (0.40, 1000.0, 0.03, -0.01)],
)
def test_strategy_risk_sensitivity_matches_finite_differences(p, n, up, down):
    out = ev.strategy_risk_sensitivity(p, n, up, down)
    sr = strategy_risk.sharpe_asymmetric
    assert out["sharpe"] == sr(p, n, up, down)

    def central(f, x, h):
        return (f(x + h) - f(x - h)) / (2 * h)

    assert out["d_sharpe_d_precision"] == pytest.approx(
        central(lambda x: sr(x, n, up, down), p, 1e-6), rel=1e-6
    )
    assert out["d_sharpe_d_frequency"] == pytest.approx(
        central(lambda x: sr(p, x, up, down), n, 1e-3), rel=1e-6
    )
    assert out["d_sharpe_d_pi_plus"] == pytest.approx(
        central(lambda x: sr(p, n, x, down), up, 1e-7), rel=1e-6
    )
    assert out["d_sharpe_d_pi_minus"] == pytest.approx(
        central(lambda x: sr(p, n, up, x), down, 1e-7), rel=1e-6
    )
    assert out["implied_precision"] is None and out["precision_margin"] is None


def test_strategy_risk_sensitivity_precision_margin():
    out = ev.strategy_risk_sensitivity(0.70, 260.0, 0.005, -0.01, target_sharpe=1.0)
    implied = strategy_risk.implied_precision_asymmetric(1.0, 260.0, 0.005, -0.01)
    assert out["implied_precision"] == implied
    assert out["precision_margin"] == pytest.approx(0.70 - implied)
    # At the implied precision the strategy reaches the target exactly.
    assert strategy_risk.sharpe_asymmetric(implied, 260.0, 0.005, -0.01) == pytest.approx(
        1.0, abs=1e-9
    )


def test_strategy_risk_sensitivity_validates_through_strategy_risk():
    with pytest.raises(ValueError):
        ev.strategy_risk_sensitivity(1.0, 260.0, 0.005, -0.01)
    with pytest.raises(ValueError):
        ev.strategy_risk_sensitivity(0.6, 260.0, -0.01, 0.005)

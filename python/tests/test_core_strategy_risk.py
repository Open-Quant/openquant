import math

import pytest
from openquant import strategy_risk


def test_symmetric_sharpe_closed_form():
    # AFML 15.2: SR = (2p - 1) / (2 sqrt(p (1 - p))) * sqrt(n)
    p, n = 0.55, 260.0
    expected = (2 * p - 1) / (2 * math.sqrt(p * (1 - p))) * math.sqrt(n)
    assert strategy_risk.sharpe_symmetric(p, n) == pytest.approx(expected, abs=1e-12)


def test_symmetric_inverse_consistency():
    # Mirrors crates/openquant/tests/strategy_risk.rs::test_symmetric_inverse_consistency
    target, n = 2.0, 260.0
    p = strategy_risk.implied_precision_symmetric(target, n)

    assert abs(strategy_risk.sharpe_symmetric(p, n) - target) < 1e-8
    assert abs(strategy_risk.implied_frequency_symmetric(p, target) - n) < 1e-8
    assert p > 0.5


def test_asymmetric_inverse_consistency():
    # Mirrors crates/openquant/tests/strategy_risk.rs::test_asymmetric_inverse_consistency
    pi_plus, pi_minus = 0.005, -0.01
    target, n = 1.5, 260.0
    p = strategy_risk.implied_precision_asymmetric(target, n, pi_plus, pi_minus)

    assert abs(strategy_risk.sharpe_asymmetric(p, n, pi_plus, pi_minus) - target) < 1e-7
    assert abs(strategy_risk.implied_frequency_asymmetric(p, target, pi_plus, pi_minus) - n) < 1e-6
    assert 0.5 <= p < 1.0


def test_sensitivity_to_small_parameter_changes():
    # Mirrors crates/openquant/tests/strategy_risk.rs::
    # test_sensitivity_to_small_parameter_changes
    base = strategy_risk.sharpe_asymmetric(0.70, 260.0, 0.005, -0.01)
    assert strategy_risk.sharpe_asymmetric(0.71, 260.0, 0.005, -0.01) > base
    assert strategy_risk.sharpe_asymmetric(0.70, 240.0, 0.005, -0.01) < base
    assert strategy_risk.sharpe_asymmetric(0.70, 260.0, 0.005, -0.011) < base


def test_strategy_failure_probability_workflow():
    # Mirrors crates/openquant/tests/strategy_risk.rs::
    # test_strategy_failure_probability_workflow
    outcomes = [0.005 if i % 10 < 7 else -0.01 for i in range(1200)]
    report = strategy_risk.estimate_strategy_failure_probability(
        outcomes,
        years_elapsed=5.0,
        target_sharpe=2.0,
        investor_horizon_years=2.0,
        bootstrap_iterations=2000,
        seed=17,
    )

    assert report["annual_bet_frequency"] > 200.0
    assert 0.0 <= report["implied_precision_threshold"] <= 1.0
    assert 0.0 <= report["empirical_failure_probability"] <= 1.0
    assert 0.0 <= report["kde_failure_probability"] <= 1.0
    assert report["bootstrap_precision_std"] > 0.0
    assert len(report["bootstrap_precision_samples"]) == 2000

    # Deterministic parts of the report, derived from the inputs: 1200 bets over 5 years,
    # every win is +0.005 and every loss is -0.01, and 70% of bets win.
    assert report["annual_bet_frequency"] == pytest.approx(240.0)
    assert report["pi_plus"] == pytest.approx(0.005)
    assert report["pi_minus"] == pytest.approx(-0.01)
    assert report["bootstrap_precision_mean"] == pytest.approx(0.7, abs=0.01)
    threshold = strategy_risk.implied_precision_asymmetric(2.0, 240.0, 0.005, -0.01)
    assert report["implied_precision_threshold"] == pytest.approx(threshold, abs=1e-9)


def test_failure_probability_rises_with_higher_target_sharpe():
    # Mirrors crates/openquant/tests/strategy_risk.rs::
    # test_failure_probability_rises_with_higher_target_sharpe
    outcomes = [0.006 if i % 10 < 7 else -0.009 for i in range(1400)]
    low = strategy_risk.estimate_strategy_failure_probability(
        outcomes, 5.0, 1.0, 2.0, bootstrap_iterations=1200, seed=33
    )
    high = strategy_risk.estimate_strategy_failure_probability(
        outcomes, 5.0, 2.0, 2.0, bootstrap_iterations=1200, seed=33
    )

    assert high["implied_precision_threshold"] > low["implied_precision_threshold"]
    assert high["kde_failure_probability"] >= low["kde_failure_probability"]


def test_strategy_risk_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="precision"):
        strategy_risk.sharpe_symmetric(1.5, 260.0)
    with pytest.raises(ValueError, match="annual_bet_frequency"):
        strategy_risk.sharpe_symmetric(0.6, -1.0)
    with pytest.raises(ValueError, match="bet_outcomes"):
        strategy_risk.estimate_strategy_failure_probability([], 5.0, 2.0, 2.0)


def test_implied_frequency_rejects_an_unreachable_target():
    # Mirrors crates/openquant/tests/strategy_risk.rs::
    # test_implied_frequency_rejects_a_negative_mean_payoff (#168). A negative mean payoff
    # gives a negative Sharpe ratio at every frequency; the squared formula used to return
    # the frequency for the opposite edge instead.
    with pytest.raises(ValueError, match="no valid root"):
        strategy_risk.implied_frequency_symmetric(0.45, 2.0)
    with pytest.raises(ValueError, match="no valid root"):
        # mean payoff 0.03 * 0.6 - 0.02 = -0.002
        strategy_risk.implied_frequency_asymmetric(0.6, 2.0, 0.01, -0.02)
    # The mirror images still work.
    assert strategy_risk.implied_frequency_symmetric(0.55, 2.0) == pytest.approx(396.0)
    assert strategy_risk.implied_frequency_asymmetric(0.8, 2.0, 0.01, -0.02) > 0.0


def test_payouts_need_only_be_ordered():
    # The closed forms require pi_plus > pi_minus and nothing about signs (#168).
    assert strategy_risk.sharpe_asymmetric(0.5, 260.0, 0.02, 0.01) > 0.0
    with pytest.raises(ValueError, match="pi_plus must be greater than pi_minus"):
        strategy_risk.sharpe_asymmetric(0.5, 260.0, -0.01, 0.01)

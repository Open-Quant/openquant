"""PSR recovery against an independent exact-input influence-score reference."""

import math
import random
from decimal import ROUND_FLOOR, Context, Decimal, Inexact, localcontext
from fractions import Fraction
from statistics import NormalDist

import pytest
from openquant import evaluation as ev

# Scaling by a power of two preserves every input bit and the per-period ratios.
FIXTURE = [x / 256 for x in [-1.3962847939999437] + [-0.39628479399994365] * 4]
COLLISION_A = [x / 256 for x in [-1.3962847939999463] + [-0.39628479399994465] * 4]
COLLISION_B = [x / 256 for x in [-1.3962847939999463] + [-0.3962847939999445] * 4]


def _reference(values, benchmark, alpha=0.05):
    """Sum squared influence scores, rather than rationalizing a quadratic."""
    n = len(values)
    r = [Fraction(x) for x in values]
    mean = sum(r) / n
    dev = [x - mean for x in r]
    variance = sum(x * x for x in dev) / n
    with localcontext() as ctx:
        ctx.prec = 140

        def dec(value):
            return Decimal(value.numerator) / Decimal(value.denominator)

        sr = dec(mean) / (dec(variance) * n / (n - 1)).sqrt()
        standardized = [dec(x) / dec(variance).sqrt() for x in dev]
        uncertainty = sum((x - sr / 2 * (x * x - 1)) ** 2 for x in standardized) / n
        difference = sr - Decimal.from_float(benchmark)
        z = difference * Decimal(n - 1).sqrt() / uncertainty.sqrt()
        psr = math.erfc(-float(z) / math.sqrt(2)) / 2
        if difference <= 0:
            mintrl = math.inf
        else:
            quantile = Decimal.from_float(NormalDist().inv_cdf(alpha))
            mintrl = float(1 + uncertainty * (quantile / difference) ** 2)
        return psr, mintrl


@pytest.mark.parametrize("scale", [1.0, 2.0**-900, 2.0**900])
@pytest.mark.parametrize("direction", [-1, 0, 1])
def test_psr_recovers_cancelled_uncertainty(scale, direction):
    values = [x * scale for x in FIXTURE]
    benchmark = ev.return_moments(values).sharpe
    if direction:
        benchmark = math.nextafter(benchmark, math.copysign(math.inf, direction))
    expected, _ = _reference(values, benchmark)
    assert ev.probabilistic_sharpe_ratio(iter(values), benchmark) == pytest.approx(
        expected, abs=2e-14
    )


def test_original_returns_distinguish_colliding_rounded_moments():
    a = ev.return_moments(COLLISION_A)
    b = ev.return_moments(COLLISION_B)
    assert (a.n_obs, a.sharpe, a.skewness, a.kurtosis) == (
        b.n_obs,
        b.sharpe,
        b.skewness,
        b.kurtosis,
    )
    actual_a = ev.probabilistic_sharpe_ratio(COLLISION_A, a.sharpe)
    actual_b = ev.probabilistic_sharpe_ratio(COLLISION_B, a.sharpe)
    assert actual_a == pytest.approx(0.00005016482622714143, abs=2e-14)
    assert actual_b == pytest.approx(0.9828377556984284, abs=2e-14)


@pytest.mark.parametrize("direction", [-1, 0, 1])
def test_mintrl_uses_unrounded_sharpe_to_compare_benchmark(direction):
    values = [-x for x in FIXTURE]
    benchmark = ev.return_moments(values).sharpe
    if direction:
        benchmark = math.nextafter(benchmark, math.copysign(math.inf, direction))
    _, expected = _reference(values, benchmark)
    assert ev.minimum_track_record_length(iter(values), benchmark) == pytest.approx(
        expected, rel=2e-13
    )
    if direction == 0:
        # The rounded Sharpe equals the benchmark; the exact-input Sharpe is higher.
        assert math.isfinite(expected)


@pytest.mark.parametrize("alpha", [math.ulp(0.0), 1e-20, 0.05, 0.9])
def test_mintrl_exceptional_quantile_avoids_one_minus_alpha_rounding(alpha):
    values = [-x for x in FIXTURE]
    benchmark = ev.return_moments(values).sharpe
    _, expected = _reference(values, benchmark, alpha)
    assert ev.minimum_track_record_length(values, benchmark, alpha) == pytest.approx(
        expected, rel=2e-13
    )


def test_deflated_sharpe_uses_return_evidence_after_computing_benchmark():
    values = [-x for x in FIXTURE]
    sd = ev.return_moments(values).sharpe / ev.expected_max_sharpe(10, 1.0)
    benchmark = ev.expected_max_sharpe(10, sd)
    expected, _ = _reference(values, benchmark)
    assert ev.deflated_sharpe_ratio(iter(values), n_trials=10, sharpe_std=sd) == pytest.approx(
        expected, abs=2e-14
    )


def test_ordinary_uncertainty_keeps_native_path(monkeypatch):
    def unexpected(*args):
        pytest.fail("ordinary returns should not invoke exact recovery")

    monkeypatch.setattr(ev, "_precise_sharpe_difference_and_variance", unexpected)
    values = [-0.02, 0.01, 0.03, -0.01, 0.005]
    m = ev.return_moments(values)
    expected = ev._bs.probabilistic_sharpe_ratio(m.sharpe, 0.0, m.n_obs, m.skewness, m.kurtosis)
    assert ev.probabilistic_sharpe_ratio(values) == expected
    assert math.isfinite(ev.minimum_track_record_length(values))


def test_recovery_is_independent_of_the_callers_decimal_context():
    benchmark = ev.return_moments(FIXTURE).sharpe
    expected = ev.probabilistic_sharpe_ratio(FIXTURE, benchmark)
    with localcontext() as ctx:
        ctx.prec = 3
        ctx.rounding = ROUND_FLOOR
        ctx.traps[Inexact] = True
        assert ev.probabilistic_sharpe_ratio(FIXTURE, benchmark) == expected


@pytest.mark.parametrize("n", [4, 6, 10, 100])
def test_near_unit_kurtosis_does_not_hide_large_sharpe_uncertainty(n):
    values = [1.0] * (n // 2) + [math.nextafter(1.0, math.inf)] * (n // 2)
    m = ev.return_moments(values)
    assert m.skewness == 0.0
    assert m.kurtosis == 1.0
    expected, _ = _reference(values, m.sharpe)
    assert ev.probabilistic_sharpe_ratio(values, m.sharpe) == pytest.approx(expected, abs=2e-14)


def test_exact_recovery_with_heterogeneous_float_denominators():
    rng = random.Random(1931)
    for _ in range(40):
        values = [
            math.ldexp(rng.uniform(-1.0, 1.0), rng.randrange(-1074, 1024))
            for _ in range(rng.randrange(3, 25))
        ]
        benchmark = ev.return_moments(values).sharpe + rng.uniform(-0.4, 0.4)
        expected, _ = _reference(values, benchmark)
        with localcontext(Context(prec=80)):
            difference, variance = ev._precise_sharpe_difference_and_variance(values, benchmark)
            z = difference * Decimal(len(values) - 1).sqrt() / variance.sqrt()
            actual = math.erfc(-float(z) / math.sqrt(2)) / 2
        assert actual == pytest.approx(expected, abs=2e-14)

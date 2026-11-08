"""Exact-input Decimal references for cancellation and range regressions."""

import math
import random
from decimal import Decimal, localcontext

import pytest
from openquant import evaluation as ev


def _reference(values):
    with localcontext() as ctx:
        ctx.prec = 1100
        r = [Decimal.from_float(x) for x in values]
        n = Decimal(len(r))
        mean = sum(r) / n
        dev = [x - mean for x in r]
        m2, m3, m4 = [sum(x**p for x in dev) / n for p in (2, 3, 4)]
        std = (m2 * n / (n - 1)).sqrt()
        return [float(x) for x in (mean, std, mean / std, m3 / m2 ** Decimal("1.5"), m4 / m2**2)]


@pytest.mark.parametrize(
    "values",
    [
        [1.0, 1.0, math.nextafter(1.0, math.inf)],
        [1.0, math.nextafter(1.0, math.inf), math.nextafter(1.0, math.inf)],
        [1e200, 2e200, 3e200],
        [1e-200, 2e-200, 3e-200],
        [-1e308, 0.0, 1e308],
        [
            -float.fromhex("0x1.fffffffffffffp+1023"),
            float.fromhex("0x1.fffffffffffffp+1023"),
            float.fromhex("0x1.fffffffffffffp+1023"),
        ],
        [1e308, math.nextafter(1e308, math.inf), 1e308],
        [math.ulp(0.0), 2 * math.ulp(0.0), 3 * math.ulp(0.0)],
        [math.ulp(0.0), math.ulp(0.0), 2 * math.ulp(0.0)],
        [0.0, 0.0, math.ulp(0.0)],
        [0.0, 0.0, 0.0, math.ulp(0.0)],
    ],
)
def test_return_moments_exact_input_reference(values):
    expected = _reference(values)
    actual = ev.return_moments(values)
    for name, want in zip(("mean", "std", "sharpe", "skewness", "kurtosis"), expected, strict=True):
        assert getattr(actual, name) == pytest.approx(
            want, rel=3e-14, abs=2e-14 if name in ("skewness", "kurtosis") else 0.0
        ), name
    assert actual.kurtosis >= actual.skewness**2 + 1 - 1e-14


@pytest.mark.parametrize(
    "values",
    [
        [-1e200, 0.0, 1e200],
        [1e-200, 2e-200, 3e-200],
        [1.0, 1.0, math.nextafter(1.0, math.inf)],
    ],
)
def test_trial_dispersion_exact_input_reference(values):
    sample_std = _reference(values)[1]
    expected = sample_std * math.sqrt((len(values) - 1) / len(values))
    assert ev._population_std(values) == pytest.approx(expected, rel=3e-14, abs=0.0)


def test_return_moments_are_permutation_invariant():
    values = [1.0, 1.0, math.nextafter(1.0, math.inf)]
    assert ev.return_moments(values) == ev.return_moments(values[::-1])


def test_random_adjacent_float_distributions_against_decimal():
    rng = random.Random(1127)
    for _ in range(40):
        n = rng.randrange(3, 40)
        base = math.ldexp(rng.uniform(0.5, 1.0), rng.randrange(-1000, 1000))
        values = [base + math.ulp(base) * rng.randrange(-8, 9) for _ in range(n)]
        actual = ev.return_moments(values)
        expected = _reference(values)
        for name, want in zip(
            ("mean", "std", "sharpe", "skewness", "kurtosis"), expected, strict=True
        ):
            tolerance = 2e-14 if name in ("skewness", "kurtosis") else 0.0
            assert getattr(actual, name) == pytest.approx(want, rel=2e-13, abs=tolerance), name


@pytest.mark.parametrize("residual", [1e-14, 1e-13, 1e-12])
def test_sharpe_preserves_tiny_cancelling_residual(residual):
    values = [1e308, -1e308, residual] + [0.0] * 97
    actual = ev.return_moments(values)
    expected = _reference(values)
    assert actual.sharpe == expected[2]


def test_mean_overflow_recovery_preserves_subnormal_residual():
    largest = float.fromhex("0x1.fffffffffffffp+1023")
    tiny = math.ulp(0.0)
    values = [largest, largest, -largest, -largest] + [tiny] * 100
    permuted = [largest, -largest, largest, -largest] + [tiny] * 100
    assert ev.return_moments(values).mean == tiny
    assert ev.return_moments(permuted).mean == tiny

"""Independent path enumeration for sequential binary-sign experiments."""

import importlib.util
import itertools
import math
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    "sequential_sign_lab",
    Path(__file__).resolve().parents[2] / "experiments/sequential_sign_lab.py",
)
assert spec is not None and spec.loader is not None
lab = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lab)


@pytest.mark.parametrize(
    "transition",
    [(0.5, 0.5, 0.5), (0.3, 0.5, 0.375), (0.1, 0.9, 0.5), (0.65, 0.65, 0.65), (0.0, 1.0, 0.5)],
)
def test_forward_probability_exhaustive_paths(transition):
    horizon = 9
    boundaries = lab.crossing_boundaries(horizon, alpha=0.2)
    p0, p1, initial = transition
    for method, boundary in boundaries.items():
        expected = np.zeros(horizon)
        for path in itertools.product((0, 1), repeat=horizon):
            weight = initial if path[0] else 1 - initial
            for previous, current in zip(path, path[1:]):
                p = p1 if previous else p0
                weight *= p if current else 1 - p
            successes = 0
            for t, x in enumerate(path, 1):
                successes += x
                if successes >= boundary[t]:
                    expected[t - 1] += weight
                    break
        actual = lab.crossing_probability(
            boundary, p_after_zero=p0, p_after_one=p1, initial_positive=initial
        )
        np.testing.assert_allclose(actual["first_crossing_probabilities"], expected, atol=2e-15)
        assert actual["crossing_probability"] + actual["survival_probability"] == pytest.approx(1.0)


def test_mixture_exact_products_and_conditional_supermartingale():
    qs = (0.55, 0.65, 0.8)
    for t in range(10):
        for s in range(t + 1):
            capital = sum((2 * q) ** s * (2 * (1 - q)) ** (t - s) for q in qs) / len(qs)
            assert math.exp(lab.log_mixture_capital(t, s, qs)) == pytest.approx(capital)
            for conditional in [0.0, 0.3, 0.5]:
                expected = conditional * math.exp(lab.log_mixture_capital(t + 1, s + 1, qs)) + (
                    1 - conditional
                ) * math.exp(lab.log_mixture_capital(t + 1, s, qs))
                assert expected <= capital + 1e-12


def test_binomial_boundaries_exact_integer_oracle():
    alpha = 0.05
    num, den = alpha.as_integer_ratio()
    boundaries = lab.crossing_boundaries(64, alpha=alpha)["repeated_fixed_test"]
    for t in range(1, 65):
        for s in range(t + 1):
            tail = sum(math.comb(t, k) for k in range(s, t + 1))
            assert (s >= boundaries[t]) == (tail * den <= num * 2**t)


def test_optional_stopping_inflation_and_invalid_anticipation():
    bounds = lab.crossing_boundaries(256)
    probabilities = {
        m: lab.crossing_probability(b)["crossing_probability"] for m, b in bounds.items()
    }
    assert probabilities["repeated_fixed_test"] > 0.1
    assert probabilities["fixed_mixture"] <= 0.05
    assert probabilities["fixed_bet"] <= 0.05
    assert probabilities["anticipatory_bet"] > 0.99


def test_extreme_alpha_log_domain():
    bounds = lab.crossing_boundaries(16, alpha=np.nextafter(0.0, 1.0))
    for b in bounds.values():
        assert lab.crossing_probability(b)["crossing_probability"] == 0
    assert np.isfinite(lab.log_mixture_capital(100000, 100000))


def test_path_replay_first_crossing():
    bounds = lab.crossing_boundaries(8, alpha=0.2)
    paths = np.array(list(itertools.product((0, 1), repeat=8)), dtype=np.uint8)
    first = lab.evaluate_paths(paths, bounds)
    for method in lab.METHODS:
        exact = lab.crossing_probability(bounds[method])["crossing_probability"]
        assert np.mean(first[method] > 0) == exact
    report = lab.run_lab(horizon=16, repetitions=5)
    repeat = lab.run_lab(**report["config"])
    assert repeat["families"] == report["families"]
    assert not report["families"]["persistent_marginally_fair"]["conditional_null"]
    assert report["families"]["dependent_conditional_null"]["conditional_null"]


@pytest.mark.parametrize("qs", [[], [0.5], [1], [0], [float("nan")], [True], [0.4, 0.6]])
def test_invalid_mixture(qs):
    with pytest.raises((ValueError, TypeError)):
        lab.crossing_boundaries(3, qs=qs)


@pytest.mark.parametrize("paths", [[[2]], [[np.nan]], [[1.0]], np.empty((1, 0)), [1, 0]])
def test_invalid_binary_paths(paths):
    with pytest.raises(ValueError):
        lab.evaluate_paths(paths, lab.crossing_boundaries(1))


def test_unsupported_horizons_and_count_ranges():
    with pytest.raises(ValueError):
        lab.crossing_boundaries(10**400)
    with pytest.raises(ValueError):
        lab.log_mixture_capital(10**400, 0)
    with pytest.raises(ValueError):
        lab.crossing_probability(np.ones(lab.MAX_HORIZON + 2, dtype=int))


def test_borderline_fixed_bet_exact_rational_decisions():
    from fractions import Fraction

    for horizon, alpha in [(9, 0.09429959537232739), (3, 0.8453085376162299)]:
        bounds = lab.crossing_boundaries(horizon, alpha=alpha)
        q = Fraction.from_float(0.65)
        for t in range(1, horizon + 1):
            for s in range(t + 1):
                crosses = (2 * q) ** s * (2 * (1 - q)) ** (t - s) * Fraction.from_float(alpha) >= 1
                assert (s >= bounds["fixed_bet"][t]) == crosses


def test_mixture_boundary_fraction_oracle():
    from fractions import Fraction

    qs = (0.55, 0.65, 0.8)
    alpha = 0.05
    boundaries = lab.crossing_boundaries(16, alpha=alpha, qs=qs)
    for t in range(1, 17):
        for s in range(t + 1):
            capital = (
                sum((2 * Fraction(q)) ** s * (2 * (1 - Fraction(q))) ** (t - s) for q in qs) / 3
            )
            assert (s >= boundaries["fixed_mixture"][t]) == (capital * Fraction(alpha) >= 1)


def test_path_boundary_validation():
    bounds = lab.crossing_boundaries(2)
    for bad in [np.array([1.0, 1.0, 1.0]), np.array([1, -1, 2]), np.array([1, 3, 3])]:
        with pytest.raises(ValueError):
            lab.evaluate_paths([[1, 0]], dict(bounds, fixed_bet=bad))

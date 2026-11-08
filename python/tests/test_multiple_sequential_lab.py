"""Joint time/candidate multiplicity checks, including exact tiny paths."""

import importlib.util
import itertools
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    "multiple_sequential_lab",
    Path(__file__).resolve().parents[2] / "experiments/multiple_sequential_lab.py",
)
assert spec is not None and spec.loader is not None
lab = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lab)


def test_exact_global_table_fraction_reference():
    qs = (0.55, 0.65, 0.8)
    rows, exponent, components = lab.capital_table(8, qs)
    for t in range(9):
        for s in range(t + 1):
            expected = sum(
                (2 * Fraction(q)) ** s * (2 * (1 - Fraction(q))) ** (t - s) for q in qs
            ) / len(qs)
            actual = Fraction(rows[t][s], components * 2 ** (exponent * t))
            assert actual == expected


def test_all_tiny_joint_paths_independent_fraction_crossing():
    qs = (0.55, 0.65, 0.8)
    alpha = 0.4
    paths = np.array(list(itertools.product((0, 1), repeat=8)), dtype=np.uint8).reshape(-1, 4, 2)
    actual = lab.evaluate_paths(paths, alpha=alpha, qs=qs)
    for r, path in enumerate(paths):
        first = 0
        count = [0, 0]
        for t, row in enumerate(path, 1):
            count = [s + int(x) for s, x in zip(count, row)]
            capital = (
                sum(
                    sum((2 * Fraction(q)) ** s * (2 * (1 - Fraction(q))) ** (t - s) for q in qs) / 3
                    for s in count
                )
                / 2
            )
            if capital * Fraction(alpha) >= 1:
                first = t
                break
        assert actual["global_first"][r] == first


def test_k1_and_fully_duplicated_global_identity():
    rng = np.random.default_rng(7)
    one = rng.integers(0, 2, size=(100, 32, 1), dtype=np.uint8)
    singleton = lab.evaluate_paths(one, alpha=0.2)
    np.testing.assert_array_equal(singleton["global_first"], singleton["uncorrected_first"][:, 0])
    np.testing.assert_array_equal(singleton["global_first"], singleton["bonferroni_first"][:, 0])
    duplicate = lab.evaluate_paths(np.repeat(one, 8, axis=2), alpha=0.2)
    np.testing.assert_array_equal(duplicate["global_first"], singleton["global_first"])
    assert np.all(
        (duplicate["bonferroni_first"] == 0)
        | (duplicate["bonferroni_first"] >= singleton["bonferroni_first"])
    )


@pytest.mark.parametrize(
    "alpha", [0.05, np.nextafter(0.0, 1.0), np.nextafter(0.0, 1.0) * 3, 1e-100]
)
@pytest.mark.parametrize("k", [1, 3, 32])
def test_exact_candidate_allocation(alpha, k):
    budget = lab.candidate_budget(alpha, k)
    assert Fraction(budget) * k <= Fraction(float(alpha))


def test_underflow_has_no_candidate_claims():
    result = lab.evaluate_paths(np.ones((1, 4, 3), dtype=np.uint8), alpha=np.nextafter(0.0, 1.0))
    assert result["candidate_budget_underflow"]
    assert not result["bonferroni_first"].any()
    assert not result["global_first"].any()


def test_independent_any_oracle_exhaustive():
    paths = np.array(list(itertools.product((0, 1), repeat=8)), dtype=np.uint8).reshape(-1, 4, 2)
    result = lab.evaluate_paths(paths, alpha=0.4)
    boundary = lab.sequential.crossing_boundaries(4, alpha=0.4)["fixed_mixture"]
    single = lab.sequential.crossing_probability(boundary)["crossing_probability"]
    assert np.mean(result["uncorrected_first"].any(axis=1)) == pytest.approx(
        lab.independent_any_crossing(single, 2)
    )


def test_report_replay_and_global_scope():
    report = lab.run_lab(horizon=8, repetitions=3, n_candidates=2)
    replay = lab.run_lab(**report["config"])
    assert report["families"] == replay["families"]
    for f in report["families"].values():
        assert "any_false_candidate_claim" not in f["methods"]["global_average"]
    assert report["families"]["correlated_null"]["independent_reference"] == {}


@pytest.mark.parametrize(
    "bad",
    [np.zeros((2, 4)), np.full((1, 2, 3), 2), np.empty((1, 0, 1)), np.ones((1, 2, 1), dtype=float)],
)
def test_invalid_path_arrays(bad):
    with pytest.raises(ValueError):
        lab.evaluate_paths(bad)

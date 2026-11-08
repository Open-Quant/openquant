import importlib.util
import math
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

s = importlib.util.spec_from_file_location(
    "mean_robustness", Path(__file__).resolve().parents[2] / "experiments/mean_robustness_lab.py"
)
assert s is not None and s.loader is not None
lab = importlib.util.module_from_spec(s)
s.loader.exec_module(lab)


@pytest.mark.parametrize("alpha", [0.5, 0.05, 0.001, 1e-12])
def test_exact_block_budget(alpha):
    b, num, den = lab.block_count(alpha)
    assert Fraction(num, den) <= Fraction(alpha)
    if b > 1:
        assert sum(
            Fraction(math.comb(b - 2, j) * 3 ** (b - 2 - j), 4 ** (b - 2))
            for j in range((b - 1) // 2, b - 1)
        ) > Fraction(alpha)


def test_unused_rows_do_not_enter_median():
    b, _, _ = lab.block_count(0.05)
    x = np.zeros(2 * b + 1)
    x[-1] = 1000
    r = lab.intervals(x)
    assert r["unused_rows"] == 1
    assert sum(r["median_of_means"]) == 0
    assert sum(r["chebyshev"]) > 0


def test_small_sample_is_explicitly_unavailable():
    assert lab.intervals([1.0, 2.0])["median_of_means"] is None


@pytest.mark.parametrize("x", [[1], [np.nan, 1], [1j, 2], [[1, 2]]])
def test_invalid_values(x):
    with pytest.raises(ValueError):
        lab.intervals(x)


@pytest.mark.parametrize("alpha", [0.0, 1.0, 1e-20, float("nan"), True])
def test_invalid_alpha(alpha):
    with pytest.raises((TypeError, ValueError)):
        lab.block_count(alpha)


def test_replay_and_clipping_target():
    r = lab.run_lab(repetitions=3, sample_size=16)
    assert r["families"] == lab.run_lab(**r["config"])["families"]
    skew = r["families"]["rare_skew"]
    assert skew["clipped_target"] < 0
    assert skew["true_mean"] == 0
    assert not r["families"]["wrong_variance"]["variance_bound_valid"]
    assert not r["families"]["repeated_rows"]["iid"]

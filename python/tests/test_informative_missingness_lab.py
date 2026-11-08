import importlib.util
import itertools
from pathlib import Path

import numpy as np
import pytest

s = importlib.util.spec_from_file_location(
    "missingness",
    Path(__file__).resolve().parents[2] / "experiments/informative_missingness_lab.py",
)
assert s is not None and s.loader is not None
lab = importlib.util.module_from_spec(s)
s.loader.exec_module(lab)


@pytest.mark.parametrize(
    "p,sp,sm", [(0.5, 0.5, 0.5), (0.5, 1, 0.25), (0.625, 0.8, 1 / 3), (0.5, 1, 0.01)]
)
def test_exhaustive_ht_moments(p, sp, sm):
    n = 3
    probs = [p * sp, (1 - p) * sm, 1 - p * sp - (1 - p) * sm]
    mean = second = 0
    for path in itertools.product(range(3), repeat=n):
        probability = np.prod([probs[c] for c in path])
        estimate = lab.estimate_counts(path.count(0), path.count(1), n, sp, sm)["ht"]
        mean += probability * estimate
        second += probability * estimate**2
    r = lab.reference(p, sp, sm, n)
    assert mean == pytest.approx(r["ht_expectation"])
    assert second - mean**2 == pytest.approx(r["ht_variance"])


def test_observational_equivalence_does_not_identify_mean():
    a = lab.reference(0.5, 1, 0.25, 32)
    b = lab.reference(0.625, 0.8, 1 / 3, 32)
    assert a["observable_probabilities"] == b["observable_probabilities"]
    assert a["true_mean"] == 0
    assert b["true_mean"] == 0.25
    assert a["observed_target"] == b["observed_target"] == 0.6


def test_no_observation_and_positivity():
    assert lab.estimate_counts(0, 0, 10, 0.5, 0.5) == {
        "observed_only": None,
        "ht": 0,
        "hajek": None,
    }
    assert lab.estimate_counts(2, 0, 10, 1, 0)["ht"] is None
    with pytest.raises(ValueError):
        lab.estimate_counts(0, 1, 10, 1, 0)


@pytest.mark.parametrize(
    "args", [(-1, 0, 3, 0.5, 0.5), (2, 2, 3, 0.5, 0.5), (True, 0, 3, 0.5, 0.5), (0, 0, 0, 0.5, 0.5)]
)
def test_invalid_counts(args):
    with pytest.raises((TypeError, ValueError)):
        lab.estimate_counts(*args)


def test_replay_flags():
    r = lab.run_lab(sample_sizes=[2], repetitions=3)
    assert r["records"] == lab.run_lab(**r["config"])["records"]
    assert r["records"][-1]["summary"]["observed_only"]["unavailable"] == 3


def test_unsigned_counts_and_tiny_propensities():
    assert (
        lab.estimate_counts(np.uint64(0), np.uint64(1), np.uint64(2), 0.5, 0.5)["observed_only"]
        == -1
    )
    with pytest.raises(ValueError):
        lab.reference(0.5, 1, 5e-324, 32)
    with pytest.raises(ValueError):
        lab.estimate_counts(0, 0, 32, 1, 5e-324)

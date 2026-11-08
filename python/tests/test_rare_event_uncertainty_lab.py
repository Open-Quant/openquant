import importlib.util
import itertools
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("scipy")
s = importlib.util.spec_from_file_location(
    "rare_event", Path(__file__).resolve().parents[2] / "experiments/rare_event_uncertainty_lab.py"
)
assert s is not None and s.loader is not None
lab = importlib.util.module_from_spec(s)
s.loader.exec_module(lab)


@pytest.mark.parametrize("p", [0.01, 0.25, 0.5])
def test_small_paths_independent_coverage(p):
    n = 4
    r = lab.evaluate(n, p)
    for method, bounds in r["intervals"].items():
        total = 0
        for path in itertools.product([0, 1], repeat=n):
            k = sum(path)
            weight = p**k * (1 - p) ** (n - k)
            if bounds[k][0] <= 1 <= bounds[k][1]:
                total += weight
        assert r["summaries"][method]["coverage"] == pytest.approx(total, abs=1e-14)


def test_missing_event_collapses_empirical_methods():
    r = lab.evaluate(128, 0.001)
    for method in [
        "plugin_chebyshev",
        "percentile_bootstrap",
        "gaussian_reference",
        "student_reference",
    ]:
        assert r["summaries"][method]["zero_event_interval"] == [0.0, 0.0]
        assert r["summaries"][method]["coverage"] <= 1 - r["no_event_probability"] + 1e-14
    assert r["summaries"]["known_support_cp"]["coverage"] >= 0.95


@pytest.mark.parametrize(
    "n,p,a",
    [
        (1, 0.1, 0.05),
        (10001, 0.1, 0.05),
        (8, 0, 0.05),
        (8, 1, 0.05),
        (8, 0.1, 1e-10),
        (True, 0.1, 0.05),
    ],
)
def test_scope(n, p, a):
    with pytest.raises((TypeError, ValueError)):
        lab.evaluate(n, p, alpha=a)


def test_bootstrap_quantiles_are_empirical_resampling_distribution():
    r = lab.evaluate(4, 0.25)
    vals = r["intervals"]["percentile_bootstrap"][1]
    # Four resampling draws from four observed values (one amplitude4, three0).
    means = sorted(sum(x) / 4 for x in itertools.product([4, 0, 0, 0], repeat=4))
    assert vals[0] == means[int(np.ceil(0.025 * 256)) - 1]
    assert vals[1] == means[int(np.ceil(0.975 * 256)) - 1]

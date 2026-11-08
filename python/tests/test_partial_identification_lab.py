import importlib.util
import math
from pathlib import Path

import pytest

pytest.importorskip("scipy")
s = importlib.util.spec_from_file_location(
    "partial_id", Path(__file__).resolve().parents[2] / "experiments/partial_identification_lab.py"
)
assert s is not None and s.loader is not None
lab = importlib.util.module_from_spec(s)
s.loader.exec_module(lab)


@pytest.mark.parametrize("a,b", [(0.5, 0.125), (0.25, 0.25), (0.0, 0.0), (0.5, 0.5)])
def test_sharp_endpoints_constructible(a, b):
    lower = lab.compatible_model(a, b, 0)
    upper = lab.compatible_model(a, b, 1 - a - b)
    assert lower["mean"] == lower["sharp_bounds"][0]
    assert upper["mean"] == pytest.approx(upper["sharp_bounds"][1])


@pytest.mark.parametrize("a,b", [(0.5, 0.125), (0.25, 0.25), (0.5, 0.5)])
def test_tiny_trinomial_whole_set_coverage(a, b):
    n = 8
    lo, hi = lab.compatible_model(a, b, 0)["sharp_bounds"]
    coverage = 0.0
    for kp in range(n + 1):
        for km in range(n - kp + 1):
            missing = n - kp - km
            mass = (
                math.factorial(n)
                / math.factorial(kp)
                / math.factorial(km)
                / math.factorial(missing)
                * a**kp
                * b**km
                * (1 - a - b) ** missing
            )
            ci = lab.outer_interval(kp, km, n)["outer_bounds"]
            if ci[0] <= lo and hi <= ci[1]:
                coverage += mass
    assert coverage >= 0.95 - 1e-14


def test_allmissing_and_fullobserved():
    assert lab.outer_interval(0, 0, 32)["outer_bounds"] == [-1, 1]
    assert lab.outer_interval(0, 0, 32)["observed_mean"] is None
    r = lab.outer_interval(16, 16, 32)
    assert r["plugin_bounds"] == [0.0, 0.0]
    assert r["outer_bounds"][0] < 0 < r["outer_bounds"][1]


def test_equivalence_paired_records():
    r = lab.run_lab(sample_sizes=[16], repetitions=4)
    a, b = r["reports"][:2]
    assert a["records"] == b["records"]
    assert a["counts_sha256"] == b["counts_sha256"]
    assert a["truth"]["mean"] != b["truth"]["mean"]
    assert r["reports"] == lab.run_lab(**r["config"])["reports"]


@pytest.mark.parametrize("args", [(0.8, 0.3, 0), (0.5, 0.25, 0.5), (-0.1, 0.1, 0), (True, 0, 0)])
def test_invalid_masses(args):
    with pytest.raises((ValueError, TypeError)):
        lab.compatible_model(*args)

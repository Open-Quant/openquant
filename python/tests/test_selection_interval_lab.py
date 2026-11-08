"""Optional research interval checks, independent of production APIs."""

import importlib.util
import math
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("scipy", reason="optional selection-interval research dependency")
from scipy.stats import binomtest

spec = importlib.util.spec_from_file_location(
    "selection_interval_lab",
    Path(__file__).resolve().parents[2] / "experiments/selection_interval_lab.py",
)
assert spec is not None and spec.loader is not None
lab = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lab)


@pytest.mark.parametrize("n", [1, 2, 10, 50, 100])
def test_reference_all_counts(n):
    for k in range(n + 1):
        actual = lab.clopper_pearson(k, n)
        expected = binomtest(k, n).proportion_ci(method="exact")
        assert actual["lower"] == pytest.approx(expected.low, abs=2e-12)
        assert actual["upper"] == pytest.approx(expected.high, abs=2e-12)


def test_reference_published():
    interval = lab.clopper_pearson(7, 50)
    assert interval["lower"] == pytest.approx(0.05819170033997342)
    assert interval["upper"] == pytest.approx(0.26739600249700846)


@pytest.mark.parametrize(
    "alpha", [0.05, 1e-100, np.nextafter(0.0, 1.0), np.nextafter(0.0, 1.0) * 3]
)
@pytest.mark.parametrize("family", [1, 3, 100000])
def test_exact_budget(alpha, family):
    tail = lab.allocated_tail(alpha, family)
    assert Fraction.from_float(tail) * 2 * family <= Fraction.from_float(float(alpha))


def test_empty_underflow_and_numerical_failure(monkeypatch):
    assert lab.clopper_pearson(0, 0)["status"] == "empty"
    assert lab.clopper_pearson(1, 2, alpha=np.nextafter(0.0, 1.0))["status"] == "tail_underflow"
    monkeypatch.setattr(lab, "betainccinv", lambda *args: np.nan)
    assert lab.clopper_pearson(1, 10)["status"] == "numerical_fallback"
    assert lab.clopper_pearson(1, 10)["upper"] == 1


def test_tiny_tail_nan_regression():
    interval = lab.clopper_pearson(1, 10, alpha=2 * np.nextafter(0.0, 1.0))
    assert np.isfinite([interval["lower"], interval["upper"]]).all()
    assert interval["lower"] <= interval["upper"]


@pytest.mark.parametrize(
    "bad", [True, 0, 1, -0.1, float("nan"), np.complex128(0.05 + 1j), [0.05], "0.05"]
)
def test_bad_alpha(bad):
    with pytest.raises((ValueError, TypeError)):
        lab.clopper_pearson(1, 2, alpha=bad)


def test_sign_counts_and_selector():
    wins, n = lab.sign_counts([[0, 1, -1], [0, -1, 1]])
    assert lab.select_candidate(wins, n) == 1
    assert lab.select_candidate([0, 0], [0, 0]) is None
    # exact cross multiplication, not a rounded floating comparison
    assert lab.select_candidate([2**54, 2**54 + 1], [2**55, 2**55]) == 1
    assert lab.sign_counts(np.empty((0, 2)))[1].tolist() == [0, 0]


@pytest.mark.parametrize("p", [0.01, 0.1, 0.5, 0.9, 0.99])
def test_independent_exact_coverage_enumeration(p):
    n = 12
    coverage = 0.0
    for k in range(n + 1):
        ci = lab.clopper_pearson(k, n)
        mass = math.comb(n, k) * p**k * (1 - p) ** (n - k)
        coverage += mass * (ci["lower"] <= p <= ci["upper"])
    assert coverage >= 0.95 - 1e-14
    assert lab.iid_max_coverage(n, 1, p=p)["coverage"] == pytest.approx(coverage)


def test_max_oracle_exact_integer_null():
    n, candidates = 10, 8
    denom = 2**n
    cumulative = 0
    coverage_num = 0
    for k in range(n + 1):
        previous = cumulative
        cumulative += math.comb(n, k)
        ci = lab.clopper_pearson(k, n)
        if ci["lower"] <= 0.5 <= ci["upper"]:
            coverage_num += cumulative**candidates - previous**candidates
    assert lab.iid_max_coverage(n, candidates)["coverage"] == pytest.approx(
        coverage_num / denom**candidates
    )
    assert lab.iid_max_coverage(64, 32)["coverage"] < 0.8
    assert lab.iid_max_coverage(64, 32, correction_size=32)["coverage"] >= 0.95


def test_symmetry_nesting_duplicates():
    for k in range(11):
        a = lab.clopper_pearson(k, 10)
        b = lab.clopper_pearson(10 - k, 10)
        wide = lab.clopper_pearson(k, 10, family_size=20)
        assert a["lower"] == pytest.approx(1 - b["upper"])
        assert wide["lower"] <= a["lower"] <= a["upper"] <= wide["upper"]


def test_replay_and_controls():
    first = lab.run_lab(repetitions=4, sample_size=16, n_candidates=3)
    second = lab.run_lab(**first["config"])
    assert first["families"] == second["families"]
    assert not first["families"]["hidden_family"]["fixed_reported_family"]
    assert not first["families"]["repeated_row"]["iid_rows"]
    hidden = first["families"]["hidden_family"]["records"][0]
    assert hidden["pool_size"] == 48
    assert hidden["intervals"]["full_pool"]["lower"] <= hidden["intervals"]["bonferroni"]["lower"]


def test_subnormal_finite_inaccurate_inverse_fails_closed():
    interval = lab.clopper_pearson(5, 10, alpha=1e-320)
    assert interval["status"] == "numerical_fallback"
    assert (interval["lower"], interval["upper"]) == (0, 1)


def test_no_selection_is_not_covered_target():
    report = lab.run_lab(repetitions=3, sample_size=2, n_candidates=2, families=["all_ties"])
    result = report["families"]["all_ties"]
    assert result["no_selection"] == 3
    assert not result["population_target_defined"]
    assert result["summary"]["naive"]["conditional_coverage"] is None
    assert result["summary"]["naive"]["unconditional_miscoverage"] == 0
    assert all(not r["intervals"] for r in result["records"])


def test_unconditional_denominator_includes_no_selection():
    interval = {"covered": False, "lower": 0.6, "upper": 0.9, "status": "ok"}
    result = lab._summary([{"intervals": {"naive": interval}}, {"intervals": {}}], "naive")
    assert result["conditional_coverage"] == 0
    assert result["unconditional_miscoverage"] == 0.5


@pytest.mark.parametrize("values", [[[np.nan]], [[1j]], [[True]], [1, 2], np.empty((2, 0))])
def test_invalid_sign_arrays(values):
    with pytest.raises(ValueError):
        lab.sign_counts(values)


@pytest.mark.parametrize("positive,n", [(True, 1), (1, False), (-1, 2), (3, 2), (1.0, 2)])
def test_invalid_counts(positive, n):
    with pytest.raises((TypeError, ValueError)):
        lab.clopper_pearson(positive, n)


@pytest.mark.parametrize("n", [1_000_001, 10**20, 10**400])
def test_huge_counts_fail_closed(n):
    for k in [0, 1, n // 2, n]:
        result = lab.clopper_pearson(k, n)
        assert result["status"] == "unsupported_count"
        assert (result["lower"], result["upper"]) == (0, 1)


@pytest.mark.parametrize("k", [0, 1, 100, 500000, 999999, 1000000])
def test_qualified_count_edge_independent_root(k):
    n = lab.MAX_QUALIFIED_COUNT
    result = lab.clopper_pearson(k, n)
    # scipy.stats uses binomial-tail Brent root inversion rather than beta inverses.
    reference = binomtest(k, n).proportion_ci(method="exact")
    assert result["status"] == "ok"
    assert result["lower"] == pytest.approx(reference.low, abs=2e-12)
    assert result["upper"] == pytest.approx(reference.high, abs=2e-12)


def test_normal_extreme_tail_independent_finite_sum():
    for n, k in [(10, 5), (100, 50), (256, 64)]:
        for alpha in [1e-100, 1e-250, 1e-300]:
            ci = lab.clopper_pearson(k, n, alpha=alpha)
            p = ci["lower"]
            logs = [
                math.log(math.comb(n, j)) + j * math.log(p) + (n - j) * math.log1p(-p)
                for j in range(k, n + 1)
            ]
            maximum = max(logs)
            logtail = maximum + math.log(sum(math.exp(x - maximum) for x in logs))
            assert logtail == pytest.approx(math.log(ci["tail_budget"]), abs=2e-10)

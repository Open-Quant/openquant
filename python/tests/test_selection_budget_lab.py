"""Exact finite model tests for optional selection-budget research."""

import importlib.util
import itertools
import math
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("scipy", reason="optional selection-interval research dependency")
spec = importlib.util.spec_from_file_location(
    "selection_budget_lab",
    Path(__file__).resolve().parents[2] / "experiments/selection_budget_lab.py",
)
assert spec is not None and spec.loader is not None
lab = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lab)


def pmf(k, n, p):
    return math.comb(n, k) * p**k * (1 - p) ** (n - k)


def exhaustive(n, k, p):
    designated = np.zeros(n + 1)
    other = np.zeros(n + 1)
    for counts in itertools.product(range(n + 1), repeat=k):
        weight = math.prod(pmf(count, n, p if j == 0 else 0.5) for j, count in enumerate(counts))
        winner = max(range(k), key=lambda j: counts[j])
        (designated if winner == 0 else other)[counts[winner]] += weight
    return designated, other


@pytest.mark.parametrize("n,k", [(2, 1), (2, 3), (4, 2), (4, 3)])
@pytest.mark.parametrize("p", [0.0, 0.5, 0.65, 1.0])
def test_winner_mass_exhaustive(n, k, p):
    actual = lab.winner_masses(n, k, p)
    expected = exhaustive(n, k, p)
    np.testing.assert_allclose(actual, expected, atol=2e-15, rtol=1e-13)


def test_equal_budget_claims_exhaustive():
    n, k, p = 8, 3, 0.8
    record = lab.evaluate_budget(n, k, signal_probability=p)
    for label, discovery, correct in [
        ("full_sample_bonferroni", n, k),
        ("split_holdout", n // 2, 1),
    ]:
        metrics = dict(mean_width=0.0, miscoverage=0.0, positive_claim=0.0)
        for counts in itertools.product(range(discovery + 1), repeat=k):
            weight = math.prod(
                pmf(c, discovery, p if j == 0 else 0.5) for j, c in enumerate(counts)
            )
            selected = max(range(k), key=lambda j: counts[j])
            target = p if selected == 0 else 0.5
            observations = (
                [(counts[selected], 1.0)]
                if label == "full_sample_bonferroni"
                else [(c, pmf(c, discovery, target)) for c in range(discovery + 1)]
            )
            for successes, audit_weight in observations:
                ci = lab.intervals.clopper_pearson(successes, discovery, family_size=correct)
                metrics["mean_width"] += weight * audit_weight * (ci["upper"] - ci["lower"])
                metrics["miscoverage"] += (
                    weight * audit_weight * (not ci["lower"] <= target <= ci["upper"])
                )
                metrics["positive_claim"] += weight * audit_weight * (ci["lower"] > 0.5)
        for metric, value in metrics.items():
            assert record[label][metric] == pytest.approx(value, abs=1e-13)


def test_allnull_and_evaluation_costs():
    r = lab.evaluate_budget(64, 8, signal_probability=0.5)
    full, split = r["full_sample_bonferroni"], r["split_holdout"]
    assert full["candidate_evaluations"] == 512
    assert split["candidate_evaluations"] == 288
    for m in (full, split):
        assert m["true_positive_claim"] == 0
        assert m["positive_claim"] == m["null_positive_claim"]
        assert m["expected_selected_sign_probability"] == 0.5
        assert m["miscoverage"] <= 0.05


def test_signal_selection_cost_and_width_tradeoff():
    r = lab.evaluate_budget(128, 32)
    full, split = r["full_sample_bonferroni"], r["split_holdout"]
    assert full["select_designated_candidate"] > split["select_designated_candidate"]
    assert full["mean_width"] > split["mean_width"]
    assert full["true_positive_claim"] > split["true_positive_claim"]


@pytest.mark.parametrize(
    "n,k,p",
    [
        (3, 2, 0.65),
        (0, 1, 0.65),
        (True, 2, 0.65),
        (4, 0, 0.65),
        (4, 2, np.nan),
        (4, 2, 1j),
        (4, 2, ".65"),
        (4, 2, 1.1),
    ],
)
def test_invalid(n, k, p):
    with pytest.raises((ValueError, TypeError)):
        lab.evaluate_budget(n, k, signal_probability=p)


def test_replay_and_cached_arrays_readonly():
    a = lab.run_lab(row_budgets=[8], candidate_counts=[1, 3], probabilities=[0.5, 0.8])
    b = lab.run_lab(**a["config"])
    assert a["records"] == b["records"]
    assert "selection_interval_lab.py" in a["provenance"]["dependency_sha256"]
    with pytest.raises(ValueError):
        lab._interval_table(8, 1, 0.05)[0][0] = 0.2

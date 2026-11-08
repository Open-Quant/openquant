"""Exact finite examples, concentration constants, and explicit assumption violations."""

from __future__ import annotations

import importlib.util
import itertools
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

MODULE = Path(__file__).resolve().parents[2] / "experiments" / "bounded_finite_family_lab.py"
spec = importlib.util.spec_from_file_location("bounded_finite_family_lab", MODULE)
assert spec is not None and spec.loader is not None
lab = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lab)


def test_exact_two_observation_case_uses_nonabsolute_unscaled_convention():
    x = [[1, 0], [0, 1]]
    # Four sign vectors have max scores [.5, .5, .5, -.5], averaging .25.
    assert lab.exact_complexity(x) == pytest.approx(0.25)
    signs = np.array(list(itertools.product([-1, 1], repeat=2)))
    np.testing.assert_array_equal(lab.antithetic_scores(x, signs), [0, 0.5, 0.5, 0])


def test_exact_oracle_matches_direct_unpaired_maximum_definition():
    x = np.array([[0.2, 0.8, 0.4], [0.9, 0.3, 0.5], [0.1, 0.4, 0.7]])
    unpaired = []
    for signs in itertools.product([-1, 1], repeat=3):
        unpaired.append(max(sum(signs[i] * x[i, j] for i in range(3)) / 3 for j in range(3)))
    assert lab.exact_complexity(x) == pytest.approx(sum(unpaired) / 8)


def test_singleton_and_identical_empirical_columns_have_exact_zero_complexity():
    x = np.array([[0.2], [0.8], [1.0]])
    for values in (x, np.repeat(x, 5, axis=1)):
        assert lab.exact_complexity(values) == 0
        estimate = lab.estimate_complexity(values, draws=10, eta=0.025, seed=0)
        assert estimate["upper"] == estimate["estimate"] == 0
        assert estimate["draws_used"] == 0
        assert estimate["method"] == "exact_identical_columns"


def test_antithetic_range_bound_is_attainable_and_nonnegative():
    x = [[1, 0], [1, 0], [1, 0]]
    values = lab.antithetic_scores(x, [[1, 1, 1], [-1, -1, -1], [-1, 1, -1]])
    np.testing.assert_allclose(values, [0.5, 0.5, 1 / 6])
    assert np.all((values >= 0) & (values <= 0.5))


def test_duplicate_and_permutation_invariance_of_complexity():
    x = np.array([[1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1]])
    expected = lab.exact_complexity(x)
    assert lab.exact_complexity(x[:, ::-1]) == expected
    assert lab.exact_complexity(np.repeat(x, 3, axis=1)) == expected
    # Range can only grow on adding columns, for each fixed sign vector.
    assert lab.exact_complexity(x[:, :2]) <= expected


def test_complement_preserves_empirical_complexity_for_lower_bound_derivation():
    x = np.array([[0.1, 0.3], [0.6, 0.4], [0.2, 0.9]])
    assert lab.exact_complexity(1 - x) == pytest.approx(lab.exact_complexity(x))


def test_monte_carlo_matches_exact_oracle_and_has_hoeffding_upper_correction():
    result = lab.estimate_complexity([[1, 0], [0, 1]], draws=10000, eta=0.025, seed=31)
    assert result["estimate"] == pytest.approx(0.25, abs=0.01)
    assert result["mc_upper_margin"] == pytest.approx(math.sqrt(math.log(40) / (8 * 10000)))
    assert result["upper"] == result["estimate"] + result["mc_upper_margin"]
    assert 0 < result["conditional_mcse"] < 0.01


def test_probability_budget_and_bound_constants_are_explicit_and_fair():
    result = lab.evaluate_matrix([[0], [1]], [0.5], total_failure=0.05, draws=2)
    assert result["sample_failure_delta"] == result["mc_failure_eta"] == 0.025
    assert result["total_failure"] == 0.05
    assert result["rademacher_sample_margin"] == pytest.approx(3 * math.sqrt(math.log(80) / 4))
    assert result["bounds"]["rademacher"]["width"] == result["rademacher_sample_margin"]
    assert result["bounds"]["finite_union"]["width"] == pytest.approx(math.sqrt(math.log(20) / 4))


def test_repeated_row_is_a_counterexample_to_applying_iid_guarantee_to_dependence():
    x = np.repeat([[1.0, 0.0]], 2048, axis=0)
    result = lab.evaluate_matrix(x, [0.5, 0.5], draws=512)
    assert result["selected_candidate"] == 0
    for method in result["bounds"].values():
        assert method["selected_lower"] > 0.5
        assert not method["selected_coverage"]
        assert not method["simultaneous_coverage"]


def test_exact_ties_select_first_and_known_true_means_determine_coverage():
    result = lab.evaluate_matrix([[0, 1], [1, 0]], [0.5, 0.5], draws=20)
    assert result["selected_candidate"] == 0
    assert result["selected_true_mean"] == 0.5
    assert all(bound["simultaneous_coverage"] for bound in result["bounds"].values())


@pytest.mark.parametrize(
    "x",
    [
        [],
        [[-0.1]],
        [[1.1]],
        [[np.nan]],
        [[np.inf]],
        [[1 + 0j]],
        [[True]],
        [["0.5"]],
        [0.5],
        [[0], [1, 0]],
    ],
)
def test_invalid_payoffs_rejected_without_silent_clipping(x):
    with pytest.raises((ValueError, TypeError)):
        lab.exact_complexity(x)


@pytest.mark.parametrize("signs", [[[0]], [[True]], [[1, -1]], [], [[1 + 0j]]])
def test_invalid_sign_inputs_rejected(signs):
    with pytest.raises((ValueError, TypeError)):
        lab.antithetic_scores([[0.5]], signs)


@pytest.mark.parametrize(
    "alpha", [0, 1, -0.1, np.inf, np.nan, True, "0.05", np.complex128(0.05 + 1j)]
)
def test_invalid_probability_budgets_rejected(alpha):
    with pytest.raises((ValueError, TypeError)):
        lab.evaluate_matrix([[0]], [0.5], total_failure=alpha)


def test_subnormal_probability_split_does_not_exceed_total_budget():
    tiny = np.nextafter(0.0, 1.0)
    alpha = 3 * tiny
    result = lab.evaluate_matrix([[0]], [0.5], total_failure=alpha, draws=2)
    assert result["sample_failure_delta"] > 0
    assert result["mc_failure_eta"] > 0
    assert result["sample_failure_delta"] + result["mc_failure_eta"] == alpha
    with pytest.raises(ValueError, match="too small"):
        lab.evaluate_matrix([[0]], [0.5], total_failure=tiny, draws=2)


def test_wilson_intervals_include_exact_zero_and_one_outcomes():
    assert lab._event_summary([False] * 3)["wilson_95_interval"][0] == 0
    assert lab._event_summary([True] * 3)["wilson_95_interval"][1] == 1


@pytest.mark.parametrize("truth", [[], [0.5, 0.6], [np.nan], [1.1], [-0.1], [True]])
def test_invalid_known_means_rejected(truth):
    with pytest.raises((ValueError, TypeError)):
        lab.evaluate_matrix([[0]], truth)


def test_exponential_exact_enumeration_is_guarded():
    with pytest.raises(ValueError, match="16"):
        lab.exact_complexity(np.zeros((17, 2)))


def test_generators_keep_support_known_means_and_explicit_dependence():
    for family in lab.FAMILIES:
        values, truth = lab.simulate_family(family, 25, 5, seed=8)
        assert values.shape == (25, 5)
        assert np.all((values == 0) | (values == 1))
        np.testing.assert_array_equal(truth[1:], 0.5)
        assert truth[0] == (0.8 if family == "known_signal" else 0.5)
        if family == "duplicated_null":
            np.testing.assert_array_equal(values[:, 0], values[:, 1])
        if family == "repeated_row":
            np.testing.assert_array_equal(values, np.repeat(values[:1], 25, axis=0))


def test_reports_reproduce_independent_of_requested_family_order():
    kwargs = dict(repetitions=3, sample_size=10, n_candidates=4, draws=10)
    a = lab.run_lab(**kwargs, families=("iid_null", "repeated_row"))
    b = lab.run_lab(**kwargs, families=("repeated_row", "iid_null"))
    assert a["families"] == b["families"]
    assert not a["families"]["repeated_row"]["iid_rows"]
    assert a["families"]["iid_null"]["iid_rows"]
    assert len(a["provenance"]["source_sha256"]) == 64
    json.dumps(a, allow_nan=False)


def test_summaries_preserve_all_replications_and_paired_width_differences():
    result = lab.run_lab(repetitions=4, sample_size=12, n_candidates=3, draws=20)
    for family in result["families"].values():
        records = family["records"]
        assert len(records) == 4
        differences = [
            r["bounds"]["rademacher"]["width"] - r["bounds"]["finite_union"]["width"]
            for r in records
        ]
        summary = family["summary"]["rademacher_minus_union_width"]
        assert summary["mean"] == pytest.approx(np.mean(differences))
        assert summary["monte_carlo_standard_error"] == pytest.approx(
            np.std(differences, ddof=1) / 2
        )
        event = family["summary"]["rademacher"]["simultaneous_coverage"]
        assert event["count"] == sum(
            r["bounds"]["rademacher"]["simultaneous_coverage"] for r in records
        )
        assert event["wilson_95_interval"][0] <= event["fraction"] <= event["wilson_95_interval"][1]


def test_cli_does_not_need_openquant_native_extension(tmp_path):
    out = tmp_path / "result.json"
    subprocess.run(
        [
            sys.executable,
            str(MODULE),
            "--out",
            str(out),
            "--repetitions",
            "2",
            "--sample-size",
            "8",
            "--candidates",
            "2",
            "--draws",
            "4",
        ],
        check=True,
        capture_output=True,
    )
    assert json.loads(out.read_text())["lab"] == "bounded_finite_family"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"repetitions": 1},
        {"sample_size": 0},
        {"n_candidates": True},
        {"draws": 1},
        {"seed": -1},
        {"families": ()},
        {"families": ("iid_null", "iid_null")},
        {"families": ("missing",)},
    ],
)
def test_invalid_run_config_fails_closed(kwargs):
    with pytest.raises((ValueError, TypeError)):
        lab.run_lab(**kwargs)

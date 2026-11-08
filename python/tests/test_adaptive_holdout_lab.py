"""Audit isolation and hand-computed controls for repeated validation feedback."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

MODULE = Path(__file__).resolve().parents[2] / "experiments" / "adaptive_holdout_lab.py"
spec = importlib.util.spec_from_file_location("adaptive_holdout_lab", MODULE)
assert spec is not None and spec.loader is not None
lab = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lab)


def arrays():
    return (
        np.array([[1, -2], [2, 0], [3, 2]], dtype=float),
        np.array([[-3, 1], [-2, 2], [-1, 3]], dtype=float),
        np.array([[1, -3], [2, -2], [3, -1]], dtype=float),
    )


def test_feedback_selects_positive_blend_and_audit_does_not_help_it():
    result = lab.evaluate_search(*arrays(), rounds=1)
    assert result["training_order"] == [0, 1]
    before, after = result["snapshots"]
    assert before["adaptive_validation_sharpe"] == pytest.approx(-2)
    assert after["proposal_choice"] == "positive_blend"
    assert after["adaptive_weights"] == [0.5, 0.5]
    assert after["adaptive_validation_sharpe"] == pytest.approx(0)
    assert after["adaptive_audit_sharpe"] == pytest.approx(0)
    assert after["single_candidate"] == 1
    assert after["single_validation_sharpe"] == pytest.approx(2)
    assert after["single_audit_sharpe"] == pytest.approx(-2)
    assert result["frozen_audit_sharpe"] == pytest.approx(2)
    assert after["undefined_proposals_cumulative"] == 1  # constant negative blend


def test_audit_perturbations_leave_entire_search_path_unchanged():
    train, valid, audit = arrays()
    a = lab.evaluate_search(train, valid, audit, rounds=1)
    b = lab.evaluate_search(train, valid, audit[::-1] * -13, rounds=1)
    assert a["training_order"] == b["training_order"]
    for x, y in zip(a["snapshots"], b["snapshots"], strict=True):
        for key in (
            "adaptive_weights",
            "proposal_choice",
            "single_candidate",
            "adaptive_validation_sharpe",
            "undefined_proposals_cumulative",
        ):
            assert x[key] == y[key]
    assert a["frozen_audit_sharpe"] == -b["frozen_audit_sharpe"]


def test_exact_duplicate_proposals_retain_incumbent_and_track_undefined_difference():
    values = np.array([[1, 1], [2, 2], [3, 3]])
    result = lab.search_validation(values, values, rounds=1)
    assert result["training_order"] == [0, 1]
    assert result["snapshots"][1]["proposal_choice"] == "retain"
    assert result["snapshots"][1]["undefined_proposals_cumulative"] == 1
    assert result["snapshots"][1]["single_candidate"] == 0


def test_constant_audit_mixture_is_preserved_as_unavailable_not_zero():
    train, valid, _ = arrays()
    audit = np.array([[1, -1], [2, -2], [3, -3]])
    result = lab.evaluate_search(train, valid, audit, rounds=1)
    final = result["snapshots"][-1]
    assert final["adaptive_audit_sharpe"] is None
    assert final["adaptive_validation_minus_audit"] is None
    assert result["frozen_audit_sharpe"] == pytest.approx(2)


def test_zero_rounds_is_frozen_training_control_and_does_not_search_validation():
    result = lab.evaluate_search(*arrays(), rounds=0)
    assert len(result["snapshots"]) == 1
    row = result["snapshots"][0]
    assert row["adaptive_weights"] == [1, 0]
    assert row["adaptive_audit_sharpe"] == result["frozen_audit_sharpe"]


def test_validation_optimism_is_monotone_but_audit_need_not_improve():
    rng = np.random.default_rng(28)
    matrices = [rng.normal(size=(60, 12)) for _ in range(3)]
    result = lab.evaluate_search(*matrices, rounds=10)
    scores = [row["adaptive_validation_sharpe"] for row in result["snapshots"]]
    assert np.all(np.diff(scores) >= 0)
    assert all(np.sum(np.abs(row["adaptive_weights"])) <= 1 for row in result["snapshots"])


def test_shorter_search_is_exact_prefix_not_reoptimized():
    rng = np.random.default_rng(7)
    train, valid = [rng.normal(size=(30, 9)) for _ in range(2)]
    short = lab.search_validation(train, valid, rounds=3)
    long = lab.search_validation(train, valid, rounds=8)
    assert short["snapshots"] == long["snapshots"][:4]


def test_negative_blend_is_a_signed_combination_and_feedback_changes_choice():
    train, _, _ = arrays()
    valid = np.array([[-1, -2], [0, -4], [1, -3]])
    result = lab.search_validation(train, valid, rounds=1)
    final = result["snapshots"][-1]
    assert final["proposal_choice"] == "negative_blend"
    assert final["adaptive_weights"] == [0.5, -0.5]
    assert final["adaptive_validation_sharpe"] == pytest.approx(3**0.5)
    assert final["adaptive_new_proposal_scores"] == 3
    assert final["single_candidate_scores_used"] == 2
    alternate = lab.search_validation(train, -valid, rounds=1)
    assert alternate["training_order"] == result["training_order"]
    assert alternate["snapshots"][-1]["adaptive_weights"] != final["adaptive_weights"]


def test_unqueried_validation_columns_do_not_affect_scores_or_rejections():
    rng = np.random.default_rng(93)
    train, valid = [rng.normal(size=(40, 8)) for _ in range(2)]
    baseline = lab.search_validation(train, valid, rounds=2)
    unseen = baseline["training_order"][3:]
    changed = valid.copy()
    changed[:, unseen] = 1e300  # Constant and extreme, but never queried in these rounds.
    assert lab.search_validation(train, changed, rounds=2) == baseline
    with pytest.raises(ValueError, match="constant"):
        lab.search_validation(train, changed, rounds=3)


def test_unqueried_large_column_does_not_underflow_selected_small_series():
    train = [[1, -1e308], [2, 0], [3, 1e308]]
    valid = [[1e-308, -1e308], [2e-308, 0], [3e-308, 1e308]]
    audit = [[1, -2], [2, 0], [3, 2]]
    result = lab.evaluate_search(train, valid, audit, rounds=0)
    assert result["snapshots"][0]["adaptive_validation_sharpe"] == pytest.approx(2)
    assert result["snapshots"][0]["adaptive_audit_sharpe"] == pytest.approx(2)


@pytest.mark.parametrize("rounds", [-1, True, 0.5, 2])
def test_invalid_round_budget_rejected(rounds):
    train, valid, _ = arrays()
    with pytest.raises((TypeError, ValueError)):
        lab.search_validation(train, valid, rounds=rounds)


def test_mismatched_and_nonfinite_inputs_rejected():
    train, valid, audit = arrays()
    with pytest.raises(ValueError):
        lab.evaluate_search(train, valid, audit[:, :1], rounds=1)
    with pytest.raises(ValueError):
        lab.search_validation(train, valid[:, :1], rounds=1)
    audit[0, 0] = np.nan
    with pytest.raises(ValueError):
        lab.evaluate_search(train, valid, audit, rounds=1)


def test_family_streams_and_records_invariant_to_order_and_replay():
    kwargs = dict(repetitions=3, sample_size=20, n_candidates=4, rounds=3)
    a = lab.run_lab(**kwargs, families=("iid_null", "serial_null"))
    b = lab.run_lab(**kwargs, families=("serial_null", "iid_null"))
    assert a["families"] == b["families"]
    assert len(a["provenance"]["source_sha256"]) == 64
    assert len(a["provenance"]["dependency_sha256"]["validation_lab.py"]) == 64
    assert a["families"]["iid_null"]["by_round"][0]["adaptive_minus_frozen_audit"][
        "mean"
    ] == pytest.approx(0, abs=1e-15)
    json.dumps(a, allow_nan=False)


def test_monte_carlo_uses_paired_replication_differences():
    result = lab.run_lab(
        repetitions=4, sample_size=20, n_candidates=4, rounds=2, families=("iid_null",)
    )
    family = result["families"]["iid_null"]
    differences = np.array(
        [
            r["snapshots"][-1]["adaptive_audit_sharpe"] - r["frozen_audit_sharpe"]
            for r in family["records"]
        ]
    )
    summary = family["by_round"][-1]["adaptive_minus_frozen_audit"]
    assert summary["mean"] == pytest.approx(differences.mean())
    assert summary["monte_carlo_standard_error"] == pytest.approx(differences.std(ddof=1) / 2)


def test_all_synthetic_controls_produce_serializable_reports():
    result = lab.run_lab(repetitions=2, sample_size=12, n_candidates=4, rounds=3)
    assert set(result["families"]) == set(lab.lab.FAMILIES)
    for family in result["families"].values():
        assert len(family["records"]) == 2
        assert len(family["by_round"]) == 4
    json.dumps(result, allow_nan=False)


def test_independent_segments_use_distinct_deterministic_streams(monkeypatch):
    original, states = lab.lab.simulate_family, []

    def record(*args, **kwargs):
        states.append(kwargs["seed"].entropy)
        return original(*args, **kwargs)

    monkeypatch.setattr(lab.lab, "simulate_family", record)
    lab.run_lab(repetitions=2, sample_size=12, n_candidates=2, rounds=1, families=("serial_null",))
    assert len(states) == len({tuple(state) for state in states}) == 6
    assert [state[3] for state in states] == [0, 1, 2, 0, 1, 2]


def test_cli_runs_without_native_extension_and_writes_strict_json(tmp_path):
    target = tmp_path / "report.json"
    subprocess.run(
        [
            sys.executable,
            str(MODULE),
            "--out",
            str(target),
            "--repetitions",
            "2",
            "--sample-size",
            "10",
            "--candidates",
            "2",
            "--rounds",
            "1",
            "--families",
            "iid_null",
        ],
        check=True,
        capture_output=True,
    )
    assert json.loads(target.read_text())["lab"] == "adaptive_holdout"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"repetitions": 1},
        {"sample_size": 3},
        {"rounds": 32},
        {"seed": -1},
        {"families": ()},
        {"families": ("iid_null", "iid_null")},
        {"families": ("bogus",)},
    ],
)
def test_invalid_experiment_configuration_rejected(kwargs):
    with pytest.raises((ValueError, TypeError)):
        lab.run_lab(**kwargs)

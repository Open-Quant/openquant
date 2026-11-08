"""Synthetic workflow checks via isolated source import, not package integration."""

import copy
import importlib.util
from pathlib import Path

import pytest

PATH = Path(__file__).resolve().parents[2] / "examples/availability_end_to_end.py"
spec = importlib.util.spec_from_file_location("availability_workflow_e2e", PATH)
assert spec is not None and spec.loader is not None
workflow = importlib.util.module_from_spec(spec)
spec.loader.exec_module(workflow)
SELECT = workflow.load_selector(True)


@pytest.fixture(scope="module")
def report():
    return workflow.run_workflow(SELECT)


def test_frozen_counts_and_information_flow(report):
    assert report["protocol"] == dict(
        seeds=[7, 19, 43], rows=96, fit_positions=[48, 64, 80], test_rows=8
    )
    unpublished = 0
    for run in report["runs"]:
        for fold in run["folds"]:
            assert (
                fold["test_count"] == len(fold["common_ids"]) + fold["missing_pit_test_count"] == 8
            )
            assert set(fold["candidate_ids"]).isdisjoint(fold["test_ids"])
            for name, arm in fold["arms"].items():
                assert {s["row_id"] for s in arm["training"]} <= set(fold["candidate_ids"])
                assert all(int(s["row_id"].split("-")[1]) % 11 != 0 for s in arm["training"])
                if name.startswith("pit_"):
                    assert arm["future_training_source_count"] == 0
                else:
                    assert arm["future_training_source_count"] == arm["training_count"]
                if name.endswith("_published"):
                    assert arm["unpublished_training_label_count"] == 0
                unpublished += arm["unpublished_training_label_count"]
                for vintage, score in arm["scores"].items():
                    assert [p["row_id"] for p in score["predictions"]] == fold["common_ids"]
                    assert score["future_source_count"] == (
                        0 if vintage == "pit" else len(fold["common_ids"])
                    )
                    assert all(p["decision"] >= fold["fit_time"] for p in score["predictions"])
                if name.startswith("final_"):
                    assert arm["scores"]["final"]["mse"] < 1e-24  # Constructed y/2 leakage control.
    assert unpublished > 0


def test_row_and_revision_reordering_does_not_change_result():
    rows, revisions = workflow.generate(7)
    original = workflow.run_fold(SELECT, rows, revisions, 48)
    assert original == workflow.run_fold(
        SELECT, list(reversed(rows)), list(reversed(revisions)), 48
    )


def test_future_append_does_not_change_historical_safe_arm():
    rows, revisions = workflow.generate(7)
    old = workflow.run_fold(SELECT, rows, revisions, 48)
    future = [
        dict(r, available=r["available"] + 10**18, value=r["value"] + 999, revision="future")
        for r in revisions
    ]
    new = workflow.run_fold(SELECT, rows, revisions + future, 48)
    # Final test diagnostics intentionally change; safe model and PIT score do not.
    assert old["arms"]["pit_published"]["model"] == new["arms"]["pit_published"]["model"]
    assert (
        old["arms"]["pit_published"]["scores"]["pit"]
        == new["arms"]["pit_published"]["scores"]["pit"]
    )


def test_heldout_targets_and_sources_do_not_change_fitted_state():
    rows, revisions = workflow.generate(19)
    old = workflow.run_fold(SELECT, rows, revisions, 48)
    changed_rows, changed_revisions = copy.deepcopy(rows), copy.deepcopy(revisions)
    heldout = set(old["test_ids"])
    for row in changed_rows:
        if row["row_id"] in heldout:
            row["y"] += 1000
    for revision in changed_revisions:
        if revision["row_id"] in heldout:
            revision["value"] -= 500
    new = workflow.run_fold(SELECT, changed_rows, changed_revisions, 48)
    for name in old["arms"]:
        assert old["arms"][name]["training"] == new["arms"][name]["training"]
        assert old["arms"][name]["model"] == new["arms"][name]["model"]


def test_snapshot_inclusive_boundaries_conflicts_and_missing():
    source = dict(row_id="a", revision="v1", event=10, available=11, value=2)
    assert workflow.snapshot([source], "a", 10) is None
    assert workflow.snapshot([source], "a", 11) == source
    assert workflow.snapshot([dict(source, event=12)], "a", 11) is None
    with pytest.raises(ValueError, match="conflicting"):
        workflow.snapshot([source, dict(source, value=3)], "a", 11)
    assert workflow.snapshot([source, dict(source, revision="v0")], "a", 11)["revision"] == "v0"


def test_false_clock_passes_but_truthful_clock_rejects(report):
    assert report["false_clock_control"]["claimed_eligible"] == [0]
    assert report["false_clock_control"]["truthful_eligible"] == []


def test_duplicate_identity_rejected_and_empty_training_has_no_fallback():
    rows, revisions = workflow.generate(7)
    with pytest.raises(ValueError, match="identity"):
        workflow.run_fold(SELECT, rows + [rows[0]], revisions, 48)
    with pytest.raises(ValueError, match="at least two"):
        workflow.fit_ols([])


def test_empty_common_cohort_rejects_instead_of_imputing():
    rows, revisions = workflow.generate(7)
    test_ids = {r["row_id"] for r in rows if 48 <= r["position"] < 56}
    revisions = [
        dict(r, available=r["available"] + 10**18) if r["row_id"] in test_ids else r
        for r in revisions
    ]
    with pytest.raises(ValueError, match="empty common"):
        workflow.run_fold(SELECT, rows, revisions, 48)


def test_stored_target_is_not_training_permission_before_label_resolution():
    rows, revisions = workflow.generate(7)
    old = workflow.run_fold(SELECT, rows, revisions, 48)
    row_id = "row-047"  # Last historical candidate, initially eligible in every arm.
    assert all(row_id in {s["row_id"] for s in arm["training"]} for arm in old["arms"].values())
    changed = copy.deepcopy(rows)
    for row in changed:
        if row["row_id"] == row_id:
            row["end"] = old["fit_time"] + workflow.HOUR
            row["publication"] = old["fit_time"] + 2 * workflow.HOUR
            row["y"] = 1e30  # A stored retrospective target must not enter any fitted model.
    new = workflow.run_fold(SELECT, changed, revisions, 48)
    assert row_id in new["candidate_ids"]
    assert set(new["safe_exclusions"][row_id]) == {
        "label_end_not_before_fit",
        "label_available_after_fit",
    }
    for name, arm in new["arms"].items():
        expected = [
            sample for sample in old["arms"][name]["training"] if sample["row_id"] != row_id
        ]
        assert arm["training"] == expected
        assert arm["model"] == workflow.fit_ols(expected)


def test_fold_with_no_mature_training_rows_fails_without_reusing_targets(monkeypatch):
    rows, revisions = workflow.generate(7)
    fit = next(row["decision"] for row in rows if row["position"] == 48)
    for row in rows:
        if row["decision"] < fit:
            row["end"] = fit + workflow.HOUR
            row["publication"] = fit + 2 * workflow.HOUR
    observed_samples = []
    original_fit = workflow.fit_ols

    def capture_fit(samples):
        observed_samples.append(samples)
        return original_fit(samples)

    monkeypatch.setattr(workflow, "fit_ols", capture_fit)
    with pytest.raises(ValueError, match="at least two training rows"):
        workflow.run_fold(SELECT, rows, revisions, 48)
    assert observed_samples == [[]]  # No other arm or withheld-row fallback is attempted.


@pytest.mark.parametrize("relative_to_decision", [-1, 0, 1])
def test_prediction_source_uses_prediction_cutoff_not_model_fit(relative_to_decision):
    rows, revisions = workflow.generate(19)
    old = workflow.run_fold(SELECT, rows, revisions, 48)
    row_id = "row-050"
    decision = next(row["decision"] for row in rows if row["row_id"] == row_id)
    publication = decision + relative_to_decision
    assert old["fit_time"] < publication
    for revision in revisions:
        if revision["row_id"] == row_id and revision["revision"] == "initial":
            revision["available"] = publication
            revision["value"] = 123.0
    new = workflow.run_fold(SELECT, rows, revisions, 48)
    for name in old["arms"]:
        assert new["arms"][name]["training"] == old["arms"][name]["training"]
        assert new["arms"][name]["model"] == old["arms"][name]["model"]
    safe = new["arms"]["pit_published"]
    predictions = {p["row_id"]: p for p in safe["scores"]["pit"]["predictions"]}
    if relative_to_decision <= 0:
        assert row_id in new["common_ids"]
        assert predictions[row_id]["source"]["available"] == publication
        assert predictions[row_id]["prediction"] == workflow.predict(safe["model"], 123.0)
        assert predictions[row_id]["prediction"] != next(
            p["prediction"]
            for p in old["arms"]["pit_published"]["scores"]["pit"]["predictions"]
            if p["row_id"] == row_id
        )
    else:
        assert row_id not in new["common_ids"]
        assert row_id not in predictions
        assert new["missing_pit_test_count"] == old["missing_pit_test_count"] + 1

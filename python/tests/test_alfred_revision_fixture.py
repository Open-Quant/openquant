"""Offline documented facts and deliberately restricted date-domain behavior."""

import copy
import importlib.util
from datetime import date
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "alfred_revision_fixture", ROOT / "examples/alfred_revision_fixture.py"
)
assert spec is not None and spec.loader is not None
example = importlib.util.module_from_spec(spec)
spec.loader.exec_module(example)


def test_public_documented_values_and_known_subset_policy():
    result = example.run()
    assert result["earlier_known_subset_selection"]["value"] == "5.3"
    assert result["later_known_subset_selection"]["value"] == "5.4"
    assert result["naive_latest_for_earlier_cutoff"]["value"] == "5.4"
    assert result["difference_percentage_points"] == "0.1"
    assert result["same_date_record_excluded"]
    assert result["old_identity_with_later_value_rejected"]
    assert "not attempted" in result["nanosecond_binding"]


def test_incomplete_history_changes_selection_without_claiming_authenticity():
    fixture = example.load_fixture()
    original = example.latest_documented_before(fixture, date(1996, 3, 9))
    omitted = copy.deepcopy(fixture)
    omitted["records"].pop()
    selected = example.latest_documented_before(omitted, date(1996, 3, 9))
    assert original["value"] == "5.4"
    assert selected["value"] == "5.3"
    example.check_selected_association(selected, selected)


def test_date_ties_fail_and_input_order_does_not_select_revision():
    fixture = example.load_fixture()
    expected = example.latest_documented_before(fixture, date(1996, 3, 9))
    fixture["records"].reverse()
    assert example.latest_documented_before(fixture, date(1996, 3, 9)) == expected
    fixture["records"].append(dict(expected, fixture_id="conflicting-local-id"))
    with pytest.raises(ValueError, match="ambiguous"):
        example.latest_documented_before(fixture, date(1996, 3, 9))

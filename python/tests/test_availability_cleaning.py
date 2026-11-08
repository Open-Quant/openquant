"""Actual Polars cleaner integration; source imports are not native-package tests."""

import copy
import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("polars")
ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "availability_cleaning_example", ROOT / "examples/availability_cleaning.py"
)
assert spec is not None and spec.loader is not None
example = importlib.util.module_from_spec(spec)
spec.loader.exec_module(example)


def test_real_cleaner_and_false_provenance_controls():
    availability, data = example.load_modules(True)
    result = example.run_example(availability, data)
    assert result["safe"]["values"] == [10.0, 20.0]
    assert result["safe"]["record_ids"] == ["1-initial", "2-initial"]
    assert result["safe"]["dropped_null_rows"] == 1
    assert result["false_revision_metadata"]["values"] == [100.0, 20.0]
    assert result["false_revision_metadata"]["eligible_indices"] == [0, 1]
    assert result["false_revision_metadata"]["actual_late_revision_rows"] == [0]
    assert result["positional_metadata"]["misaligned_rows"] == 2
    assert result["positional_metadata"]["eligible_indices"] == [0, 1]


def test_selection_is_order_invariant_but_archive_last_input_deduplication_is_not():
    _, data = example.load_modules(True)
    rows = example.fixture()
    reverse = list(reversed(rows))
    a = data.clean_ohlcv(example.frame(example.historical_records(rows)))
    b = data.clean_ohlcv(example.frame(example.historical_records(reverse)))
    assert a.equals(b)
    naive_a = data.clean_ohlcv(example.frame(rows))
    naive_b = data.clean_ohlcv(example.frame(reverse))
    assert naive_a["close"].to_list() == [100.0, 20.0]
    assert naive_b["close"].to_list() == [10.0, 20.0]


def test_join_is_by_unique_key_after_actual_sort_and_null_drop():
    _, data = example.load_modules(True)
    selected = example.historical_records(example.fixture())
    cleaned = data.clean_ohlcv(example.frame(selected))
    aligned = example.align_cleaned(cleaned, list(reversed(selected)))
    assert [r["record_id"] for r in aligned] == ["1-initial", "2-initial"]
    with pytest.raises(ValueError, match="exactly one"):
        example.align_cleaned(cleaned, selected + [selected[0]])
    with pytest.raises(ValueError, match="no exact source"):
        example.align_cleaned(cleaned, selected[:1])


def test_ambiguous_revision_publication_rejected_before_cleaning():
    rows = example.fixture()
    duplicate = copy.deepcopy(rows[1])
    duplicate.update(record_id="conflict", value=77.0)
    with pytest.raises(ValueError, match="ambiguous"):
        example.historical_records(rows + [duplicate])


@pytest.mark.parametrize("field,value", [("event_ns", 1), ("decision_ns", 0)])
def test_precision_loss_or_inconsistent_decision_rejected(field, value):
    rows = example.fixture()
    rows[1][field] = value
    with pytest.raises(ValueError):
        example.historical_records(rows)


def test_unknown_release_is_not_inferred_from_event_time():
    rows = example.fixture()
    rows[0]["available_ns"] = None
    selected = example.historical_records(rows)
    assert "2-initial" not in [r["record_id"] for r in selected]

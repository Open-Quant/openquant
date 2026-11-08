"""Typed selected-revision binding through real Polars canonicalization."""

import copy
import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("polars")
ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "provenance_cleaning_example", ROOT / "examples/provenance_cleaning.py"
)
assert spec is not None and spec.loader is not None
example = importlib.util.module_from_spec(spec)
spec.loader.exec_module(example)


def test_real_cleaner_payload_guard_and_forgery_counterexample():
    data, provenance = example.modules(True)
    result = example.run(data, provenance)
    assert result["key_only_baseline_accepts_changed_value"]
    assert result["unchanged_selected_value"] == 10.0
    assert "payload differs" in result["changed_value_rejected"]
    assert result["self_consistent_backdate_passes"]
    assert result["false_value"] == 100.0
    assert result["selected_digest"] != result["forged_digest"]


@pytest.mark.parametrize("value", [None, True, float("nan"), float("inf"), 2**53 + 1, 10**400])
def test_normalization_refuses_missing_or_lossy_numbers(value):
    with pytest.raises(ValueError):
        example.float64_payload(value)


def test_exact_large_integer_and_signed_zero_preserved():
    assert example.float64_payload(2**53)["close"] == float(2**53)
    assert example.float64_payload(-0.0)["close"].hex() == "-0x0.0p+0"


def test_latest_null_is_not_dropped_before_revision_selection():
    _, provenance = example.modules(True)
    rows = example.fixture()
    rows[1]["close"] = None
    with pytest.raises(ValueError, match="nulls are not skipped"):
        example.bind_inputs(provenance, rows)


def test_nonunique_source_key_and_timestamp_precision_fail_before_cleaning():
    _, provenance = example.modules(True)
    rows = example.fixture()
    alternate = dict(rows[0], source_id="second-source")
    with pytest.raises(ValueError, match="merges distinct"):
        example.bind_inputs(provenance, rows + [alternate])
    rows[0]["event_ns"] += 1
    with pytest.raises(ValueError, match="microsecond"):
        example.bind_inputs(provenance, rows)


def test_default_missing_rows_reject_explicit_drop_is_accounted():
    data, provenance = example.modules(True)
    selected, keys = example.bind_inputs(provenance, example.fixture())
    cleaned = data.clean_ohlcv(example.to_frame(selected, keys))
    with pytest.raises(ValueError, match="dropped"):
        example.rejoin_frame(provenance, selected, keys, cleaned.head(0))
    result = example.rejoin_frame(provenance, selected, keys, cleaned.head(0), require_all=False)
    assert result.bundles == ()
    assert result.dropped_keys == (selected.bundles[0].revision.key,)


def test_ordering_and_future_append_do_not_change_bound_historical_payload():
    data, provenance = example.modules(True)
    rows = example.fixture()
    first, keys = example.bind_inputs(provenance, rows)
    future = dict(
        rows[1], revision_id="later", available_ns=rows[1]["available_ns"] + 1, close=500.0
    )
    second, other_keys = example.bind_inputs(provenance, [future, *reversed(copy.deepcopy(rows))])
    assert first.bundles[0].digest == second.bundles[0].digest
    assert data.clean_ohlcv(example.to_frame(first, keys)).equals(
        data.clean_ohlcv(example.to_frame(second, other_keys))
    )

"""Observable cutoff and accounting controls for the synthetic quote tape."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from experiments import quote_visibility_contract as quote


def _snapshot(fixture: dict[str, Any], at: Any) -> dict[str, Any]:
    from experiments.quote_execution_evidence import snapshot

    return snapshot(fixture, at)


def _attribute(fixture: dict[str, Any], start: Any, end: Any) -> dict[str, Any]:
    from experiments.quote_execution_evidence import attribute

    return attribute(fixture, start, end)


@pytest.fixture
def tape() -> dict[str, Any]:
    return json.loads(quote.DEMO.read_text())


def test_cutoff_final_fill_is_counted_exactly_once(tape: dict[str, Any]) -> None:
    partial = _snapshot(tape, 2)
    assert partial["status"] == "partial"
    assert partial["filled_shares"] == 2
    assert partial["remaining_shares"] == 1
    assert partial["cash"] == "40491/500"
    assert partial["equity"] == "48491/500"
    assert partial["terminal_observation"] is None
    complete = _snapshot(tape, 4)
    assert complete["status"] == "completed"
    assert complete["filled_shares"] == 3
    assert complete["remaining_shares"] == 0
    assert complete["last_fill"] == {"at": 3, "shares": 1, "price": "10", "commission": "101/100"}
    assert complete["terminal_observation"] == {"kind": "completed_from_fill", "at": 3}
    assert complete["status_evidence_at"] == 3
    assert complete["last_replay_bucket_at"] == 4
    assert complete["cash"] == "17493/250"
    assert complete["synthetic_only"] is True
    assert complete["external_engine_validated"] is False


def test_deadline_and_empty_clock_progress_do_not_create_expiry(tape: dict[str, Any]) -> None:
    tape["order"]["shares"] = 4
    tape["buckets"][-1]["at"] = tape["buckets"][-1]["available_at"] = 6
    at_deadline = _snapshot(tape, 3)
    assert at_deadline["status"] == "partial"
    assert at_deadline["past_deadline_unobserved"] is False
    between = _snapshot(tape, 5)
    assert between["status"] == "partial"  # last evidenced, not currently executable
    assert between["status_evidence_at"] == 3
    assert between["past_deadline_unobserved"] is True
    assert between["terminal_observation"] is None
    after = _snapshot(tape, 6)
    assert after["status"] == "expired"
    assert after["past_deadline_unobserved"] is False
    assert after["terminal_observation"] == {"kind": "expiry_observed_in_replay", "at": 6}
    assert after["last_fill"]["at"] == 3
    assert after["remaining_shares"] == 1
    assert after["inventory"] == "3"
    assert after["cash"] == "17493/250"


def test_mark_can_arrive_in_gap_without_fill_or_expiry(tape: dict[str, Any]) -> None:
    tape["order"]["shares"] = 4
    tape["buckets"] = tape["buckets"][:2] + [tape["buckets"][-1]]
    tape["buckets"][-1]["at"] = tape["buckets"][-1]["available_at"] = 6
    before, after = _snapshot(tape, 2), _snapshot(tape, 4)
    assert before["mark"] == {"price": "8", "observed_at": 0, "available_at": 0}
    assert after["mark"] == {"price": "12", "observed_at": 2, "available_at": 3}
    assert after["last_replay_bucket_at"] == 2
    assert after["last_fill"] == before["last_fill"]
    assert after["cash"] == before["cash"]
    assert after["equity"] == "52491/500"
    assert after["terminal_observation"] is None


def test_future_funding_failure_cannot_poison_earlier_snapshot(tape: dict[str, Any]) -> None:
    tape["cash"] = "20"  # first fill costs 19.018, second would overdraw
    earlier = _snapshot(tape, 2)
    assert earlier["cash"] == "491/500"
    assert earlier["status"] == "partial"
    tape["marks"].pop()
    assert _snapshot(tape, 2) == earlier
    with pytest.raises(ValueError, match="cash shortfall at 3"):
        _snapshot(tape, 3)


def test_no_bucket_yet_is_not_order_acceptance(tape: dict[str, Any]) -> None:
    tape["order"]["decision_at"] = 2
    tape["order"]["earliest_fill_at"] = 3
    pending = _snapshot(tape, 0)
    assert pending["status"] == "pending"
    assert pending["status_evidence_at"] is None
    assert pending["last_replay_bucket_at"] is None
    declared = _snapshot(tape, 2)
    assert declared["status"] == "active"
    assert declared["status_evidence_at"] == 2
    assert declared["terminal_observation"] is None
    assert declared["filled_shares"] == 0


@pytest.mark.parametrize("bad_at", [True, 1.5, -1, 1_000_001])
def test_dark_cutoff_rejects_nontime(tape: dict[str, Any], bad_at: Any) -> None:
    with pytest.raises(ValueError, match="timestamp"):
        _snapshot(tape, bad_at)


def test_future_structural_fixture_error_is_rejected(tape: dict[str, Any]) -> None:
    tape["buckets"][-1]["bid"] = "99"
    with pytest.raises(ValueError, match="crossed bid/ask"):
        _snapshot(tape, 2)


def test_partial_sale_does_not_erase_residual_or_inventory(tape: dict[str, Any]) -> None:
    tape["inventory"] = "3"
    tape["order"].update(side="sell", limit="9")
    result = _snapshot(tape, 4)
    assert result["status"] == "expired"
    assert result["filled_shares"] == -1
    assert result["remaining_shares"] == 2
    assert result["inventory"] == "2"
    assert result["cash"] == "107991/1000"
    assert result["last_fill"]["at"] == 3


def test_whole_and_subinterval_attribution_distinguish_carry(tape: dict[str, Any]) -> None:
    whole = _attribute(tape, 0, 3)
    assert whole["status"] == "reconciled_marked"
    assert whole["equity_change"] == "1493/250"
    assert whole["terms"] == {
        "opening_inventory_mark_move": "0",
        "fills_to_end_mark": "8",
        "commission_debit": "-507/250",
    }
    sub = _attribute(tape, 2, 3)
    assert sub["equity_change"] == "899/100"
    assert sub["cash_change"] == "-1101/100"
    assert sub["fill_count"] == 1
    assert sub["terms"] == {
        "opening_inventory_mark_move": "8",
        "fills_to_end_mark": "2",
        "commission_debit": "-101/100",
    }


def test_sale_attribution_respects_signed_fill_and_residual(tape: dict[str, Any]) -> None:
    tape["inventory"] = "3"
    tape["order"].update(side="sell", limit="9")
    result = _attribute(tape, 0, 4)
    assert result["equity_change"] == "7991/1000"
    assert result["end"]["remaining_shares"] == 2
    assert result["terms"] == {
        "opening_inventory_mark_move": "12",
        "fills_to_end_mark": "-3",
        "commission_debit": "-1009/1000",
    }


def test_unvalued_endpoint_never_uses_reference_close(tape: dict[str, Any]) -> None:
    tape["marks"] = []
    unmarked = _attribute(tape, 0, 2)
    assert unmarked["status"] == "unavailable_endpoint_mark"
    assert unmarked["unavailable_endpoints"] == ["end"]
    assert unmarked["cash_change"] == "-9509/500"
    assert unmarked["interval_commissions"] == "509/500"
    assert unmarked["equity_change"] is None
    assert unmarked["terms"] is None
    flat = _attribute(tape, 0, 1)
    assert flat["status"] == "flat_cash_only"
    assert flat["equity_change"] == "0"
    assert flat["terms"] is None


def test_missing_opening_inventory_mark_is_not_backfilled(tape: dict[str, Any]) -> None:
    tape["inventory"] = "3"
    tape["marks"] = tape["marks"][1:]
    result = _attribute(tape, 0, 3)
    assert result["status"] == "unavailable_endpoint_mark"
    assert result["unavailable_endpoints"] == ["start"]
    assert result["end"]["equity"] == "35493/250"
    assert result["terms"] is None


def test_window_attribute_stops_at_its_end_even_if_future_fill_fails(tape: dict[str, Any]) -> None:
    tape["cash"] = "20"
    result = _attribute(tape, 0, 2)
    assert result["equity_change"] == "-1509/500"
    assert result["end"]["remaining_shares"] == 1


@pytest.mark.parametrize("start,end", [(2, 2), (3, 2)])
def test_window_requires_strictly_increasing_endpoints(
    tape: dict[str, Any], start: int, end: int
) -> None:
    with pytest.raises(ValueError, match="start_at must precede end_at"):
        _attribute(tape, start, end)


def _cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "experiments.quote_execution_evidence", *args],
        capture_output=True,
        text=True,
        check=False,
    )


def test_cli_inspects_demo_and_rejects_duplicate_fields(tmp_path: Path) -> None:
    demo = _cli("snapshot", "2")
    assert demo.returncode == 0, demo.stderr
    assert json.loads(demo.stdout)["remaining_shares"] == 1
    assert _cli("snapshot", "2").stdout == demo.stdout
    attribution = _cli("attribute", "0", "3")
    assert attribution.returncode == 0, attribution.stderr
    assert json.loads(attribution.stdout)["equity_change"] == "1493/250"
    path = tmp_path / "duplicate.json"
    path.write_text('{"schema":1,"schema":2}')
    rejected = _cli("snapshot", "2", "--fixture", str(path))
    assert rejected.returncode != 0
    assert "duplicate JSON field" in rejected.stderr
    assert not rejected.stdout


def _benchmark(fixture: dict[str, Any], at: int) -> dict[str, Any]:
    from experiments.quote_execution_evidence import decision_benchmark

    return decision_benchmark(fixture, at)


def test_benchmark_separates_residual_from_fees_and_observed_fill_price(
    tape: dict[str, Any],
) -> None:
    tape["order"]["shares"] = 4
    result = _benchmark(tape, 3)
    assert result["status"] == "valued_paper_benchmark"
    # Paper 4*(12-8)=16; actual contributed 2*(12-9)+(12-10)-2.028 = 5.972.
    assert result["paper_incremental_marked_value"] == "16"
    assert result["actual_incremental_marked_value"] == "1493/250"
    assert result["shortfall"] == "2507/250"
    assert result["terms"] == {
        "fill_price_gap": "4",
        "residual_opportunity": "4",
        "commission_paid": "507/250",
    }
    assert result["signed_requested_shares"] == 4
    assert result["signed_residual_shares"] == 1
    assert result["snapshot"]["status"] == "partial"
    assert result["paper_funding_and_execution_assessed"] is False


def test_benchmark_sell_residual_and_execution_have_signed_benefit(tape: dict[str, Any]) -> None:
    tape["inventory"] = "3"
    tape["order"].update(side="sell", limit="9")
    result = _benchmark(tape, 4)
    assert result["signed_requested_shares"] == -3
    assert result["signed_residual_shares"] == -2
    assert result["paper_incremental_marked_value"] == "-12"
    assert result["actual_incremental_marked_value"] == "-4009/1000"
    assert result["shortfall"] == "-7991/1000"
    assert result["terms"] == {
        "fill_price_gap": "-1",
        "residual_opportunity": "-8",
        "commission_paid": "1009/1000",
    }
    assert result["snapshot"]["inventory"] == "2"


def test_benchmark_uses_independent_available_decision_mark(tape: dict[str, Any]) -> None:
    tape["marks"][0]["available_at"] = 1  # observable later, not at decision
    result = _benchmark(tape, 3)
    assert result["status"] == "unavailable_decision_mark"
    assert result["decision_mark"] is None
    assert result["snapshot"]["mark"]["price"] == "12"
    assert result["snapshot"]["commission_paid"] == "507/250"
    assert result["terms"] is None
    assert result["shortfall"] is None
    assert result["actual_incremental_marked_value"] == "1493/250"
    assert result["paper_incremental_marked_value"] is None


def test_benchmark_does_not_assume_unfilled_shares_were_paid_or_liquidated(
    tape: dict[str, Any],
) -> None:
    tape["cash"] = "0"
    for bucket in tape["buckets"]:
        bucket["capacity"] = 0
    result = _benchmark(tape, 4)
    assert result["snapshot"]["status"] == "expired"
    assert result["snapshot"]["cash"] == "0"
    assert result["snapshot"]["inventory"] == "0"
    assert result["paper_funding_and_execution_assessed"] is False
    assert result["terms"] == {
        "fill_price_gap": "0",
        "residual_opportunity": "12",
        "commission_paid": "0",
    }
    assert result["shortfall"] == "12"


def test_benchmark_at_decision_is_zero_and_prior_cutoff_is_rejected(
    tape: dict[str, Any],
) -> None:
    tape["order"]["decision_at"] = tape["order"]["earliest_fill_at"] = 1
    with pytest.raises(ValueError, match="cutoff precedes the declared decision"):
        _benchmark(tape, 0)
    result = _benchmark(tape, 1)
    assert result["decision_mark"] == {"price": "8", "observed_at": 0, "available_at": 0}
    assert result["snapshot"]["filled_shares"] == 0
    assert result["shortfall"] == "0"


def test_benchmark_mark_only_arrival_revalues_residual_without_a_fill(
    tape: dict[str, Any],
) -> None:
    tape["order"]["shares"] = 4
    tape["buckets"].pop(2)
    previous, after = _benchmark(tape, 2), _benchmark(tape, 3)
    assert previous["terms"]["residual_opportunity"] == "0"
    assert after["terms"]["residual_opportunity"] == "8"
    assert after["terms"]["fill_price_gap"] == previous["terms"]["fill_price_gap"] == "2"
    assert after["snapshot"]["last_fill"] == previous["snapshot"]["last_fill"]
    assert after["snapshot"]["cash"] == previous["snapshot"]["cash"]


def test_benchmark_cannot_value_absent_marks_or_unexecuted_inventory(
    tape: dict[str, Any],
) -> None:
    tape["marks"] = []
    with_fills = _benchmark(tape, 3)
    assert with_fills["snapshot"]["inventory"] == "3"
    assert with_fills["shortfall"] is None
    assert with_fills["actual_incremental_marked_value"] is None
    without_fills = _benchmark(tape, 1)
    assert without_fills["actual_incremental_marked_value"] == "0"
    assert without_fills["shortfall"] is None


def test_cli_benchmark_preserves_signed_unvalued_residual() -> None:
    result = _cli("benchmark", "2")
    assert result.returncode == 0, result.stderr
    data = json.loads(result.stdout)
    assert data["signed_residual_shares"] == 1
    assert data["terms"]["residual_opportunity"] == "0"
    assert data["shortfall"] == "1509/500"

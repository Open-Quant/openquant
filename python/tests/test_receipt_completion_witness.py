"""Synthetic request completeness controls at the caller-facing entry point."""

from __future__ import annotations

from collections.abc import Iterator
from fractions import Fraction as Q

import pytest

from experiments.receipt_completion_witness import (
    MAX_DELIVERIES,
    Completion,
    RequestManifest,
    ScopedReceipt,
    assess,
)
from experiments.receipt_order_conformance import Execution, Receipt


def _valid() -> tuple[list[Execution], list[Receipt]]:
    fills = [
        Execution("buy", 1, 4, Q(2), Q(10), Q(1)),
        Execution("sell", 1, 5, Q(-1), Q(9), Q(1, 2)),
        Execution("buy", 2, 7, Q(1), Q(11), Q(1, 2)),
    ]
    return fills, [Receipt(i, f"r{i}", x) for i, x in enumerate(fills, 1)]


def test_completed_request_counts_distinct_receipts_not_delivery_attempts() -> None:
    fills, rows = _valid()
    delivery = [ScopedReceipt("A", rows[i]) for i in (2, 0, 1, 0, 2)]
    result = assess(RequestManifest("A", 3), fills, delivery, completion=Completion("A", 3))
    assert result["status"] == "matched_completed_synthetic_request"
    assert result["observed_count"] == 3
    assert result["redeliveries"] == 2
    comparison = result["comparison"]
    assert comparison is not None
    assert comparison["observed_account"] == ("2", "-24", "2")
    assert not result["terminal_authority"]
    assert not result["external_engine_validated"]


@pytest.mark.parametrize("manifest_count", [2, 3])
def test_agreeing_short_batches_without_completion_never_match(manifest_count: int) -> None:
    fills, rows = _valid()
    result = assess(
        RequestManifest("A", manifest_count), fills[:2], (ScopedReceipt("A", x) for x in rows[:2])
    )
    assert result["status"] == "incomplete_missing_completion"
    assert result["observed_count"] == 2
    assert result["comparison"] is None


@pytest.mark.parametrize(
    "declared_count,differences", [(3, ["observed"]), (2, ["completion", "observed"])]
)
def test_completion_does_not_excuse_short_count(
    declared_count: int, differences: list[str]
) -> None:
    fills, rows = _valid()
    result = assess(
        RequestManifest("A", 3),
        fills,
        (ScopedReceipt("A", x) for x in rows[:2]),
        completion=Completion("A", declared_count),
    )
    assert result["status"] == "count_mismatch"
    assert result["count_differences"] == differences
    assert result["comparison"] is None


def test_completion_must_bind_the_exact_request() -> None:
    fills, rows = _valid()
    result = assess(
        RequestManifest("A", 3),
        fills,
        (ScopedReceipt("A", x) for x in rows),
        completion=Completion("a", 3),
    )
    assert result["status"] == "request_scope_mismatch"
    assert result["comparison"] is None


def test_each_receipt_must_bind_the_exact_request() -> None:
    fills, rows = _valid()
    result = assess(
        RequestManifest("A", 3),
        fills,
        [ScopedReceipt("A", rows[0]), ScopedReceipt("B", rows[1]), ScopedReceipt("A", rows[2])],
        completion=Completion("A", 3),
    )
    assert result["status"] == "request_scope_mismatch"
    assert result["comparison"] is None


def test_explicit_zero_differs_from_missing_witness() -> None:
    empty = RequestManifest("A", 0)
    assert assess(empty, [], [])["status"] == "incomplete_missing_completion"
    result = assess(empty, [], [], completion=Completion("A", 0))
    assert result["status"] == "matched_completed_synthetic_request"
    assert result["observed_count"] == 0
    assert not result["terminal_authority"]


def test_extra_sequence_is_not_removed_by_cutoff() -> None:
    fills, rows = _valid()
    changed = [*rows[:2], Receipt(4, "r4", fills[2])]
    result = assess(
        RequestManifest("A", 3),
        fills,
        (ScopedReceipt("A", x) for x in changed),
        completion=Completion("A", 3),
    )
    assert result["status"] == "sequence_mismatch"
    assert result["comparison"] is None


def test_reference_count_is_not_inferred_from_observed_rows() -> None:
    fills, rows = _valid()
    result = assess(
        RequestManifest("A", 3),
        fills[:2],
        (ScopedReceipt("A", x) for x in rows),
        completion=Completion("A", 3),
    )
    assert result["status"] == "count_mismatch"
    assert result["count_differences"] == ["reference"]


def test_matching_witness_does_not_excuse_wrong_economic_fill() -> None:
    fills, rows = _valid()
    changed = Execution("buy", 2, 8, Q(1), Q(11), Q(1, 2))
    observed = [*rows[:2], Receipt(3, "r3", changed)]
    result = assess(
        RequestManifest("A", 3),
        fills,
        (ScopedReceipt("A", x) for x in observed),
        completion=Completion("A", 3),
    )
    assert result["status"] == "execution_mismatch"
    comparison = result["comparison"]
    assert comparison is not None
    assert comparison["changed_executions"] == [("buy", 2)]
    assert comparison["account_equal"] is True


def test_conflicting_reuse_of_source_sequence_is_invalid_evidence() -> None:
    fills, rows = _valid()
    conflicted = [*rows, Receipt(3, "other", fills[2])]
    with pytest.raises(ValueError, match="conflicting receipt"):
        assess(
            RequestManifest("A", 3),
            fills,
            (ScopedReceipt("A", x) for x in conflicted),
            completion=Completion("A", 3),
        )


def test_raw_delivery_work_is_bounded_even_when_every_extra_row_is_a_retry() -> None:
    fills, rows = _valid()
    pulls = 0

    def infinite() -> Iterator[ScopedReceipt]:
        nonlocal pulls
        while True:
            pulls += 1
            if pulls > MAX_DELIVERIES + 1:
                pytest.fail("read beyond the work budget plus one lookahead")
            yield ScopedReceipt("A", rows[0])

    with pytest.raises(ValueError, match="delivery limit"):
        assess(RequestManifest("A", 1), fills[:1], infinite())
    assert pulls == MAX_DELIVERIES + 1


def test_agreement_of_every_in_band_count_cannot_replace_upstream_manifest() -> None:
    fills, rows = _valid()
    result = assess(
        RequestManifest("A", 3),
        fills[:2],
        (ScopedReceipt("A", x) for x in rows[:2]),
        completion=Completion("A", 2),
    )
    assert result["status"] == "count_mismatch"
    assert result["count_differences"] == ["completion", "observed", "reference"]
    assert result["comparison"] is None


def test_confirmed_empty_request_with_extra_receipt_is_not_empty() -> None:
    _, rows = _valid()
    result = assess(
        RequestManifest("A", 0), [], [ScopedReceipt("A", rows[0])], completion=Completion("A", 0)
    )
    assert result["status"] == "count_mismatch"
    assert result["count_differences"] == ["observed"]

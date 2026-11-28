"""Bounded synthetic completion evidence, separate from the fill comparator."""

from collections.abc import Iterable
from dataclasses import dataclass
from itertools import islice
from typing import TypedDict

from experiments.receipt_order_conformance import Comparison, Execution, Receipt, compare

MAX_RECEIPTS = 256
MAX_DELIVERIES = 1024


def _validate_request_id(value: str) -> None:
    if not isinstance(value, str) or not value.strip() or len(value) > 64:
        raise ValueError("request ID must be nonempty and at most 64 characters")


def _validate_count(value: int) -> None:
    if type(value) is not int or not 0 <= value <= MAX_RECEIPTS:
        raise ValueError("receipt count must be a bounded nonnegative integer")


@dataclass(frozen=True)
class RequestManifest:
    """Independent expected count, not inferred from either compared tape."""

    request_id: str
    expected_count: int

    def __post_init__(self) -> None:
        _validate_request_id(self.request_id)
        _validate_count(self.expected_count)


@dataclass(frozen=True)
class ScopedReceipt:
    request_id: str
    receipt: Receipt

    def __post_init__(self) -> None:
        _validate_request_id(self.request_id)
        if not isinstance(self.receipt, Receipt) or not isinstance(
            self.receipt.execution, Execution
        ):
            raise ValueError("scoped receipt must contain an execution receipt")


@dataclass(frozen=True)
class Completion:
    """Declared end of this synthetic request, not order terminality."""

    request_id: str
    receipt_count: int

    def __post_init__(self) -> None:
        _validate_request_id(self.request_id)
        _validate_count(self.receipt_count)


class WitnessReport(TypedDict):
    schema: str
    request_id: str
    status: str
    expected_count: int
    reference_count: int
    completion_count: int | None
    observed_count: int | None
    redeliveries: int | None
    count_differences: list[str]
    comparison: Comparison | None
    terminal_authority: bool
    external_engine_validated: bool


def _unique(rows: list[ScopedReceipt]) -> dict[int, Receipt]:
    """Count only exact retries once; conflicting reuse is invalid evidence."""
    by_sequence: dict[int, Receipt] = {}
    by_identity: dict[str, Receipt] = {}
    keys: set[tuple[str, int]] = set()
    for scoped in rows:
        row = scoped.receipt
        if by_sequence.get(row.sequence) == row:
            continue
        if row.sequence in by_sequence or row.identity in by_identity:
            raise ValueError("conflicting receipt identity or source sequence")
        if row.execution.key in keys:
            raise ValueError("duplicate order-local execution ordinal")
        by_sequence[row.sequence] = by_identity[row.identity] = row
        keys.add(row.execution.key)
    return by_sequence


def assess(
    manifest: RequestManifest,
    expected: Iterable[Execution],
    receipts: Iterable[ScopedReceipt],
    *,
    completion: Completion | None = None,
) -> WitnessReport:
    """Require independent request scope, count and end marker before matching.

    A report assesses a finite supplied synthetic batch, not asynchronous
    delivery, request-ID freshness, feed authenticity or order terminality.
    Observed sequences must be contiguous and request-local; the reference
    names the same fills by order and execution ordinal, not source sequence.
    """
    if not isinstance(manifest, RequestManifest):
        raise ValueError("request manifest required")
    if completion is not None and not isinstance(completion, Completion):
        raise ValueError("invalid completion record")
    reference = list(islice(expected, MAX_RECEIPTS + 1))
    if len(reference) > MAX_RECEIPTS:
        raise ValueError("reference limit exceeded")
    if any(not isinstance(row, Execution) for row in reference):
        raise ValueError("reference must contain executions")
    deliveries = list(islice(receipts, MAX_DELIVERIES + 1))
    if len(deliveries) > MAX_DELIVERIES:
        raise ValueError("delivery limit exceeded")
    if any(not isinstance(row, ScopedReceipt) for row in deliveries):
        raise ValueError("observed rows require request-scoped receipts")

    report: WitnessReport = {
        "schema": "openquant_receipt_completion_witness_v1",
        "request_id": manifest.request_id,
        "status": "request_scope_mismatch",
        "expected_count": manifest.expected_count,
        "reference_count": len(reference),
        "completion_count": completion.receipt_count if completion is not None else None,
        "observed_count": None,
        "redeliveries": None,
        "count_differences": [],
        "comparison": None,
        "terminal_authority": False,
        "external_engine_validated": False,
    }
    if (completion is not None and completion.request_id != manifest.request_id) or any(
        row.request_id != manifest.request_id for row in deliveries
    ):
        return report

    distinct = _unique(deliveries)
    report["observed_count"] = len(distinct)
    report["redeliveries"] = len(deliveries) - len(distinct)
    if completion is None:
        report["status"] = "incomplete_missing_completion"
        return report

    required = manifest.expected_count
    differences = [
        name
        for name, value in (
            ("completion", completion.receipt_count),
            ("observed", len(distinct)),
            ("reference", len(reference)),
        )
        if value != required
    ]
    if differences:
        report["status"] = "count_mismatch"
        report["count_differences"] = differences
        return report
    if any(sequence > required for sequence in distinct):
        report["status"] = "sequence_mismatch"
        return report

    comparison = compare(reference, (row.receipt for row in deliveries), through=required)
    report["comparison"] = comparison
    report["status"] = (
        "matched_completed_synthetic_request"
        if comparison["status"] == "matched_declared_synthetic_tape"
        else "execution_mismatch"
    )
    return report

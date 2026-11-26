"""Content-bound synthetic receipt comparison, not source authentication."""

import hashlib
import re
import struct
from collections.abc import Iterable
from dataclasses import dataclass
from fractions import Fraction
from itertools import islice
from typing import TypedDict

from experiments.receipt_completion_witness import (
    MAX_DELIVERIES,
    MAX_RECEIPTS,
    Completion,
    RequestManifest,
    ScopedReceipt,
    WitnessReport,
    _unique,
    _validate_request_id,
    assess,
)
from experiments.receipt_order_conformance import Execution, Receipt

SCHEMA = "openquant_receipt_content_commitment_v1"
_DOMAIN = b"openquant:synthetic-receipt-content\x00v1\x00"
_MAX_TEXT_BYTES = 256
_MAX_INTEGER_BITS = 128
_MAX_DENOMINATOR_BITS = 16


@dataclass(frozen=True)
class RequestScope:
    """One caller-selected request manifest and generation, compared exactly."""

    manifest: RequestManifest
    generation: str

    def __post_init__(self) -> None:
        if not isinstance(self.manifest, RequestManifest):
            raise ValueError("request manifest required")
        _validate_request_id(self.generation)


@dataclass(frozen=True)
class FrozenCommitment:
    """An externally supplied digest; independence and trust are caller obligations."""

    scope: RequestScope
    sha256: str
    schema: str = SCHEMA

    def __post_init__(self) -> None:
        if not isinstance(self.scope, RequestScope):
            raise ValueError("request scope required")
        if self.schema != SCHEMA:
            raise ValueError("unsupported receipt commitment schema")
        if not isinstance(self.sha256, str) or re.fullmatch("[0-9a-f]{64}", self.sha256) is None:
            raise ValueError("commitment SHA-256 must have 64 lowercase hexadecimal digits")


@dataclass(frozen=True)
class GenerationReceipt:
    generation: str
    row: ScopedReceipt

    def __post_init__(self) -> None:
        _validate_request_id(self.generation)
        if not isinstance(self.row, ScopedReceipt):
            raise ValueError("generation receipt requires a request-scoped row")


@dataclass(frozen=True)
class GenerationCompletion:
    generation: str
    completion: Completion

    def __post_init__(self) -> None:
        _validate_request_id(self.generation)
        if not isinstance(self.completion, Completion):
            raise ValueError("generation completion requires a completion record")


class CommitmentReport(TypedDict):
    schema: str
    status: str
    witness: WitnessReport | None
    observed_sha256: str | None
    digest_matches: bool | None
    terminal_authority: bool
    external_engine_validated: bool


def _text(value: str) -> bytes:
    if len(value) > _MAX_TEXT_BYTES:
        raise ValueError("text field exceeds commitment size bound")
    try:
        encoded = value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError("commitment text must be valid UTF-8") from exc
    if len(encoded) > _MAX_TEXT_BYTES:
        raise ValueError("text field exceeds commitment size bound")
    return encoded


def _integer(value: int, *, bit_limit: int = _MAX_INTEGER_BITS) -> bytes:
    if value.bit_length() > bit_limit:
        raise ValueError("integer field exceeds commitment size bound")
    return str(value).encode("ascii")


def _fraction(value: Fraction) -> bytes:
    return (
        _integer(value.numerator)
        + b"/"
        + _integer(value.denominator, bit_limit=_MAX_DENOMINATOR_BITS)
    )


def _execution_fields(row: Execution) -> tuple[bytes, ...]:
    if not isinstance(row, Execution):
        raise ValueError("reference must contain executions")
    return (
        _text(row.order),
        _integer(row.ordinal),
        _integer(row.at),
        _fraction(row.shares),
        _fraction(row.price),
        _fraction(row.fee),
    )


def _receipt_fields(row: Receipt) -> tuple[bytes, ...]:
    return _integer(row.sequence), _text(row.identity), *_execution_fields(row.execution)


def digest(scope: RequestScope, receipts: Iterable[ScopedReceipt]) -> str:
    """Producer-side canonical digest for a full declared tape, including scope.

    Freeze the expected value before either comparator tape is produced.
    The consumer must not derive the expected digest from its own inputs.
    """
    rows = list(islice(receipts, MAX_DELIVERIES + 1))
    if len(rows) > MAX_DELIVERIES:
        raise ValueError("delivery limit exceeded")
    if any(
        not isinstance(row, ScopedReceipt) or row.request_id != scope.manifest.request_id
        for row in rows
    ):
        raise ValueError("receipt request does not match commitment scope")
    for row in rows:
        _receipt_fields(row.receipt)
    unique = _unique(rows)
    count = scope.manifest.expected_count
    if sorted(unique) != list(range(1, count + 1)):
        raise ValueError("commitment requires the full contiguous source sequence")
    sha = hashlib.sha256(_DOMAIN)

    def frame(fields: Iterable[bytes]) -> None:
        for value in fields:
            sha.update(struct.pack(">I", len(value)))
            sha.update(value)

    frame((_text(scope.manifest.request_id), _text(scope.generation), _integer(count)))
    for sequence in range(1, count + 1):
        frame(_receipt_fields(unique[sequence]))
    return sha.hexdigest()


def assess_committed(
    scope: RequestScope,
    expected: Iterable[Execution],
    receipts: Iterable[GenerationReceipt],
    *,
    completion: GenerationCompletion | None = None,
    commitment: FrozenCommitment | None = None,
) -> CommitmentReport:
    """Check a frozen commitment and compare economics; neither confers authority."""
    if not isinstance(scope, RequestScope):
        raise ValueError("request scope required")
    if completion is not None and not isinstance(completion, GenerationCompletion):
        raise ValueError("generation completion required")
    if commitment is not None and not isinstance(commitment, FrozenCommitment):
        raise ValueError("frozen commitment required")

    report: CommitmentReport = {
        "schema": SCHEMA,
        "status": "generation_scope_mismatch",
        "witness": None,
        "observed_sha256": None,
        "digest_matches": None,
        "terminal_authority": False,
        "external_engine_validated": False,
    }
    if (commitment is not None and commitment.scope != scope) or (
        completion is not None and completion.generation != scope.generation
    ):
        return report
    delivered = list(islice(receipts, MAX_DELIVERIES + 1))
    if len(delivered) > MAX_DELIVERIES:
        raise ValueError("delivery limit exceeded")
    if any(not isinstance(row, GenerationReceipt) for row in delivered):
        raise ValueError("observed rows require generation-scoped receipts")
    if any(row.generation != scope.generation for row in delivered):
        return report

    reference = list(islice(expected, MAX_RECEIPTS + 1))
    if len(reference) > MAX_RECEIPTS:
        raise ValueError("reference limit exceeded")
    for execution in reference:
        _execution_fields(execution)
    rows = [row.row for row in delivered]
    for scoped in rows:
        _receipt_fields(scoped.receipt)

    witness = assess(
        scope.manifest,
        reference,
        rows,
        completion=completion.completion if completion is not None else None,
    )
    report["witness"] = witness
    if witness["comparison"] is None:
        report["status"] = "witness_unqualified"
        return report
    if commitment is None:
        report["status"] = "missing_commitment"
        return report

    observed_sha = digest(scope, rows)
    report["observed_sha256"] = observed_sha
    report["digest_matches"] = observed_sha == commitment.sha256
    if not report["digest_matches"]:
        report["status"] = "content_mismatch"
    elif witness["status"] != "matched_completed_synthetic_request":
        report["status"] = "execution_mismatch"
    else:
        report["status"] = "matched_committed_synthetic_request"
    return report

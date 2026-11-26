"""Tests at the consumer interface for an independently frozen synthetic tape."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace
from fractions import Fraction as Q

import pytest

from experiments.receipt_completion_witness import (
    MAX_DELIVERIES,
    Completion,
    RequestManifest,
    ScopedReceipt,
    assess,
)
from experiments.receipt_content_commitment import (
    FrozenCommitment,
    GenerationCompletion,
    GenerationReceipt,
    RequestScope,
    assess_committed,
    digest,
)
from experiments.receipt_order_conformance import Execution, Receipt


def _fixture() -> tuple[list[Execution], list[ScopedReceipt]]:
    executions = [
        Execution("buy", 1, 3, Q(2), Q(10), Q(1, 2)),
        Execution("sale", 1, 5, Q(-1), Q(21, 2), Q(1, 4)),
        Execution("buy", 2, 7, Q(1), Q(11), Q(3, 10)),
    ]
    return executions, [
        ScopedReceipt("snapshot", Receipt(i, f"fill:{i}", execution))
        for i, execution in enumerate(executions, 1)
    ]


def test_matching_count_and_fills_can_both_be_wrong() -> None:
    expected, received = _fixture()
    manifest = RequestManifest("snapshot", 3)
    scope = RequestScope(manifest, "run-1")
    pinned = FrozenCommitment(scope, digest(scope, received))
    mutated = replace(expected[1], at=6)
    expected[1] = mutated
    received[1] = replace(received[1], receipt=replace(received[1].receipt, execution=mutated))
    completion = Completion("snapshot", 3)
    assert assess(manifest, expected, received, completion=completion)["status"] == (
        "matched_completed_synthetic_request"
    )
    result = assess_committed(
        scope,
        expected,
        [GenerationReceipt("run-1", row) for row in received],
        completion=GenerationCompletion("run-1", completion),
        commitment=pinned,
    )
    assert result["status"] == "content_mismatch"
    assert result["digest_matches"] is False
    witness = result["witness"]
    assert witness is not None
    assert witness["status"] == "matched_completed_synthetic_request"
    assert not result["terminal_authority"]


def test_delivery_reordering_and_exact_retries_do_not_change_commitment() -> None:
    fills, rows = _fixture()
    scope = RequestScope(RequestManifest("snapshot", 3), "run-1")
    hash_value = digest(scope, rows)
    replay = [rows[2], rows[1], rows[2], rows[0], rows[0]]
    assert digest(scope, replay) == hash_value
    result = assess_committed(
        scope,
        reversed(fills),
        (GenerationReceipt("run-1", row) for row in replay),
        completion=GenerationCompletion("run-1", Completion("snapshot", 3)),
        commitment=FrozenCommitment(scope, hash_value),
    )
    assert result["status"] == "matched_committed_synthetic_request"
    assert result["digest_matches"] is True
    assert result["observed_sha256"] == hash_value
    witness = result["witness"]
    assert witness is not None and witness["redeliveries"] == 2
    assert not result["terminal_authority"] and not result["external_engine_validated"]


@pytest.mark.parametrize("stale", ["commitment", "completion", "row"])
def test_request_id_reuse_is_scoped_by_generation(stale: str) -> None:
    fills, rows = _fixture()
    m = RequestManifest("snapshot", 3)
    new, old = RequestScope(m, "new"), RequestScope(m, "old")
    assert digest(new, rows) != digest(old, rows)
    pinned = old if stale == "commitment" else new
    observed = [
        GenerationReceipt("old" if stale == "row" and i == 1 else "new", row)
        for i, row in enumerate(rows)
    ]
    result = assess_committed(
        new,
        fills,
        observed,
        completion=GenerationCompletion(
            "old" if stale == "completion" else "new", Completion("snapshot", 3)
        ),
        commitment=FrozenCommitment(pinned, digest(pinned, rows)),
    )
    assert result["status"] == "generation_scope_mismatch"
    assert result["observed_sha256"] is None and result["witness"] is None


def test_explicit_zero_requires_both_end_marker_and_empty_commitment() -> None:
    scope = RequestScope(RequestManifest("empty", 0), "run-1")
    completion = GenerationCompletion("run-1", Completion("empty", 0))
    pinned = FrozenCommitment(scope, digest(scope, []))
    assert assess_committed(scope, [], [], commitment=pinned)["status"] == "witness_unqualified"
    assert assess_committed(scope, [], [], completion=completion)["status"] == "missing_commitment"
    wrong = FrozenCommitment(scope, "0" * 64)
    assert assess_committed(scope, [], [], completion=completion, commitment=wrong)["status"] == (
        "content_mismatch"
    )
    assert assess_committed(scope, [], [], completion=completion, commitment=pinned)["status"] == (
        "matched_committed_synthetic_request"
    )


def test_verified_observed_content_does_not_excuse_wrong_reference() -> None:
    fills, rows = _fixture()
    scope = RequestScope(RequestManifest("snapshot", 3), "run-1")
    pinned = FrozenCommitment(scope, digest(scope, rows))
    fills[0] = replace(fills[0], fee=Q(9))
    result = assess_committed(
        scope,
        fills,
        [GenerationReceipt("run-1", row) for row in rows],
        completion=GenerationCompletion("run-1", Completion("snapshot", 3)),
        commitment=pinned,
    )
    assert result["status"] == "execution_mismatch"
    assert result["digest_matches"] is True


def test_receipt_id_and_source_sequence_are_independent_of_economics() -> None:
    _, rows = _fixture()
    scope = RequestScope(RequestManifest("snapshot", 3), "run-1")
    original = digest(scope, rows)
    swapped = [
        replace(rows[1], receipt=replace(rows[1].receipt, sequence=1)),
        replace(rows[0], receipt=replace(rows[0].receipt, sequence=2)),
        rows[2],
    ]
    renamed = [replace(rows[0], receipt=replace(rows[0].receipt, identity="alias")), *rows[1:]]
    assert digest(scope, swapped) != original
    assert digest(scope, renamed) != original


def test_version_and_digest_validation_are_not_implicit_coercions() -> None:
    scope = RequestScope(RequestManifest("snapshot", 3), "run-1")
    with pytest.raises(ValueError, match="schema"):
        FrozenCommitment(scope, "a" * 64, schema="v2")
    for bad in ("A" * 64, "f" * 63, "x" * 64):
        with pytest.raises(ValueError, match="SHA-256"):
            FrozenCommitment(scope, bad)


def test_producer_will_not_freeze_a_partial_or_foreign_tape() -> None:
    _, rows = _fixture()
    scope = RequestScope(RequestManifest("snapshot", 3), "run-1")
    with pytest.raises(ValueError, match="sequence"):
        digest(scope, rows[:2])
    with pytest.raises(ValueError, match="request"):
        digest(scope, [replace(rows[0], request_id="foreign"), *rows[1:]])


def test_commitment_payload_arithmetic_is_bounded_before_decimal_conversion() -> None:
    _, rows = _fixture()
    scope = RequestScope(RequestManifest("snapshot", 3), "run-1")
    huge = replace(rows[0].receipt.execution, price=Q(10**5000))
    bad = replace(rows[0], receipt=replace(rows[0].receipt, execution=huge))
    with pytest.raises(ValueError, match="integer field exceeds"):
        digest(scope, [bad, *rows[1:]])


def test_consumer_bounds_raw_retry_discovery_and_processing() -> None:
    fills, rows = _fixture()
    scope = RequestScope(RequestManifest("snapshot", 3), "run-1")
    pulls = 0

    def retries() -> Iterator[GenerationReceipt]:
        nonlocal pulls
        while True:
            pulls += 1
            if pulls > MAX_DELIVERIES + 1:
                pytest.fail("read beyond the request bound")
            yield GenerationReceipt("run-1", rows[0])

    with pytest.raises(ValueError, match="delivery limit"):
        assess_committed(scope, fills, retries())
    assert pulls == MAX_DELIVERIES + 1


def test_independent_wire_vector_binds_utf8_and_exact_fractions() -> None:
    request = "Ré/"
    generation = "g雪1"
    fill1 = Execution("O二", 1, 17, Q(3, 2), Q(25, 4), Q(1, 10))
    fill2 = Execution("O二", 2, 20, Q(-1, 3), Q(7), Q(0))
    rows = [
        ScopedReceipt(request, Receipt(1, "r/1", fill1)),
        ScopedReceipt(request, Receipt(2, "r/2", fill2)),
    ]
    assert digest(RequestScope(RequestManifest(request, 2), generation), rows) == (
        "6c0da19257f9c9b6b69cd9e1bc0701c7ed4cbac93e3c4f3dd4a39750102a6b5f"
    )
    assert digest(RequestScope(RequestManifest(request, 0), generation), []) == (
        "4dcc242a71e5b83bddc839024ed332edc9c1a22297ac69c046443b44ec8c1395"
    )


def test_missing_tail_does_not_produce_observed_content_hash() -> None:
    fills, rows = _fixture()
    scope = RequestScope(RequestManifest("snapshot", 3), "run-1")
    report = assess_committed(
        scope,
        fills,
        [GenerationReceipt("run-1", row) for row in rows[:2]],
        completion=GenerationCompletion("run-1", Completion("snapshot", 3)),
        commitment=FrozenCommitment(scope, digest(scope, rows)),
    )
    assert report["status"] == "witness_unqualified"
    assert report["observed_sha256"] is None and report["digest_matches"] is None
    witness = report["witness"]
    assert witness is not None and witness["count_differences"] == ["observed"]


def test_consumer_bounds_both_tapes_before_old_exact_arithmetic() -> None:
    fills, rows = _fixture()
    scope = RequestScope(RequestManifest("snapshot", 3), "run-1")
    commitment = FrozenCommitment(scope, digest(scope, rows))
    fills[0] = replace(fills[0], fee=Q(1, 65537))
    with pytest.raises(ValueError, match="integer field exceeds"):
        assess_committed(
            scope,
            fills,
            (GenerationReceipt("run-1", row) for row in rows),
            completion=GenerationCompletion("run-1", Completion("snapshot", 3)),
            commitment=commitment,
        )

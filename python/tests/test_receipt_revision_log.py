from dataclasses import replace
from fractions import Fraction as Q
from itertools import permutations

import pytest

from experiments.receipt_completion_witness import RequestManifest
from experiments.receipt_content_commitment import RequestScope
from experiments.receipt_order_conformance import Execution
from experiments.receipt_revision_log import Bust, Correct, Event, Fill, Tape, freeze, replay


def scope(count=3, generation="gen-1"):
    return RequestScope(RequestManifest("request-1", count), generation)


def execution(quantity=2, price=10, fee=Q(1, 2), at=12, order="O", ordinal=1):
    return Execution(order, ordinal, at, Q(quantity), Q(price), fee)


FILL = Fill(1, "r1", execution())
CORRECT = Correct(2, "r2", "r1", execution(Q(3, 2), 12, Q(1, 4)))
BUST = Bust(3, "r3", "r2")
ROWS: tuple[Event, ...] = (FILL, CORRECT, BUST)


def test_as_of_exact_economics_and_deltas():
    pin = freeze(scope(), ROWS)
    tape = replay(scope(), ROWS, pin)
    assert isinstance(tape, Tape)
    assert tape.as_of(0) == (0, 0, 0)
    assert tape.as_of(1) == (2, Q(-41, 2), Q(1, 2))
    assert tape.as_of(2) == (Q(3, 2), Q(-73, 4), Q(1, 4))
    assert tape.as_of(3) == (0, 0, 0)
    assert tape.delta_at(2) == (Q(-1, 2), Q(9, 4), Q(-1, 4))
    assert tape.delta_at(3) == (Q(-3, 2), Q(73, 4), Q(-1, 4))
    assert not tape.terminal_authority
    assert not tape.external_engine_validated


def test_known_by_and_effective_at_are_independent():
    append = Fill(4, "r4", execution(-1, 20, Q(1), at=10, ordinal=2))
    rows = (*ROWS, append)
    tape = replay(scope(4), rows, freeze(scope(4), rows))
    assert tape is not None
    assert tape.as_of(1, effective_through=11) == (0, 0, 0)
    assert tape.as_of(1, effective_through=12) == (2, Q(-41, 2), Q(1, 2))
    assert tape.as_of(3, effective_through=11) == (0, 0, 0)
    assert tape.as_of(4, effective_through=11) == (-1, 19, 1)
    assert tape.as_of(4, effective_through=9) == (0, 0, 0)
    assert tape.as_of(2, effective_through=12) == (Q(3, 2), Q(-73, 4), Q(1, 4))


def test_delivery_permutations_and_exact_retries():
    pin = freeze(scope(), ROWS)
    for rows in permutations(ROWS):
        tape = replay(scope(), (*rows, rows[0]), pin)
        assert tape is not None
        assert tape.redeliveries == 1
        assert tape.events == ROWS


@pytest.mark.parametrize(
    "rows",
    [
        (FILL, BUST),
        (FILL, CORRECT, replace(BUST, target="r1")),
        (FILL, CORRECT, replace(BUST, identity="r2")),
        (FILL, CORRECT, replace(BUST, sequence=2)),
        (FILL, replace(CORRECT, target="future"), BUST),
        (FILL, replace(CORRECT, execution=execution(order="OTHER")), BUST),
        (FILL, replace(CORRECT, execution=execution(at=13)), BUST),
        (FILL, replace(CORRECT, execution=execution(-2)), BUST),
        (FILL, CORRECT, BUST, Fill(4, "r4", execution())),
        (FILL, CORRECT, BUST, Correct(4, "r4", "r3", execution())),
    ],
)
def test_invalid_log_never_qualifies(rows):
    with pytest.raises(ValueError):
        freeze(scope(4 if len(rows) == 4 else 3), rows)


def test_same_final_economics_different_histories_fail_commitment():
    altered = (replace(FILL, execution=execution(price=9)), CORRECT, BUST)
    assert freeze(scope(), altered).sha256 != freeze(scope(), ROWS).sha256
    assert replay(scope(), altered, freeze(scope(), ROWS)) is None
    a = replay(scope(), altered, freeze(scope(), altered))
    b = replay(scope(), ROWS, freeze(scope(), ROWS))
    assert a is not None and b is not None
    assert a.as_of(3) == b.as_of(3)
    assert a.as_of(1) != b.as_of(1)


def test_scope_and_explicit_empty():
    pin = freeze(scope(), ROWS)
    assert replay(scope(generation="gen-2"), ROWS, pin) is None
    empty = replay(scope(0), [], freeze(scope(0), []))
    assert empty is not None and empty.as_of(0) == (0, 0, 0)
    with pytest.raises(ValueError):
        empty.as_of(1)


def test_prefix_pin_is_immutable_across_later_arrival():
    early = replay(scope(1), (FILL,), freeze(scope(1), (FILL,)))
    late = replay(scope(), ROWS, freeze(scope(), ROWS))
    assert early is not None and late is not None
    assert early.as_of(1) == late.as_of(1)
    with pytest.raises(ValueError):
        early.as_of(2)


def test_repeated_chain_and_bust_are_exact():
    corrected_twice = Correct(3, "r3", "r2", execution(1, 9, Q(1, 8)))
    revised = (FILL, CORRECT, corrected_twice, Bust(4, "r4", "r3"))
    tape = replay(scope(4), revised, freeze(scope(4), revised))
    assert tape is not None
    assert tape.as_of(3) == (1, Q(-73, 8), Q(1, 8))
    assert tape.as_of(4) == (0, 0, 0)


def test_individually_qualified_tapes_can_disagree_at_the_old_prefix():
    early = replay(scope(1), (FILL,), freeze(scope(1), (FILL,)))
    future = replay(scope(), ROWS, freeze(scope(), ROWS))
    altered = (replace(FILL, execution=execution(price=9)), CORRECT, BUST)
    rewritten = replay(scope(), altered, freeze(scope(), altered))
    new_gen = scope(generation="gen-2")
    relabeled = replay(new_gen, ROWS, freeze(new_gen, ROWS))
    assert early and future and rewritten and relabeled
    assert future.extends(early)
    assert future.extends(future)
    assert not early.extends(future)
    assert not rewritten.extends(early)
    assert not relabeled.extends(early)


def test_literal_serialization_vectors():
    s = RequestScope(RequestManifest("réq", 3), "g")
    rows = (
        Fill(1, "α", Execution("Ô", 5, 0, Q(-2, 3), Q(7, 2), Q(0))),
        Correct(2, "β", "α", Execution("Ô", 5, 0, Q(-1), Q(4), Q(1, 5))),
        Bust(3, "γ", "β"),
    )
    assert (
        freeze(s, rows).sha256 == "0a169ee54dc383209d6f4d2dff14c87c4d61e78cbe25602ddc790cf8439fef2e"
    )
    assert freeze(RequestScope(RequestManifest("réq", 0), "g"), ()).sha256 == (
        "925adac6c40a440f54ff87141c41833f5552001767fce864888bf92a32fdcac1"
    )

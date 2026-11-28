from dataclasses import replace
from fractions import Fraction as Q

import pytest

from experiments.receipt_availability import Observation, freeze_observer, qualify
from experiments.receipt_completion_witness import RequestManifest
from experiments.receipt_content_commitment import RequestScope
from experiments.receipt_order_conformance import Execution
from experiments.receipt_revision_log import Bust, Correct, Fill, freeze, replay

FILL = Fill(1, "e1", Execution("order-A", 1, 10, Q(2), Q(10), Q(1, 2)))
CORRECT = Correct(2, "e2", "e1", Execution("order-A", 1, 10, Q(3, 2), Q(12), Q(1, 4)))
BUST = Bust(3, "e3", "e2")
ROWS = (FILL, CORRECT, BUST)
OBS = (Observation("e1", 20), Observation("e2", 40), Observation("e3", 45))


def tape(rows):
    scope = RequestScope(RequestManifest("demo", len(rows)), "source-run-1")
    accepted = replay(scope, rows, freeze(scope, rows))
    assert accepted is not None
    return accepted


def checkpoint(rows, observations, through, *, observer="observer-A", start=15):
    prepared = tape(rows)
    pin = freeze_observer(prepared, observer, start, through, observations)
    approved = qualify(prepared, observations, pin)
    assert approved is not None
    return approved


def test_delayed_correction_does_not_leak_into_earlier_decision():
    snap = checkpoint(ROWS, OBS, 50)
    assert snap.at(19, effective_through=10) == (0, 0, 0)
    assert snap.at(20, effective_through=10) == (2, Q(-41, 2), Q(1, 2))
    assert snap.at(39, effective_through=10) == snap.at(20)
    assert snap.at(40, effective_through=10) == (Q(3, 2), Q(-73, 4), Q(1, 4))
    assert snap.at(45, effective_through=10) == (0, 0, 0)
    assert snap.at(40, effective_through=9) == (0, 0, 0)
    assert not snap.terminal_authority


@pytest.mark.parametrize("cut", [14, 51, True, 40.0])
def test_uncovered_or_lossy_cut_is_not_zero_activity(cut):
    snap = checkpoint(ROWS, OBS, 50)
    with pytest.raises(ValueError):
        snap.at(cut)


def test_gapped_observation_is_not_a_usable_prefix():
    obs = (Observation("e1", 20), Observation("e2", 40), Observation("e3", 30))
    snap = checkpoint(ROWS, obs, 50)
    assert snap.at(29) == (2, Q(-41, 2), Q(1, 2))
    with pytest.raises(ValueError, match="gap"):
        snap.at(30)
    assert snap.at(40) == (0, 0, 0)


def test_tied_observations_enter_together_and_input_order_is_not_identity():
    obs = (Observation("e1", 20), Observation("e2", 20), Observation("e3", 40))
    snap = checkpoint(ROWS, obs, 50)
    assert snap.at(19) == (0, 0, 0)
    assert snap.at(20) == (Q(3, 2), Q(-73, 4), Q(1, 4))
    assert qualify(tape(ROWS), [*reversed(obs), obs[0]], snap.pin) == snap


def test_observer_pin_is_independent_and_bound_to_exact_tape():
    snap = checkpoint(ROWS, OBS, 50)
    assert qualify(tape(ROWS), (Observation("e1", 21), *OBS[1:]), snap.pin) is None
    altered = (replace(FILL, execution=replace(FILL.execution, price=Q(9))), CORRECT, BUST)
    assert tape(ROWS).as_of(3) == tape(altered).as_of(3)
    assert qualify(tape(altered), OBS, snap.pin) is None
    assert qualify(tape(ROWS), OBS, replace(snap.pin, through=51)) is None
    assert qualify(tape(ROWS), OBS, replace(snap.pin, observer="observer-B")) is None


@pytest.mark.parametrize(
    "obs",
    [
        OBS[:-1],
        (*OBS[:-1], Observation("unknown", 40)),
        (*OBS, Observation("e2", 39)),
        (Observation("e1", 14), *OBS[1:]),
        (*OBS[:-1], Observation("e3", 51)),
    ],
)
def test_ambiguous_or_missing_observations_reject(obs):
    with pytest.raises(ValueError):
        freeze_observer(tape(ROWS), "observer-A", 15, 50, obs)


@pytest.mark.parametrize("tick", [-1, True, 20.0, "20", 1 << 128])
def test_ticks_are_exact_bounded_integers(tick):
    with pytest.raises(ValueError):
        Observation("e1", tick)


def test_forward_checkpoint_keeps_earlier_queries_stable():
    early = checkpoint((FILL,), OBS[:1], 30)
    late = checkpoint(ROWS, OBS, 50)
    assert late.extends(early)
    for cut in range(15, 31):
        assert early.at(cut) == late.at(cut)
    assert late.extends(late)
    assert not early.extends(late)


def test_same_ending_balance_does_not_hide_rewritten_history():
    early = checkpoint((FILL,), OBS[:1], 30)
    late = checkpoint(ROWS, OBS, 50)
    forged = (replace(FILL, execution=replace(FILL.execution, price=Q(9))), CORRECT, BUST)
    rewritten = checkpoint(forged, OBS, 50)
    assert rewritten.at(50) == late.at(50)
    assert rewritten.at(30) != early.at(30)
    assert not rewritten.extends(early)


@pytest.mark.parametrize(
    "obs, observer, start",
    [
        ((Observation("e1", 21), *OBS[1:]), "observer-A", 15),
        ((OBS[0], Observation("e2", 29), OBS[2]), "observer-A", 15),
        ((OBS[0], Observation("e2", 30), OBS[2]), "observer-A", 15),
        (OBS, "observer-B", 15),
        (OBS, "observer-A", 16),
    ],
)
def test_valid_standalone_certificates_cannot_rewrite_a_pinned_cut(obs, observer, start):
    early = checkpoint((FILL,), OBS[:1], 30)
    candidate = checkpoint(ROWS, obs, 50, observer=observer, start=start)
    assert not candidate.extends(early)


def test_empty_forward_checkpoint_neither_claims_future_nor_forbids_later_arrival():
    empty = checkpoint((), (), 19)
    late = checkpoint(ROWS, OBS, 50)
    assert late.extends(empty)
    assert late.at(19) == empty.at(19) == (0, 0, 0)
    with pytest.raises(ValueError, match="coverage"):
        empty.at(20)


def test_interval_reports_all_complete_prefixes_not_just_equal_balances():
    snap = checkpoint(ROWS, OBS, 50)
    result = snap.cut_segments(19, 46)
    assert [(s.start, s.through, s.prefix) for s in result] == [
        (19, 19, 0),
        (20, 39, 1),
        (40, 44, 2),
        (45, 46, 3),
    ]
    assert snap.at(19) == snap.at(46)  # Endpoint-only check is a false positive.
    assert snap.require_cut_prefix(20, 39) == 1
    with pytest.raises(ValueError, match="multiple source prefixes"):
        snap.require_cut_prefix(19, 46)


def test_interval_with_gaps_keeps_explicit_abstention_ranges():
    obs = (OBS[0], Observation("e2", 40), Observation("e3", 30))
    snap = checkpoint(ROWS, obs, 50)
    assert [(s.start, s.through, s.prefix) for s in snap.cut_segments(29, 40)] == [
        (29, 29, 1),
        (30, 39, None),
        (40, 40, 3),
    ]
    with pytest.raises(ValueError, match="gap"):
        snap.require_cut_prefix(29, 40)
    with pytest.raises(ValueError, match="gap"):
        snap.require_cut_prefix(31, 35)


def test_interval_treats_tied_arrivals_as_one_clock_world():
    obs = (OBS[0], Observation("e2", 20), Observation("e3", 20))
    snap = checkpoint(ROWS, obs, 50)
    assert [(s.start, s.through, s.prefix) for s in snap.cut_segments(19, 20)] == [
        (19, 19, 0),
        (20, 20, 3),
    ]


def test_interval_can_span_large_clock_coordinates_without_scanning_ticks():
    through = 1 << 100
    snap = checkpoint((FILL,), (Observation("e1", through),), through)
    assert [(s.start, s.through, s.prefix) for s in snap.cut_segments(15, through)] == [
        (15, through - 1, 0),
        (through, through, 1),
    ]


@pytest.mark.parametrize("start, through", [(14, 20), (20, 51), (21, 20), (True, 20), (20, 40.0)])
def test_invalid_interval_is_not_narrowed_into_coverage(start, through):
    with pytest.raises(ValueError):
        checkpoint(ROWS, OBS, 50).cut_segments(start, through)


def test_segments_match_exhaustive_exact_cutoffs_with_gaps_and_ties():
    from itertools import product

    for times in product(range(15, 20), repeat=len(ROWS)):
        observations = tuple(
            Observation(event_id, tick)
            for event_id, tick in zip(
                (FILL.identity, CORRECT.identity, BUST.identity), times, strict=True
            )
        )
        snap = checkpoint(ROWS, observations, 20)
        # Enumerate exact cuts independently, testing source indices rather
        # than balances so a fill and bust cannot hide different histories.
        for lo in range(15, 21):
            for hi in range(lo, 21):
                segments = snap.cut_segments(lo, hi)
                by_tick = [
                    (tick, s.prefix) for s in segments for tick in range(s.start, s.through + 1)
                ]
                expected = []
                for tick in range(lo, hi + 1):
                    seen = [i for i, time in enumerate(times, 1) if time <= tick]
                    prefix = len(seen) if seen == list(range(1, len(seen) + 1)) else None
                    expected.append((tick, prefix))
                    if prefix is None:
                        with pytest.raises(ValueError, match="gap"):
                            snap.at(tick)
                    else:
                        assert snap.at(tick) == snap.tape.as_of(prefix)
                assert by_tick == expected
                assert all(a.prefix != b.prefix for a, b in zip(segments, segments[1:]))

"""Direct-source association tests, not native package integration or authentication."""

import importlib.util
import itertools
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

PATH = Path(__file__).resolve().parents[1] / "openquant/provenance.py"
spec = importlib.util.spec_from_file_location("provenance_under_test", PATH)
assert spec is not None and spec.loader is not None
p = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = p
spec.loader.exec_module(p)


def revision(value=10.0, *, row="r", source="s", name="v1", event=1, available=2):
    return p.make_revision(row, source, name, event, available, {"close": value})


def cleaned(r, payload=None, event=None):
    return (
        r.row_id,
        r.source_id,
        r.event_ns if event is None else event,
        r.payload if payload is None else payload,
    )


def test_asof_selects_event_then_available_and_inclusive_boundaries():
    old = revision(name="old", event=1, available=8)
    new = revision(20.0, name="new", event=5, available=6)
    future = revision(30.0, name="future", event=5, available=9)
    for order in itertools.permutations([old, new, future]):
        selected = p.select_revisions(order, {("r", "s"): 8, ("missing", "s"): 8})
        assert selected.bundles[0].revision is new
        assert selected.missing_keys == (("missing", "s"),)
        assert p.select_revisions(order, {("r", "s"): 9}).bundles[0].revision is future
    assert p.select_revisions([new], {("r", "s"): 5}).missing_keys == (("r", "s"),)


def test_exact_duplicates_idempotent_but_identity_conflicts_global():
    a = revision()
    assert p.select_revisions([a, a], {a.key: 2}) == p.select_revisions([a], {a.key: 2})
    with pytest.raises(ValueError, match="conflicting content"):
        p.select_revisions([a, revision(11.0)], {})
    # Conflict is rejected even if future/unrequested; not future-append invariance.
    with pytest.raises(ValueError, match="conflicting content"):
        p.select_revisions([revision(1, available=100), revision(1.0, available=100)], {a.key: 2})


@pytest.mark.parametrize("other", [1.0, -0.0])
def test_typed_and_signedzero_conflicts_do_not_use_python_equality(other):
    first = 1 if other == 1 else 0.0
    with pytest.raises(ValueError, match="conflicting content"):
        p.select_revisions([revision(first), revision(other)], {("r", "s"): 2})
    selected = p.select_revisions([revision(first)], {("r", "s"): 2})
    with pytest.raises(ValueError, match="payload differs"):
        p.rejoin_selected(selected, [cleaned(revision(), {"close": other})])


def test_distinct_tied_revisions_reject_even_equal_values():
    a, b = revision(), revision(name="v2")
    for order in ([a, b], [b, a]):
        with pytest.raises(ValueError, match="ambiguous"):
            p.select_revisions(order, {a.key: 2})
    # Unknown records do not make a tie at an earlier winning timestamp.
    prior = revision(name="prior", event=0, available=0)
    assert p.select_revisions([prior, a, b], {a.key: 1}).bundles[0].revision is prior


@pytest.mark.parametrize("value", [None, True, float("nan"), float("inf"), "10", [10]])
def test_invalid_latest_payload_is_rejected_before_any_selection(value):
    older = revision()
    with pytest.raises(ValueError, match="payload values"):
        records = [older, revision(value, name="invalid-latest", available=100)]
        p.select_revisions(records, {older.key: 2})


@pytest.mark.parametrize("clock", [None, True, 2.0, -(2**63), 2**63, "2020-01-01T00:00:00Z"])
def test_unknown_or_implicit_clocks_fail_closed(clock):
    with pytest.raises(ValueError, match="clock"):
        revision(available=clock)
    with pytest.raises(ValueError, match="clock"):
        p.select_revisions([revision()], {("r", "s"): clock})


def test_clock_and_bundle_constructor_boundaries():
    with pytest.raises(ValueError, match="precede"):
        revision(event=2, available=1)
    with pytest.raises(ValueError, match="eligible"):
        p.SelectedRevision(revision(), 1)
    r = revision(event=-(2**63) + 1, available=2**63 - 1)
    assert p.select_revisions([r], {r.key: 2**63 - 1}).bundles[0].revision is r


def test_rejoin_reorder_explicit_drops_and_original_selection_indices():
    a, b, c = revision(row="a"), revision(row="b"), revision(row="c")
    selection = p.select_revisions([a, b, c], {a.key: 2, b.key: 2, c.key: 2})
    result = p.rejoin_selected(selection, [cleaned(c), cleaned(a)], require_all=False)
    assert result.selection_indices == (2, 0)
    assert result.bundles == (selection.bundles[2], selection.bundles[0])
    assert result.dropped_keys == (b.key,)
    with pytest.raises(ValueError, match="dropped"):
        p.rejoin_selected(selection, [cleaned(c), cleaned(a)])
    assert p.rejoin_selected(selection, [], require_all=False).dropped_keys == (a.key, b.key, c.key)


def test_same_value_wrong_event_and_nanosecond_coercion_reject():
    r = revision(event=1001, available=1001)
    selection = p.select_revisions([r], {r.key: 1001})
    with pytest.raises(ValueError, match="event differs"):
        p.rejoin_selected(selection, [cleaned(r, event=1000)])


def test_unknown_duplicate_and_missing_identity_cannot_reappear():
    r = revision()
    selected = p.select_revisions([r], {r.key: 2, ("missing", "s"): 2})
    for rows in (
        [cleaned(r), cleaned(r)],
        [cleaned(revision(row="unknown"))],
        [cleaned(revision(row="missing"))],
    ):
        with pytest.raises(ValueError, match="identity"):
            p.rejoin_selected(selected, rows)


def test_swapped_values_under_correct_ids_are_detected():
    a, b = revision(10.0, row="a"), revision(20.0, row="b")
    selected = p.select_revisions([a, b], {a.key: 2, b.key: 2})
    with pytest.raises(ValueError, match="payload differs"):
        p.rejoin_selected(selected, [cleaned(a, b.payload), cleaned(b, a.payload)])


def test_mutable_input_is_detached_and_field_order_canonical():
    values = {"volume": 2, "close": 1.0}
    r = p.make_revision("r", "s", "v", 1, 2, values)
    values["close"] = 999
    same = p.make_revision("r", "s", "v", 1, 2, [("close", 1.0), ("volume", 2)])
    assert p.SelectedRevision(r, 2).digest == p.SelectedRevision(same, 2).digest
    assert p.SelectedRevision(r, 2).digest != p.SelectedRevision(r, 3).digest
    with pytest.raises(FrozenInstanceError):
        r.payload = ()


@pytest.mark.parametrize("payload", [[("x", 1), ("x", 2)], {"": 1}, {1: 1}, {}, "x", [("x", 1, 2)]])
def test_invalid_or_duplicate_payload_fields_reject(payload):
    with pytest.raises(ValueError):
        p.make_revision("r", "s", "v", 1, 2, payload)


def test_arbitrary_integer_is_bound_exactly_and_lossy_float_rejoin_rejected():
    r = revision(2**53 + 1)
    selected = p.select_revisions([r], {r.key: 2})
    assert p.rejoin_selected(selected, [cleaned(r)]).bundles == selected.bundles
    with pytest.raises(ValueError, match="payload differs"):
        p.rejoin_selected(selected, [cleaned(r, {"close": float(2**53 + 1)})])


def test_structured_encoding_avoids_delimiter_collision_and_scopes_revision_id():
    a = revision(row="a|b", source="c", name="same")
    b = revision(row="a", source="b|c", name="same")
    selection = p.select_revisions([a, b], {a.key: 2, b.key: 2})
    assert len({bundle.digest for bundle in selection.bundles}) == 2
    assert p.rejoin_selected(selection, [cleaned(b), cleaned(a)]).selection_indices == (1, 0)


def test_self_consistent_backdating_or_rehashed_identity_is_not_authentication():
    true_revision = revision(99.0, available=12)
    assert p.select_revisions([true_revision], {true_revision.key: 2}).missing_keys == (
        true_revision.key,
    )
    forged = revision(99.0, available=1)
    forged_selection = p.select_revisions([forged], {forged.key: 2})
    assert p.rejoin_selected(forged_selection, [cleaned(forged)]).bundles
    swapped = revision(99.0, row="other", available=1)
    assert p.rejoin_selected(
        p.select_revisions([swapped], {swapped.key: 2}), [cleaned(swapped)]
    ).bundles


def test_public_selection_constructor_checks_identity_partition():
    bundle = p.SelectedRevision(revision(), 2)
    with pytest.raises(ValueError, match="duplicate"):
        p.Selection((bundle, bundle), ())
    with pytest.raises(ValueError, match="duplicate"):
        p.Selection((bundle,), (bundle.revision.key,))


def test_integer_encoding_avoids_interpreter_decimal_digit_limit():
    r = revision(1 << 20000)
    selected = p.select_revisions([r], {r.key: 2})
    assert p.rejoin_selected(selected, [cleaned(r)]).bundles == selected.bundles


def test_stale_digest_and_changed_revision_are_rejected_at_rejoin_boundary():
    r = revision()
    selected = p.select_revisions([r], {r.key: 2})
    object.__setattr__(selected.bundles[0], "digest", "stale-transfer")
    with pytest.raises(ValueError, match="digest is inconsistent"):
        p.rejoin_selected(selected, [cleaned(r)])
    selected = p.select_revisions([r], {r.key: 2})
    object.__setattr__(r, "available_ns", 1)
    with pytest.raises(ValueError, match="digest is inconsistent"):
        p.rejoin_selected(selected, [cleaned(r)])


def test_malformed_clock_after_frozen_object_bypass_is_not_accepted_at_rejoin():
    r = revision()
    selected = p.select_revisions([r], {r.key: 2})
    object.__setattr__(r, "available_ns", None)
    with pytest.raises(ValueError, match="clock"):
        p.rejoin_selected(selected, [cleaned(r)])


def test_bundle_cannot_prove_history_completeness_or_selection_execution():
    old, new = revision(), revision(20.0, name="v2", available=3)
    full = p.select_revisions([old, new], {old.key: 3})
    omitted = p.select_revisions([old], {old.key: 3})
    assert full.bundles[0].revision is new
    assert omitted.bundles[0].revision is old
    assert p.rejoin_selected(omitted, [cleaned(old)]).bundles
    manually_selected = p.Selection((p.SelectedRevision(old, 3),), ())
    assert p.rejoin_selected(manually_selected, [cleaned(old)]).bundles


@pytest.mark.parametrize("left,right", [(1, 1.0), (0.0, -0.0)])
def test_revision_equality_preserves_typed_conflicts_through_set_deduplication(left, right):
    a, b = revision(left), revision(right)
    assert a != b
    assert len({a, b}) == 2
    for candidates in ([a, b], {a, b}):
        with pytest.raises(ValueError, match="conflicting content"):
            p.select_revisions(candidates, {a.key: 2})
    assert p.SelectedRevision(a, 2) != p.SelectedRevision(b, 2)


def test_equal_revisions_have_equal_hashes_and_remain_idempotent():
    a = p.make_revision("r", "s", "v", 1, 2, {"close": 1.0, "volume": 2})
    b = p.make_revision("r", "s", "v", 1, 2, [("volume", 2), ("close", 1.0)])
    assert a == b
    assert hash(a) == hash(b)  # No assertion that unequal objects cannot collide.
    assert len({a, b}) == 1
    assert a != object()
    assert p.select_revisions({a, b}, {a.key: 2}).bundles == (p.SelectedRevision(a, 2),)
    assert p.SelectedRevision(a, 2) == p.SelectedRevision(b, 2)
    assert hash(p.SelectedRevision(a, 2)) == hash(p.SelectedRevision(b, 2))

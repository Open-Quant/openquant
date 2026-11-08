"""Isolated direct-source tests; not native installed-package qualification."""

import importlib.util
import itertools
import math
import sys
import types
from dataclasses import replace
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1] / "openquant"
PACKAGE = "feature_assembly_test_package"
package = types.ModuleType(PACKAGE)
package.__path__ = [str(ROOT)]
sys.modules[PACKAGE] = package


def load(name):
    spec = importlib.util.spec_from_file_location(f"{PACKAGE}.{name}", ROOT / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


p = load("provenance")
f = load("feature_assembly")
SPEC = (("price", "market", "close"), ("rate", "macro", "rate"))


def bundle(row, source, payload, available=5, decision=10):
    return p.SelectedRevision(p.make_revision(row, source, "v1", 1, available, payload), decision)


def selection(*bundles):
    return p.Selection(bundles, ())


def test_complete_missing_sources_and_missing_fields_count_all_requests():
    selected = selection(
        bundle("A", "market", {"close": 4}),
        bundle("A", "macro", {"rate": -0.0}, 8),
        bundle("B", "market", {"close": 6}),
        bundle("C", "market", {"close": 7}),
        bundle("C", "macro", {"other": 2}),
    )
    result = f.assemble_features(SPEC, dict.fromkeys(["A", "B", "C", "D"], 10), selected)
    assert (result.requested_count, result.complete_count, result.excluded_count) == (4, 1, 3)
    row = result.rows[0]
    assert row.values == (4, -0.0) and type(row.values[0]) is int
    assert math.copysign(1, row.values[1]) == -1
    assert row.max_available_ns == 8
    assert row.bundles == selected.bundles[:2]
    assert [(c.present, c.missing_source, c.missing_field) for c in result.coverage] == [
        (3, 1, 0),
        (1, 2, 1),
    ]
    assert [[m.reason for m in r.missing] for r in result.excluded] == [
        ["missing_source"],
        ["missing_field"],
        ["missing_source", "missing_source"],
    ]
    assert not hasattr(result.excluded[0], "max_available_ns")


def test_aliases_fields_order_and_single_source_lineage():
    b = bundle("A", "market", {"close": 1, "other": 1.0})
    spec = (("a", "market", "close"), ("b", "market", "other"), ("alias", "market", "close"))
    row = f.assemble_features(spec, {"A": 10}, selection(b)).rows[0]
    assert tuple(type(v) for v in row.values) == (int, float, int)
    assert row.bundles == (b,)
    assert row != replace(row, values=(1.0, 1, 1))
    assert len({row, replace(row, values=(1.0, 1, 1))}) == 2
    assert replace(row, values=(0.0,)) != replace(row, values=(-0.0,))


def test_source_and_request_permutations_preserve_identity():
    bundles = [
        bundle(r, s, {field: value})
        for r, value in [("A", 1), ("B", 2)]
        for s, field in [("market", "close"), ("macro", "rate")]
    ]
    for order in itertools.permutations(bundles):
        for rows in [("A", "B"), ("B", "A")]:
            result = f.assemble_features(SPEC, dict.fromkeys(rows, 10), selection(*order))
            assert tuple(r.row_id for r in result.rows) == rows
            assert [r.values for r in result.rows] == [(1, 1) if r == "A" else (2, 2) for r in rows]


def test_decision_mismatch_rejected_even_when_field_missing_or_other_source_missing():
    for payload in ({"rate": 1}, {"other": 1}):
        with pytest.raises(ValueError, match="decision differs"):
            f.assemble_features(
                SPEC, {"A": 10}, selection(bundle("A", "macro", payload, decision=11))
            )


def test_valid_unused_sources_irrelevant_but_invalid_unused_bundles_rejected():
    base = selection(bundle("A", "market", {"close": 1}), bundle("A", "macro", {"rate": 2}))
    expected = f.assemble_features(SPEC, {"A": 10}, base)
    unused = bundle("other", "unused", {"v": 3}, decision=99)
    assert f.assemble_features(SPEC, {"A": 10}, selection(*base.bundles, unused)) == expected
    object.__setattr__(unused.revision, "payload", (("v", 4),))
    with pytest.raises(ValueError, match="digest"):
        f.assemble_features(SPEC, {"A": 10}, selection(*base.bundles, unused))


def test_empty_and_zero_complete_requests_are_explicit():
    empty = f.assemble_features(SPEC, {}, selection())
    assert empty.rows == empty.excluded == ()
    assert empty.requested_count == empty.complete_count == empty.excluded_count == 0
    assert all(c.present == c.missing_field == c.missing_source == 0 for c in empty.coverage)
    missing = f.assemble_features(SPEC, {"A": 10}, selection())
    assert missing.rows == () and missing.excluded_count == 1


@pytest.mark.parametrize(
    "spec",
    [
        (),
        "abc",
        {("a", "s", "f")},
        (("a", "s"),),
        (("", "s", "f"),),
        (("a", "s", "f"), ("a", "s", "g")),
    ],
)
def test_invalid_specs(spec):
    with pytest.raises(ValueError):
        f.assemble_features(spec, {}, selection())


@pytest.mark.parametrize(
    "decisions", [[], {"": 1}, {"A": True}, {"A": 1.0}, {"A": -(2**63)}, {"A": 2**63}]
)
def test_invalid_requests(decisions):
    with pytest.raises(ValueError):
        f.assemble_features(SPEC, decisions, selection())


def test_mutated_duplicate_selection_is_revalidated():
    b = bundle("A", "market", {"close": 1})
    selected = selection(b)
    object.__setattr__(selected, "bundles", (b, b))
    with pytest.raises(ValueError):
        f.assemble_features(SPEC, {}, selected)


def test_coverage_denominator_retains_informative_exclusions():
    selected = selection(*(bundle(str(i), "market", {"close": i}) for i in range(10)))
    result = f.assemble_features(SPEC[:1], dict.fromkeys(map(str, range(16)), 10), selected)
    assert (result.complete_count, result.requested_count) == (10, 16)
    assert result.coverage[0].missing_source == 6

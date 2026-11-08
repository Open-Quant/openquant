"""Point-in-time availability, provenance and invariance contracts."""

import importlib.util
import itertools
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "point_in_time_lab", Path(__file__).resolve().parents[2] / "experiments/point_in_time_lab.py"
)
assert spec is not None and spec.loader is not None
lab = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lab)
T = datetime(2020, 1, 1, tzinfo=UTC)


def row(event=0, available=0, value=1, revision="r", entity="A", feature="x"):
    return dict(
        entity=entity,
        feature=feature,
        event_time=T + timedelta(hours=event),
        available_time=T + timedelta(hours=available),
        value=value,
        revision_id=revision,
    )


def test_both_clocks_and_inclusive_boundary():
    rows = [row(0, 1, 1, "initial"), row(0, 3, 2, "revision"), row(4, 2, 9, "future-event")]
    index = lab.PointInTimeIndex(rows)
    assert index.lookup("A", "x", T)["selected"] is None
    assert index.lookup("A", "x", T + timedelta(hours=1))["selected"]["value"] == 1
    assert index.lookup("A", "x", T + timedelta(hours=2))["selected"]["value"] == 1
    assert index.lookup("A", "x", T + timedelta(hours=3))["selected"]["value"] == 2
    assert index.lookup("A", "x", T + timedelta(hours=4))["selected"]["value"] == 9
    assert (
        index.lookup("A", "x", T + timedelta(hours=1), mode="event_only_final_revision")[
            "selected"
        ]["value"]
        == 2
    )


def test_event_precedes_revision_recency():
    index = lab.PointInTimeIndex([row(0, 5, 99, "old-event-revised"), row(2, 3, 2, "new-event")])
    assert index.lookup("A", "x", T + timedelta(hours=5))["selected"]["value"] == 2


def test_duplicates_offsets_and_permutations():
    a = row(value=-0.0, revision="z")
    b = row(value=0, revision="a")
    b["event_time"] = "2019-12-31T19:00:00-05:00"
    expected = None
    for rows in itertools.permutations([a, b, a.copy()]):
        result = lab.PointInTimeIndex(rows).lookup("A", "x", T)
        assert result["selected"]["value"] == 0.0
        assert result["selected"]["revision_ids"] == ["a", "z"]
        if expected is not None:
            assert result == expected
        expected = result


def test_future_append_invariance_including_future_conflict():
    base = [row(0, 0, 1)]
    future = [
        row(0, 5, 2, "future"),
        row(0, 5, 3, "future-conflict"),
        row(3, 0, 99, "future-event"),
    ]
    decision = T + timedelta(hours=1)
    assert lab.PointInTimeIndex(base).lookup("A", "x", decision) == lab.PointInTimeIndex(
        base + future
    ).lookup("A", "x", decision)
    with pytest.raises(ValueError, match="conflicting"):
        lab.PointInTimeIndex(base + future[:2]).lookup("A", "x", T + timedelta(hours=5))
    assert lab.PointInTimeIndex([]).lookup("B", "x", decision) == lab.PointInTimeIndex(
        [row(5, 5, entity="B")]
    ).lookup("B", "x", decision)


def test_input_mutation_cannot_rewrite_index():
    record = row()
    index = lab.PointInTimeIndex([record])
    record["value"] = 99
    selected = index.lookup("A", "x", T)
    selected["selected"]["revision_ids"].append("fake")
    assert index.lookup("A", "x", T)["selected"] == dict(
        event_time=T.isoformat(), available_time=T.isoformat(), value=1, revision_ids=["r"]
    )


@pytest.mark.parametrize(
    "timestamp",
    ["2020-01-01T00:00:00", datetime(2020, 1, 1), "2020-01-01T00:00:00.1234567Z", 123, None],
)
def test_invalid_or_unqualified_timestamp(timestamp):
    with pytest.raises((ValueError, TypeError)):
        lab.utc_timestamp(timestamp)


@pytest.mark.parametrize("value", [True, float("nan"), float("inf"), "1", 1 + 2j])
def test_invalid_values(value):
    with pytest.raises(ValueError):
        lab.PointInTimeIndex([row(value=value)])


def test_missing_and_separate_entities_features():
    index = lab.PointInTimeIndex([row(entity="B"), row(feature="y")])
    assert index.lookup("A", "x", T)["status"] == "missing"
    with pytest.raises(ValueError):
        index.lookup("A", "x", T, mode="unknown")
    with pytest.raises(ValueError):
        lab.PointInTimeIndex([dict(row(), extra=1)])


def test_synthetic_timing_and_fixed_prediction_contract():
    report = lab.run_lab(repetitions=3, days=16)
    for family, data in report["families"].items():
        safe = data["summary"]["point_in_time"]
        assert safe["future_publications_used"] == 0
        if family == "delayed_release":
            assert safe["missing"] == 3
            assert safe["older_events_used"] == 45
        if family in ("late_revision", "signal_then_revision"):
            naive = data["summary"]["event_only_final_revision"]
            assert naive["correct_fraction_all_decisions"] == 1
            assert naive["future_publications_used"] == 48
        for choice in data["representative_first_replication"]["decisions"]:
            assert lab.utc_timestamp(choice["label_available_time"]) > lab.utc_timestamp(
                choice["decision_time"]
            )
    repeat = lab.run_lab(**report["config"])
    assert repeat["families"] == report["families"]


@pytest.mark.parametrize("value", [2**53 + 1, 10**400])
def test_integer_precision_loss_fails_closed(value):
    with pytest.raises(ValueError, match="binary64"):
        lab.PointInTimeIndex([row(value=value)])
    assert (
        lab.PointInTimeIndex([row(value=2**53)]).lookup("A", "x", T)["selected"]["value"] == 2**53
    )


def test_fractional_value_precision_contract():
    from fractions import Fraction

    with pytest.raises(ValueError, match="binary64"):
        lab.PointInTimeIndex([row(value=Fraction(1, 10))])
    assert (
        lab.PointInTimeIndex([row(value=Fraction(1, 2))]).lookup("A", "x", T)["selected"]["value"]
        == 0.5
    )


@pytest.mark.parametrize(
    "timestamp",
    [
        "2020-01-01T12:00:00-00:00:00.000001",
        "2020-01-01T12:00:00+01:00:01",
        "2020-01-01T12:00:00-00:00",
        "2020-01-01T12:00Z",
    ],
)
def test_unsupported_or_unknown_offset_grammar(timestamp):
    with pytest.raises(ValueError):
        lab.utc_timestamp(timestamp)


def test_aware_datetime_fractional_offset_normalizes_exactly():
    from datetime import timezone

    stamp = datetime(2020, 1, 1, 12, tzinfo=timezone(-timedelta(microseconds=1)))
    assert lab.utc_timestamp(stamp) == T + timedelta(hours=12, microseconds=1)

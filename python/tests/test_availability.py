"""Direct-file availability contract tests; these do not validate native package import."""

import importlib.util
import itertools
import sys
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import pytest

SOURCE = Path(__file__).resolve().parents[1] / "openquant/availability.py"
spec = importlib.util.spec_from_file_location("availability_under_test", SOURCE)
assert spec is not None and spec.loader is not None
availability = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = availability
spec.loader.exec_module(availability)
select = availability.select_available_training


def test_nanosecond_boundaries_and_explicit_label_end_override():
    args = ([8, 8, 8, 10], [9, 10, 9, 10], [10, 10, 11, 10], [8, 8, 8, 10])
    strict = select(*args, fit_time=10, clock="unix_ns")
    assert strict.eligible_indices == (0,)
    assert dict(strict.exclusions) == {
        1: ("label_end_not_before_fit",),
        2: ("label_available_after_fit",),
        3: ("decision_not_before_fit", "label_end_not_before_fit"),
    }
    inclusive = select(*args, fit_time=10, clock="unix_ns", label_end_inclusive=True)
    assert inclusive.eligible_indices == (0, 1)
    assert dict(inclusive.exclusions)[3] == ("decision_not_before_fit",)


def test_historical_feature_revision_clock_not_fit_clock():
    result = select([1, 1], [2, 2], [3, 3], [1, 2], fit_time=5, clock="unix_ns")
    assert result.eligible_indices == (0,)
    assert dict(result.exclusions)[1] == ("feature_unavailable_at_decision",)


def test_missing_clocks_are_retained_as_exclusions():
    result = select(
        [0, 0, 0], [1, None, 1], [2, 2, None], [None, 0, 0], fit_time=3, clock="unix_ns"
    )
    assert result.eligible_indices == ()
    assert dict(result.exclusions) == {
        0: ("feature_availability_unknown",),
        1: ("label_end_unknown",),
        2: ("label_availability_unknown",),
    }


def test_zero_length_labels_duplicate_decisions_and_candidate_order():
    result = select(
        [1, 1, 2, 3],
        [1, 1, 2, 3],
        [1, 1, 2, 3],
        [1, 1, 2, 3],
        fit_time=4,
        clock="unix_ns",
        candidate_indices=[3, 1],
    )
    assert result.candidate_indices == (3, 1)
    assert result.eligible_indices == (3, 1)
    assert result.exclusions == ()


@pytest.mark.parametrize("indices", [[0, 0], [-1], [2], [True], [0.0], "0"])
def test_invalid_candidate_indices(indices):
    with pytest.raises((ValueError, TypeError)):
        select(
            [0, 0], [1, 1], [1, 1], [0, 0], fit_time=2, clock="unix_ns", candidate_indices=indices
        )


@pytest.mark.parametrize("decision,end,release", [(2, 1, 3), (1, 3, 2), (2, None, 1)])
def test_inconsistent_known_label_clocks_rejected(decision, end, release):
    with pytest.raises(ValueError):
        select([decision], [end], [release], [decision], fit_time=5, clock="unix_ns")


def test_noncandidate_rows_still_validated():
    with pytest.raises(ValueError):
        select([0, 2], [1, 1], [1, 3], [0, 2], fit_time=5, clock="unix_ns", candidate_indices=[0])


@pytest.mark.parametrize("value", [True, 1.0, None, -(2**63), 2**63, -(2**63) - 1])
def test_invalid_unix_ns_decision_or_fit(value):
    with pytest.raises((ValueError, TypeError)):
        select([value], [0], [0], [0], fit_time=1, clock="unix_ns")
    with pytest.raises((ValueError, TypeError)):
        select([], [], [], [], fit_time=value, clock="unix_ns")


def test_exact_int64_limits_without_float_roundtrip():
    low, high = -(2**63) + 1, 2**63 - 1
    result = select(
        [low, high - 2],
        [low, high - 1],
        [low, high],
        [low, high - 2],
        fit_time=high,
        clock="unix_ns",
    )
    assert result.eligible_indices == (0, 1)
    assert result.fit_time_ns == high


def test_utc_offset_equivalence_and_exact_microseconds():
    result = select(
        ["1969-12-31T19:00:00-05:00"],
        ["1970-01-01T00:00:00.000001Z"],
        ["1970-01-01T00:00:00.000002+00:00"],
        [datetime(1970, 1, 1, tzinfo=UTC)],
        fit_time="1970-01-01T00:00:00.000002Z",
        clock="utc",
    )
    assert result.eligible_indices == (0,)
    assert result.fit_time_ns == 2000


def test_datetime_fractional_offset_and_negative_epoch():
    fit = datetime(1970, 1, 1, tzinfo=timezone(-timedelta(microseconds=1)))
    result = select(
        [datetime(1969, 12, 31, 23, 59, 59, 999999, tzinfo=UTC)],
        [datetime(1970, 1, 1, tzinfo=UTC)],
        [fit],
        [datetime(1969, 12, 31, 23, 59, 59, 999999, tzinfo=UTC)],
        fit_time=fit,
        clock="utc",
    )
    assert result.fit_time_ns == 1000
    assert result.eligible_indices == (0,)


@pytest.mark.parametrize(
    "timestamp",
    [
        datetime(2020, 1, 1),
        "2020-01-01T00:00:00",
        "2020-01-01T00:00:00.1234567Z",
        "2020-01-01T00:00:00-00:00:00.000001",
        "2020-01-01T00:00:00-00:00",
        "2020-01-01T00:00Z",
        "9999-01-01T00:00:00Z",
        0,
    ],
)
def test_utc_rejects_implicit_or_lossy_clocks(timestamp):
    with pytest.raises((ValueError, TypeError)):
        select([], [], [], [], fit_time=timestamp, clock="utc")


def test_unsupported_datetime_subclass_rejected():
    class ExtraPrecision(datetime):
        pass

    with pytest.raises((ValueError, TypeError)):
        select([], [], [], [], fit_time=ExtraPrecision(2020, 1, 1, tzinfo=UTC), clock="utc")


def test_empty_input_and_input_mutation_isolation():
    assert select([], [], [], [], fit_time=0, clock="unix_ns").eligible_indices == ()
    decisions, candidates = [0], [0]
    result = select(
        decisions, [1], [1], [0], fit_time=2, clock="unix_ns", candidate_indices=candidates
    )
    decisions[0], candidates[0] = 99, 99
    assert result.eligible_indices == (0,)
    assert result.candidate_indices == (0,)
    with pytest.raises(AttributeError):
        result.eligible_indices = (99,)


def test_shape_and_mode_validation():
    with pytest.raises(ValueError):
        select([0], [], [1], [0], fit_time=2, clock="unix_ns")
    with pytest.raises(ValueError):
        select([], [], [], [], fit_time=0, clock="bars")
    with pytest.raises((ValueError, TypeError)):
        select([], [], [], [], fit_time=0, clock="unix_ns", label_end_inclusive=1)


def test_exhaustive_boolean_eligibility_oracle():
    # Direct predicates and candidate intersections, not implementation helpers.
    for decision, end, release, feature, fit in itertools.product(range(3), repeat=5):
        if end < decision or release < end:
            continue
        for inclusive in (False, True):
            expected = (
                decision < fit
                and feature <= decision
                and release <= fit
                and (end <= fit if inclusive else end < fit)
            )
            result = select(
                [decision],
                [end],
                [release],
                [feature],
                fit_time=fit,
                clock="unix_ns",
                label_end_inclusive=inclusive,
            )
            assert bool(result.eligible_indices) == expected


@pytest.mark.parametrize("offset", ["+00:60", "-00:60", "+05:99", "-05:99", "+24:00", "-24:00"])
def test_invalid_offset_fields_are_not_normalized(offset):
    with pytest.raises(ValueError, match="offset"):
        select([], [], [], [], fit_time="2020-01-01T00:00:00" + offset, clock="utc")


def test_extreme_valid_offset_and_noniterable_candidates():
    result = select([], [], [], [], fit_time="1970-01-02T00:00:00+23:59", clock="utc")
    assert result.fit_time_ns == 60_000_000_000
    with pytest.raises(ValueError):
        select([], [], [], [], fit_time=0, clock="unix_ns", candidate_indices=3)


def test_runnable_example_works_without_site_packages():
    import json
    import subprocess

    example = SOURCE.parents[2] / "examples/availability_workflow.py"
    completed = subprocess.run(
        [sys.executable, "-S", str(example), "--source-tree"],
        capture_output=True,
        text=True,
        check=True,
    )
    result = json.loads(completed.stdout)
    assert result["selection"]["eligible_indices"] == [0]
    assert result["training_only_center"] == 10.0
    assert result["heldout_index"] not in result["selection"]["eligible_indices"]
    assert result["import_mode"] == "isolated_source_module"

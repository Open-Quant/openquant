"""Audit supplied source-availability and label-maturity clocks before fitting.

This module is implemented using only the Python standard library. The package's
normal ``openquant`` import still initializes its native extension and other
existing dependencies; direct-file tests do not establish full-package readiness.

The helper is a conservative historical-vintage filter, not a cross-validation
splitter. It cannot reconstruct revisions, authenticate timestamps, establish
statistical independence, or replace purging, embargo or fold-local preprocessing.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from numbers import Integral
from typing import Literal, NamedTuple, cast

__all__ = ["AvailabilitySelection", "select_available_training"]

Clock = Literal["utc", "unix_ns"]
Timestamp = int | str | datetime
_MIN_NS = -(2**63) + 1  # int64 minimum is the NumPy datetime NaT sentinel.
_MAX_NS = 2**63 - 1
_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)
_ISO = re.compile(
    r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?(?:Z|[+-]\d{2}:\d{2})",
    flags=re.ASCII,
)


class AvailabilitySelection(NamedTuple):
    """Immutable original-row selection and explicit exclusion reasons.

    ``candidate_indices`` and ``eligible_indices`` preserve supplied candidate
    order. ``exclusions`` contains ``(original_index, reasons)`` tuples in that
    same relative order. Every candidate occurs in exactly one output partition.
    A row can have multiple reasons. ``fit_time_ns`` is exact UTC Unix nanoseconds.
    """

    candidate_indices: tuple[int, ...]
    eligible_indices: tuple[int, ...]
    exclusions: tuple[tuple[int, tuple[str, ...]], ...]
    fit_time_ns: int
    label_end_inclusive: bool


def _timestamp(value: object, clock: Clock, name: str) -> int:
    if clock == "unix_ns":
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise ValueError(f"{name} must be an exact integer Unix-nanosecond timestamp")
        result = int(value)
    else:
        if isinstance(value, str):
            if not _ISO.fullmatch(value) or value.endswith("-00:00"):
                raise ValueError(
                    f"{name} requires ISO seconds, at most microseconds and Z or a known ±HH:MM offset"
                )
            if not value.endswith("Z"):
                offset_hour, offset_minute = int(value[-5:-3]), int(value[-2:])
                if offset_hour >= 24 or offset_minute >= 60:
                    raise ValueError(f"{name} has an invalid UTC offset")
            try:
                value = datetime.fromisoformat(value)
            except ValueError as exc:
                raise ValueError(f"{name} is not a valid calendar timestamp") from exc
        # Pandas Timestamp and other subclasses may carry precision beyond the
        # builtin datetime fields. Require explicit exact-ns conversion instead.
        if type(value) is not datetime or value.tzinfo is None or value.utcoffset() is None:
            raise ValueError(f"{name} must be an aware builtin datetime or supported ISO string")
        try:
            delta = value.astimezone(UTC) - _EPOCH
        except (OverflowError, ValueError) as exc:
            raise ValueError(f"{name} cannot be represented in UTC") from exc
        result = ((delta.days * 86400 + delta.seconds) * 1_000_000 + delta.microseconds) * 1000
    if not _MIN_NS <= result <= _MAX_NS:
        raise ValueError(f"{name} is outside supported int64 nanoseconds or is the NaT sentinel")
    return result


def _column(values: Iterable, clock: Clock, name: str, *, missing: bool) -> tuple[int | None, ...]:
    if isinstance(values, (str, bytes, Mapping)):
        raise ValueError(f"{name} must be a one-dimensional iterable, not a string or mapping")
    try:
        items = list(values)
    except TypeError as exc:
        raise ValueError(f"{name} must be a one-dimensional iterable") from exc
    return tuple(
        None if value is None and missing else _timestamp(value, clock, f"{name}[{index}]")
        for index, value in enumerate(items)
    )


def _candidates(values: Iterable[int] | None, length: int) -> tuple[int, ...]:
    if values is None:
        return tuple(range(length))
    if isinstance(values, (str, bytes, Mapping)):
        raise ValueError("candidate_indices must be a one-dimensional iterable of integer indices")
    try:
        items = list(values)
    except TypeError as exc:
        raise ValueError("candidate_indices must be an iterable of integer indices") from exc
    result: list[int] = []
    seen: set[int] = set()
    for value in items:
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise ValueError("candidate_indices must contain integers, not booleans or floats")
        index = int(value)
        if index < 0 or index >= length or index in seen:
            raise ValueError("candidate_indices must be distinct and in range")
        seen.add(index)
        result.append(index)
    return tuple(result)


def select_available_training(
    decision_times: Iterable[Timestamp],
    label_end_times: Iterable[Timestamp | None],
    label_available_times: Iterable[Timestamp | None],
    feature_available_times: Iterable[Timestamp | None],
    *,
    fit_time: Timestamp,
    clock: Clock,
    label_end_inclusive: bool = False,
    candidate_indices: Iterable[int] | None = None,
) -> AvailabilitySelection:
    """Filter original sample indices using declared information-availability clocks.

    Parameters
    ----------
    decision_times
        Historical sample decisions. Missing decisions are invalid. Ties and
        arbitrary row order are allowed; this function does not sort or split.
    label_end_times
        End of each label's information span; ``None`` means unresolved/unknown.
        Known ends cannot precede their decisions. Zero-length spans are valid.
    label_available_times
        Actual usable publication time of the finalized label; ``None`` means
        unknown/unreleased. Known publication cannot precede decision or known
        label end. A resolution timestamp alone is not a publication timestamp.
    feature_available_times
        Latest actual source/revision availability needed for ALL exact historical
        raw/source feature values in each row, BEFORE fold-local learned transforms. ``None`` means unknown/missing information.
        This is not a batch ``computed_at`` time: deterministic reconstruction
        from past-only inputs can run later. The caller must preserve source
        metadata, align it with the retained rows, and fit learned transforms
        only within training folds. A scaler or PCA fitted at fit_time on eligible
        training rows can legitimately depend on later training rows relative to
        an individual historical decision. Do not pass that transform's fitted_at
        as source availability. Audit historical source snapshots, intersect the
        splitter's training indices, then fit learned transforms/model on those
        training rows and apply them to held-out rows. This helper inspects no
        feature values or learned preprocessing state.
    fit_time
        Cutoff at which information published at that exact time is already
        usable. Sample decisions must be strictly earlier than this cutoff.
    clock
        Required explicit clock domain. ``unix_ns`` accepts exact integer UTC
        Unix nanoseconds in int64 range, excluding the NaT sentinel. ``utc``
        accepts aware builtin datetime or extended ISO strings with seconds,
        at most six fractional digits, and ``Z`` or a known ``±HH:MM`` offset.
        Naive clocks, unknown ``-00:00``, submicrosecond strings, offset seconds
        in strings, datetime subclasses, floating epoch seconds and implicit bar
        positions are not converted. Builtin aware datetime objects may carry
        valid subminute/fractional timezone offsets, normalized exactly here.
        Use explicit integer nanoseconds for finer observation clocks.
    label_end_inclusive
        False by default: label end must be strictly before fit. This conservative
        boundary excludes a closed label span ending exactly where a test window
        starts at fit. True permits equality under an after-observation maturity
        convention. Label publication still must be at or before fit in both modes.
        Neither convention replaces purging against actual test intervals.
    candidate_indices
        Optional pre-existing splitter's train indices. The result is only a
        subset of these indices in their supplied order, never a replacement split.
        Indices must be unique and in range. None audits all rows.

    Returns
    -------
    AvailabilitySelection
        Immutable original indices and explicit reasons, suitable for indexing
        the same aligned feature/label arrays. None clocks cause row exclusions;
        malformed clocks, length mismatches and known chronology inconsistencies
        raise ValueError, including on noncandidate rows. Inputs are not mutated.

    Notes
    -----
    ``clean_ohlcv`` currently discards extra metadata and emits UTC-naive
    microseconds. Preserve availability and stable row identity separately before
    cleaning; explicitly attach UTC only when that upstream contract is verified.
    Never reconstruct source availability from event time or silently realign by
    position after cleaning dropped or sorted rows. CV's t1 is label resolution,
    and its integer input may mean bars; callers must explicitly convert actual
    UTC clocks here. Passing arbitrary integers cannot prove their physical units.
    """
    if not isinstance(clock, str) or clock not in ("utc", "unix_ns"):
        raise ValueError("clock must explicitly be 'utc' or 'unix_ns'")
    if type(label_end_inclusive) is not bool:
        raise ValueError("label_end_inclusive must be a boolean")
    cutoff = _timestamp(fit_time, clock, "fit_time")
    decisions = cast(
        tuple[int, ...], _column(decision_times, clock, "decision_times", missing=False)
    )
    ends = _column(label_end_times, clock, "label_end_times", missing=True)
    releases = _column(label_available_times, clock, "label_available_times", missing=True)
    features = _column(feature_available_times, clock, "feature_available_times", missing=True)
    length = len(decisions)
    if any(len(column) != length for column in (ends, releases, features)):
        raise ValueError("all timestamp columns must have equal length")
    candidates = _candidates(candidate_indices, length)
    for index, (decision, end, release) in enumerate(zip(decisions, ends, releases, strict=True)):
        if end is not None and end < decision:
            raise ValueError(f"label_end_times[{index}] precedes its decision")
        if release is not None and (release < decision or (end is not None and release < end)):
            raise ValueError(f"label_available_times[{index}] precedes its decision or label end")
    eligible: list[int] = []
    exclusions: list[tuple[int, tuple[str, ...]]] = []
    for index in candidates:
        decision, end, release, feature = (
            decisions[index],
            ends[index],
            releases[index],
            features[index],
        )
        reasons = []
        if decision >= cutoff:
            reasons.append("decision_not_before_fit")
        if feature is None:
            reasons.append("feature_availability_unknown")
        elif feature > decision:
            reasons.append("feature_unavailable_at_decision")
        if end is None:
            reasons.append("label_end_unknown")
        elif end > cutoff or (end == cutoff and not label_end_inclusive):
            reasons.append(
                "label_end_after_fit" if label_end_inclusive else "label_end_not_before_fit"
            )
        if release is None:
            reasons.append("label_availability_unknown")
        elif release > cutoff:
            reasons.append("label_available_after_fit")
        if reasons:
            exclusions.append((index, tuple(reasons)))
        else:
            eligible.append(index)
    return AvailabilitySelection(
        candidates, tuple(eligible), tuple(exclusions), cutoff, label_end_inclusive
    )

---
title: "availability"
description: "Filter original training indices by historical source availability and label maturity without replacing the cross-validation split."
status: authored
last_authored: '2026-11-07'
audience:
  - quant-dev
  - platform-engineering
module: "availability"
api_surface: "python-only"
citation:
  - "OpenQuant python/openquant/availability.py: clock conversion, chronology validation and candidate-index filtering."
  - "OpenQuant examples/availability_workflow.py and examples/availability_end_to_end.py: source availability, label maturity and row alignment."
python_api:
  - "availability.AvailabilitySelection"
  - "availability.select_available_training"
sidebar:
  badge: Module
---

`select_available_training` audits clocks supplied for historical samples, then returns
a subset of the original candidate indices in their supplied order. It does not inspect
feature values, reconstruct revisions, or create a cross-validation split. Use
[provenance](/modules/provenance/) to bind selected source values and
[feature assembly](/modules/feature-assembly/) to require complete declared inputs first.

## The five boundaries

For a sample to remain eligible:

- Its decision must be strictly before `fit_time`.
- The latest source availability needed for its exact historical raw features must be
  at or before that sample's decision.
- Its label end must be strictly before `fit_time` by default. Set
  `label_end_inclusive=True` only when the intended convention permits equality.
- Its finalized label must have been published at or before `fit_time` in either mode.
- All required clocks must be known. `None` in an end, publication or feature-availability
  column excludes the row; a missing decision is invalid.

An observed label end is not proof that the finalized label was available then. A later
batch computation time is likewise not the availability of the historical source inputs.
Fit learned preprocessing on the eligible training fold, then apply it to held-out rows;
do not substitute a scaler's fitting timestamp for source availability.

## Preserve the original row indices

This synthetic fixture uses explicit UTC Unix nanoseconds. Candidate order is deliberately
different from input order; the filter does not sort or add rows withheld by the splitter.

```python
from openquant.availability import select_available_training

base = 1_577_836_800_000_000_000  # 2020-01-01T00:00:00Z
second = 1_000_000_000
times = lambda offsets: [base + offset * second for offset in offsets]
arguments = dict(
    decision_times=times([1, 2, 3]),
    label_end_times=times([4, 5, 10]),
    label_available_times=times([4, 6, 10]),
    feature_available_times=times([1, 3, 2]),
    fit_time=base + 10 * second,
    clock="unix_ns",
    candidate_indices=[2, 0, 1],
)
strict = select_available_training(**arguments)
inclusive = select_available_training(**arguments, label_end_inclusive=True)
assert strict.eligible_indices == (0,)
assert strict.exclusions == (
    (2, ("label_end_not_before_fit",)),
    (1, ("feature_unavailable_at_decision",)),
)
assert inclusive.eligible_indices == (2, 0)
print("strict:", strict.eligible_indices, "inclusive:", inclusive.eligible_indices)
```

```text
strict: (0,) inclusive: (2, 0)
```

`AvailabilitySelection` also records `candidate_indices`, `fit_time_ns` and the selected
boundary convention. Every candidate occurs exactly once among eligible indices and
exclusions; one exclusion can list several reasons. These positions index the original
aligned arrays, not a compacted feature matrix. After dropping or reordering rows, remap
by stable IDs rather than reusing old positions against a new array.

## Clock and error contracts

`clock` is required. `unix_ns` accepts exact integer nanoseconds within signed int64 range,
excluding the minimum-value NaT sentinel; booleans and floating epoch seconds are rejected.
`utc` accepts aware builtin `datetime` objects or ISO strings with seconds, up to six
fractional digits, and `Z` or a known numeric offset. Naive datetimes, datetime subclasses,
date-only strings and unknown `-00:00` offsets are rejected. Use explicit integer
nanoseconds for finer precision. Integers alone cannot prove that the caller used UTC
nanoseconds rather than bar positions.

Length mismatches, duplicate/out-of-range candidate indices, malformed clocks and known
chronology contradictions raise `ValueError`, including on noncandidate rows. Label ends
cannot precede decisions, and known label publication cannot precede either the decision
or its known end. Inputs are not mutated.

## Limits and the next check

The function audits declared clocks, not their truth or completeness. It does not replace
purging against actual test intervals, embargo, fold-local learning or a provider audit.
Intersect it with the splitter's training indices and retain exclusion reasons alongside
the experiment. See [point-in-time evidence](/workflows/point-in-time-data/) and
[cross-validation](/modules/cross-validation/).

The helper uses the standard library internally, but normal `openquant` imports still
initialize the package's native extension and existing dependencies. The checked-in
availability examples provide explicit `--source-tree` modes; those isolated runs do not
establish that the installed package is ready.

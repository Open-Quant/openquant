---
title: Select data available at a historical decision
description: Separate observation dates, revisions and availability, then identify the exact selected values with an offline example.
status: authored
last_authored: '2026-11-07'
audience:
  - quant-dev
  - platform-engineering
sidebar:
  order: 6
---

A row dated January can contain a value first published in February and revised in March.
Using the March value in a February decision leaks information even if the rows are sorted
and the model uses chronological folds. This guide explains selection before modeling
with a standalone offline example; it does not fetch a provider's data.

For the package APIs, use [provenance](/modules/provenance/) to select declared revisions
and rejoin exact values, [feature assembly](/modules/feature-assembly/) to require complete
declared inputs, and [availability](/modules/availability/) to filter original training
indices by source and label clocks. Their consistency checks do not remove the provider
truth and incomplete-history limitations below.

[ALFRED's introduction](https://fred.stlouisfed.org/docs/api/fred/alfred.html) describes
archiving original releases and later revisions for the same economic observation. Its
example distinguishes the January 1990 unemployment observation from its February release
and a revision years later. An observation's date alone cannot identify which value a
historical decision could use.

## Keep the clocks separate

| Field or concept | Question it answers | What it does not establish |
| --- | --- | --- |
| Observation date or period | What period does the measurement describe? | When the value became known |
| Revision date or version | Which publication or replacement of that measurement is this? | The exact intraday arrival time at your system |
| Availability time | From what instant does the declared selection policy permit using this version? | That the timestamp or its source is truthful |
| Ingestion time | When did your system receive or persist the record? | When the source first published it |
| Decision cutoff | What information boundary applies to this particular decision? | That subsequent transformations or model selection respect it |

Define whether a study reconstructs **source-public information** or **what your system
actually received**. For the latter, a published value delayed in delivery is unavailable
until receipt and any required processing are complete. Retrospective downloads cannot
by themselves reconstruct a missing historical receipt log. Do not relabel today's download
time as historical source availability, or replace missing availability with observation time.

ALFRED distinguishes observation dates from vintage dates and the intervals during which
a value was the latest available revision. A vintage can be a date with no new release;
it requests the state on that date. [Download format definitions](https://alfred.stlouisfed.org/help/downloaddata)
explain these dimensions. “Latest observation” and “latest revision of each observation”
are separate choices.

## Choose as-of or latest deliberately

For a fixed historical cutoff, an **as-of selection** admits only eligible versions and
then resolves revisions for each observation. Selecting the last row in a file is not this
operation. **Latest selection** uses the newest versions in the available snapshot, which
may be appropriate for a current descriptive report but not a historical decision.

For the FRED observations endpoint, `realtime_start=D` and `realtime_end=D` request a
single historical real-time date; these bounds are inclusive and default to today when
omitted. They are separate from `observation_start` and `observation_end`. See the
[real-time period contract](https://fred.stlouisfed.org/docs/api/fred/realtime_period.html).
The [observations endpoint](https://fred.stlouisfed.org/docs/api/fred/series_observations.html)
also supports `vintage_dates` and several output modes, including all observations,
only new/revised observations, and initial releases. Preserve the query and output mode:
an initial-release dataset is not the same as the latest value known at every cutoff.
Do not treat every returned `realtime_start` as an original publication timestamp; response
intervals must be interpreted with the requested real-time window and output semantics.

### Date-only metadata is not an intraday timestamp

A `YYYY-MM-DD` vintage does not identify whether a value was available before or after a
same-day trade. This is a limitation inferred from the date-only contract, not an intraday
service guarantee. Obtain supported release-time and delivery evidence, or declare and test
a conservative lag policy appropriate to the decision clock. Converting a date to midnight
creates precision that the evidence did not provide. Specify timezone, trading calendar,
processing delay and whether equality at the cutoff is eligible.

ALFRED says additions typically occur within one business day of release, and describes
fallback sources for release dates and verification of vintage values. That is not a promise
of zero delay, complete history or receipt at your strategy's decision time.
[ALFRED's collection and verification guidance](https://alfred.stlouisfed.org/help) is the
basis for this distinction. Even faithful selection of a provider's metadata remains
conditional on the metadata's accuracy and coverage.

## Run the offline example

`examples/point_in_time_data.py` needs Python 3.11 or newer and only its standard library. Run it from the
repository root; it uses invented records, makes no network requests and writes no files.
Its `Record`, `select` and identity helpers are teaching code local to this script, not
public OpenQuant APIs or an ALFRED response parser.

```bash
python3 examples/point_in_time_data.py
```

The documentation's executable check runs that same file:

```python
from pathlib import Path
import runpy

example = Path("examples/point_in_time_data.py")
assert example.is_file(), "run from the repository root"
runpy.run_path(str(example), run_name="__main__")
```

```text
as-of value: 100.0; latest fixture value: 102.0
release boundaries, ambiguity and content identities verified
```

The January observation is first available at `2024-02-01T12:00:00Z`, with value `100.0`.
Its March revision has value `102.0`. A February 15 cutoff selects the first version and
excludes both the revision and a second observation not yet released. The timestamps are
explicit synthetic assumptions, not times inferred from real vintage dates.

The policy groups by provider, series and observation date, filters `available_at <= cutoff`,
and takes the greatest revision date among eligible records. Ties raise an error rather
than guessing an order from version names or input rows. `cutoff=None` means latest **within
this fixture**, not latest from a live service. This simple policy assumes revision dates
order versions; another source may need sequence numbers, retractions, missing-value
markers or interval validity instead. The example does not implement those cases.

Assertions check the instant before release, the inclusive release boundary, revision
replacement, conflicting versions and input-row permutation. They also check which
identities change when selected values or the archive change.

## Identify the values actually used

The script computes two SHA-256 identities from canonical, sorted JSON:

- `archive_identity` covers all fixture records, including unselected revisions. It is a
  normalized record-content identity, not a byte hash of a downloaded response.
- `selected_identity` covers the selected records' exact decimal strings, provider,
  series, observation/revision dates, availability and version, together with the mode,
  cutoff, schema and selection-policy identifier. It identifies a selection receipt,
  not just a vector of numerical values.

Adding a valid future record changes the archive identity but leaves the historical
selection identity unchanged. Changing a selected value, its version or provider changes
selection identity. Changing the cutoff also changes it, even if the selected numbers stay
the same. Exact strings are intentional: `"100.0"` and `"100.00"` are different recorded
representations here. A production normalization policy must define its own treatment.

Keep the raw response bytes and their digest separately, along with query parameters,
retrieval evidence, selected and excluded record identifiers, transformation settings and
code revision. A hash binds the recorded content; it does not certify the provider, prove
historical availability or establish that the records are complete. Hashing a record ID
without its selected value would also miss a value changed under that same ID. If values,
metadata and the reference digest are replaced together, they can remain internally
consistent while describing a false history. Detecting change requires a trusted prior
receipt or independent evidence; it still does not prove that omitted revisions never
existed. A revision-association check alone cannot establish provider truth or complete
history.

## Carry the boundary into OpenQuant workflows

Select eligible revisions upstream before [cleaning data](/modules/data/). The current
`data.clean_ohlcv` keeps canonical OHLCV columns and resolves duplicate `(symbol, ts)` keys
by input-row policy; it does not perform vintage selection and drops extra provenance
columns. Retain the selection receipt beside the cleaned data. Likewise, an
[adapter's](/modules/adapters/) `as_of` weight label records a caller-supplied date; it does
not verify when inputs were available.

For each historical decision, select the permitted snapshot, construct features using only
that snapshot and the allowed history, then fit preprocessing and models within their
training boundaries. A valid vintage join does not cure full-sample normalization, revised
universe membership or a label that was unresolved at training time. Record any separate
rules for delistings and corporate-action revisions rather than assuming this economic-data
example covers them. The [model-selection walkthrough](/workflows/model-selection-holdout/)
shows a training/holdout boundary; it does not supply a provider archive.

## Interpret the evidence before using a result

| Evidence retained | What it supports | What still needs separate validation |
| --- | --- | --- |
| This synthetic example passes | The stated selection and identity invariants hold on these fixtures | Real provider schemas, coverage, corrections, outages and timing |
| A provider vintage response and query | What that provider returned for the recorded request | Authentic historical availability and completeness for the study |
| A raw digest and selection receipt | Repeating or detecting changes in the recorded inputs and policy | Whether the underlying values and timestamps are true |
| Historical ingestion and processing logs | When the recorded system received and prepared data | Clock quality, missing logs and whether the strategy used that data |
| A deterministic backtest or notebook | Reproducibility under its inputs and assumptions | Live fills, costs, capacity, operational reliability and future performance |

Before interpreting a production study, check:

1. Is the result clearly labeled synthetic, historical provider replay or live observation?
2. Are observation period, revision ordering, availability basis, timezone and cutoff explicit?
3. Are ambiguous, missing, retracted and late records handled by a documented rule rather than silently filled?
4. Can another reviewer recover the raw snapshot, exact selected values, query, transformations and implementation?
5. Have representative provider cases and receipt delays been checked independently of these synthetic fixtures?
6. Are timing, modeling, execution and statistical claims supported by their own evidence?

Passing one row of this checklist does not establish the others. Preserve unresolved
limitations with the result rather than turning a successful offline check into a production
validation claim.

---
title: "data_processing"
description: "Rust OHLCV preparation: stable deduplication, frequency-aware gap reports, and explicit missing and off-grid bars on regular calendars."
status: authored
last_authored: '2026-10-10'
audience:
  - quant-dev
  - platform-engineering
module: "data_processing"
api_surface: "rust"
citation:
  - "OpenQuant crates/openquant/src/data_processing.rs: shared row, column and DataFrame contracts."
  - "OpenQuant crates/openquant/tests/data_processing.rs: stable deduplication, inferred spacing, weekday gaps, and alignment reports."
  - "OpenQuant crates/openquant/tests/docs_data_processing.rs: executable example for this page."
rust_api:
  - "clean_ohlcv_rows"
  - "clean_ohlcv_columns"
  - "clean_ohlcv_df"
  - "quality_report"
  - "quality_report_columns"
  - "quality_report_df"
  - "align_calendar_rows"
  - "align_calendar_columns"
  - "align_calendar_df"
  - "OhlcvRow"
  - "OhlcvColumns"
  - "AlignedOhlcvRow"
  - "AlignedOhlcvColumns"
  - "DataQualityReport"
  - "CalendarAlignmentReport"
  - "DataProcessingError"
sidebar:
  badge: Module
---

A repeated bar can overweight an observation; a missing bar can turn an apparent one-period
return into a multi-period return. `openquant::data_processing` prepares OHLCV observations
before [bars](/modules/data-structures/), [labels](/modules/labeling/) or features: sort and
deduplicate keys, inspect gaps, then optionally put each symbol onto an explicit time grid.
It is an ingestion utility rather than an AFML estimator. Its behavioral reference is the
repository's data-processing implementation and tests, not a numbered book snippet.

## Three representations, one implementation

All nine functions use the same Polars implementation. Cleaned and aligned outputs are
sorted by symbol, then timestamp. Report functions inspect a sorted copy without changing
the input.

| Operation | Rows | Columns | Polars DataFrame |
| --- | --- | --- | --- |
| Sort and deduplicate | `clean_ohlcv_rows(rows, keep_last)` | `clean_ohlcv_columns(columns, keep_last)` | `clean_ohlcv_df(df, keep_last)` |
| Inspect without deduplicating | `quality_report(rows, removed)` | `quality_report_columns(columns, removed)` | `quality_report_df(df, removed)` |
| Regular calendar grid | `align_calendar_rows(rows, seconds)` | `align_calendar_columns(columns, seconds)` | `align_calendar_df(df, seconds)` |

Cleaning returns `(cleaned_data, DataQualityReport)`. Alignment returns
`(aligned_data, CalendarAlignmentReport)`. The column and DataFrame forms return `Result`;
the row cleaner and `quality_report` return their values directly and can panic on an
underlying Polars failure. `align_calendar_rows` returns `Result` too.

| Representation | Input contract | Aligned output |
| --- | --- | --- |
| `OhlcvRow` | `timestamp: chrono::NaiveDateTime` interpreted as UTC, `symbol: String`, six `f64` fields: `open`, `high`, `low`, `close`, `volume`, `adj_close` | `AlignedOhlcvRow`: the same key, six `Option<f64>` fields, `is_missing_bar: bool` |
| `OhlcvColumns` | `timestamps_us: Vec<i64>` since the Unix epoch in UTC, `symbols: Vec<String>`, six `Vec<f64>` fields; all eight vectors must have equal length | `AlignedOhlcvColumns`: six `Vec<Option<f64>>` fields and `is_missing_bar: Vec<bool>` |
| `DataFrame` | Required columns `symbol` (String), `ts_us` (Int64 epoch microseconds), and the six OHLCV fields (Float64) | Same columns with nullable OHLCV values, plus Boolean `is_missing_bar` |

Rust expects this schema already prepared: it does not parse timestamp strings, resolve
column aliases, supply `adj_close`, or convert local wall time to UTC. Convert timezones
before constructing rows, and retain microsecond precision when moving between forms.
Required names are checked, and key types are accessed as String/Int64; this is not an
exhaustive dtype or value validator for the six numerical columns. In particular the
DataFrame cleaner/report can pass through numerical columns without casting them.

## One duplicate, one missing weekday

This deliberately tiny fixture is synthetic. January 2 has two observations for `AAA`, in
input order; January 4 has none. A duplicate means the same **symbol and timestamp**,
regardless of whether prices agree. `keep_last = true` retains 10.5; `false` retains 10.0.
Stable sorting preserves input order among equal keys, so this choice is deterministic.

```rust
use chrono::NaiveDate;
use openquant::data_processing::{
    align_calendar_rows, clean_ohlcv_rows, quality_report, DataProcessingError, OhlcvRow,
};

let bar = |day, close| OhlcvRow {
    timestamp: NaiveDate::from_ymd_opt(2024, 1, day).unwrap().and_hms_opt(0, 0, 0).unwrap(),
    symbol: "AAA".to_string(),
    open: close,
    high: close,
    low: close,
    close,
    volume: 1.0,
    adj_close: close,
};
let rows = vec![bar(2, 10.0), bar(2, 10.5), bar(3, 11.0), bar(5, 12.0)];
let raw_report = quality_report(&rows, 0);
assert_eq!(raw_report.row_count, 4);
assert_eq!(raw_report.duplicate_key_count, 1);

let (clean, report) = clean_ohlcv_rows(&rows, true);
assert_eq!(clean.iter().map(|r| r.close).collect::<Vec<_>>(), [10.5, 11.0, 12.0]);
assert_eq!(clean_ohlcv_rows(&rows, false).0[0].close, 10.0);
assert_eq!(report.duplicate_key_count, 0);
assert_eq!(report.rows_removed_by_deduplication, 1);
assert_eq!(report.inferred_interval_us, Some(86_400_000_000));
assert_eq!(report.gap_interval_count, 1);
// Report-only calls carry forward the supplied removal count.
assert_eq!(quality_report(&clean, 1), report);

let (aligned, alignment) = align_calendar_rows(&rows, 86_400)?;
let missing: Vec<bool> = aligned.iter().map(|r| r.is_missing_bar).collect();
assert_eq!(missing, [false, false, true, false]);
assert_eq!(aligned[2].timestamp, bar(4, 0.0).timestamp);
assert_eq!(aligned[2].close, None);
assert_eq!(alignment.rows_removed_by_deduplication, 1);
assert!(alignment.off_grid_bars.is_empty());

// A two-day grid starts on Jan 2: Jan 3 and Jan 5 do not land on it.
let (coarse, omitted) = align_calendar_rows(&rows, 172_800)?;
assert_eq!(coarse.len(), 2); // Jan 2, Jan 4; no rounding or aggregation
assert_eq!(coarse[0].close, Some(10.5));
assert_eq!(coarse[1].close, None);
assert_eq!(omitted.off_grid_bars, vec![
    ("AAA".to_string(), bar(3, 0.0).timestamp),
    ("AAA".to_string(), bar(5, 0.0).timestamp),
]);
assert!(matches!(align_calendar_rows(&rows, 0), Err(DataProcessingError::NonPositiveInterval)));
println!("raw={} clean={} removed={} gaps={}",
    raw_report.row_count, report.row_count, report.rows_removed_by_deduplication,
    report.gap_interval_count);
println!("daily missing={missing:?}; two-day off-grid={}", omitted.off_grid_bars.len());
```

```text
raw=4 clean=3 removed=1 gaps=1
daily missing=[false, false, true, false]; two-day off-grid=2
```

The snippet is executed in `crates/openquant/tests/docs_data_processing.rs`, in addition to
the site's Rust compilation check. Its expected values follow directly from the four
input rows and the January 2024 weekday calendar.

<figure>
<img class="dark:sl-hidden" src="/figures/data-processing-light.svg" alt="Synthetic AAA bars on January 2, 3 and 5, 2024. The raw input repeats January 2 with closes 10 and 10.5. Keeping the last duplicate leaves closes 10.5, 11 and 12. Daily alignment adds January 4 with null values and is_missing_bar true." />
<img class="light:sl-hidden" src="/figures/data-processing-dark.svg" alt="Synthetic AAA bars on January 2, 3 and 5, 2024. The raw input repeats January 2 with closes 10 and 10.5. Keeping the last duplicate leaves closes 10.5, 11 and 12. Daily alignment adds January 4 with null values and is_missing_bar true." />
<figcaption>Synthetic AAA bars: keep-last deduplication removes one repeated key; a daily grid adds January 4 with null values and an explicit missing-bar flag. No prices are interpolated.</figcaption>
</figure>

## Reading a quality report

`DataQualityReport` contains eight fields:

| Field | Meaning |
| --- | --- |
| `row_count`, `symbol_count` | Observations and distinct symbols in the inspected data |
| `duplicate_key_count` | Extra rows sharing a key; three equal keys contribute two, not one; zero after cleaning |
| `inferred_interval_us` | Most common positive within-symbol spacing, pooled over symbols; smallest spacing wins ties; `None` if there are no positive spacings |
| `gap_interval_count` | Consecutive within-symbol intervals that qualify as gaps; not the number of absent bars |
| `ts_min`, `ts_max` | Earliest and latest UTC `NaiveDateTime`, `None` for empty input |
| `rows_removed_by_deduplication` | Computed by cleaning; passed through unchanged from `removed` by report-only functions |

Zero spacings from duplicates do not participate in inference. In the example, one-day
and two-day spacings tie, so the inferred interval is one day. No symbol contributes a
spacing to another symbol, but the final inferred interval is shared across the dataset.
A mixed-frequency dataset can therefore produce misleading gaps; inspect frequencies
separately when that matters.

For an inferred interval within one hour of 24 hours (inclusive), a gap means **at least
one skipped Monday–Friday UTC date strictly between two bars**. Friday to Monday is not a
gap; Friday to Tuesday is one. Exchange holidays count as skipped weekdays because no
holiday calendar is consulted. The one-hour tolerance accommodates daily spacing shifts
around daylight saving time; it does not introduce an exchange timezone or session model.

For every other inferred frequency, a gap is simply a positive spacing longer than that
inferred interval. Intraday overnight and weekend breaks therefore count. With no inferred
interval, the count is zero. A sparse series with only two distinct timestamps can infer
its entire separation as the normal interval: the report does not know your intended
frequency and cannot certify completeness.

## Alignment is a grid, not a trading calendar

Each symbol is first cleaned with **keep last**, then independently reindexed from its own
first timestamp to its last, in positive `interval_seconds` steps. The last timestamp is
included only if it falls on that grid. Daily grids include weekends and holidays; symbols
need not share a start, end or grid phase. There is no resampling, price aggregation,
interpolation, forward fill, or common cross-symbol date range.

Missing grid points have null values for all six OHLCV fields, including volume.
`CalendarAlignmentReport.rows_removed_by_deduplication` counts input duplicate rows removed;
`off_grid_bars` lists `(String, NaiveDateTime)` keys of remaining off-grid observations,
sorted by symbol and time. Those observations are absent from the output. Always inspect
this report before treating an aligned dataset as complete, especially after changing
intervals or time-of-day conventions.

For DataFrames, `is_missing_bar` is implemented as **`open` is null**. Consequently a
pre-existing row with a null opening price also receives `true`; it is not exclusively a
join-provenance indicator. `OhlcvRow` and `OhlcvColumns` cannot contain null prices, so their
new nulls represent missing grid bars. `NaN` is a floating-point value, not a null.

## Python uses the `data` schema

There is no Python `data_processing` module. The existing [`openquant.data`](/modules/data/)
helpers expose related cleaning, reporting and alignment operations, with their own schema
normalization. They are implemented independently in Python/Polars rather than dispatching
to these Rust functions. They use `ts` as a Polars datetime, not Rust's Int64 `ts_us`, and accept
aliases and a missing `adj_close`. Null canonical rows are dropped during Python cleaning;
Rust DataFrame cleaning does not offer that normalization. This is a companion API example,
not an additional binding to the nine Rust entry points.

```python
import polars as pl
from openquant import data

prices = [10.0, 10.5, 11.0, 12.0]
raw = pl.DataFrame({
    "ts": ["2024-01-02", "2024-01-02", "2024-01-03", "2024-01-05"],
    "symbol": ["AAA"] * 4,
    "open": prices, "high": prices, "low": prices, "close": prices,
    "volume": [1.0] * 4,
})
clean, report = data.clean_ohlcv(raw, return_report=True)
aligned, alignment = data.align_calendar(raw, interval="1d", return_report=True)
assert clean["close"].to_list() == [10.5, 11.0, 12.0]
assert report["rows_removed_by_deduplication"] == 1
assert report["gap_interval_count"] == 1
assert aligned["close"].to_list() == [10.5, 11.0, None, 12.0]
assert alignment["off_grid_bar_count"] == 0
print("Python data closes:", aligned["close"].to_list())
```

```text
Python data closes: [10.5, 11.0, None, 12.0]
```

Python reports are dictionaries; their timestamp bounds are strings and their alignment
report contains a DataFrame of omitted bars plus `off_grid_bar_count`. Rust reports use
the typed structs above. See [data](/modules/data/) for loading, cache provenance, wrapper
null handling and the Python API reference.

## Errors and limits

`DataProcessingError` has four variants:

| Variant | Cause |
| --- | --- |
| `Frame { context, reason }` | Missing required column, an accessed column's wrong dtype, a failed Polars operation, or an out-of-range off-grid timestamp |
| `NullValue { column, row }` | Null `symbol` or `ts_us`; `row` is in sorted order |
| `LengthMismatch(String)` | Unequal column-vector lengths, with lengths in the message |
| `NonPositiveInterval` | Alignment interval is zero or negative |

Empty, correctly shaped inputs are supported: reports have zero counts and `None` bounds
and interval (preserving a report-only removal count); alignment adds an empty missing-bar
flag. Prices, volume, OHLC ordering, finiteness and economic plausibility are not validated.
Cleaning does not repair prices or adjust corporate actions; `adj_close` is supplied data.

Use representable UTC timestamps and practical positive intervals: positivity is checked,
but interval-to-microsecond multiplication and timestamp arithmetic are not generally
protected against overflow. Extremely small intervals over long histories can allocate
very large grids. The wrappers materialize Polars data; the row and column APIs are
convenience representations, not streaming or zero-copy interfaces.

The behavioral tests are in
[`crates/openquant/tests/data_processing.rs`](https://github.com/Open-Quant/openquant/blob/main/crates/openquant/tests/data_processing.rs),
including frequency inference, weekday and intraday gaps, stable first/last selection,
and off-grid reports. After hygiene, [data structures](/modules/data-structures/)
constructs event bars and [filters](/modules/filters/) selects events for labeling.

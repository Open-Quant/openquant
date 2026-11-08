---
title: Diagnose input errors
description: Identify validated input failures and assumptions that Python and Rust callers must check themselves.
status: authored
last_authored: '2026-11-07'
audience:
  - quant-dev
  - platform-engineering
sidebar:
  order: 3
---

Start with the function's documented input contract and the namespace you called.
A [Python wrapper and its native counterpart](/module-reference/api-task-guide/#python-wrappers-and-native-calls)
can accept different timestamp formats and return different types. Fix the input
at its source; do not catch an exception and substitute zeros for failed calculations.

## Match the failure to the input

These are current validated cases, not an exhaustive exception list. A dependency
can raise its own exception, and Python argument conversion can fail before Rust runs.

| Task and symptom | What is checked | What to change |
| --- | --- | --- |
| `data.clean_ohlcv`: `missing required OHLCV columns` | Required columns after known aliases are normalized | Supply `ts`, `symbol`, `open`, `high`, `low`, `close`, `volume`. `adj_close` is optional and defaults to `close`; it is not an adjustment calculation. |
| `data.clean_ohlcv`: Polars `InvalidOperationError` converting a numeric column | Price and volume columns use strict `Float64` casts; a string such as `"bad"` raises instead of becoming null | Inspect the offending source values and define their treatment before cleaning. Timestamp parsing is different: unparseable timestamps become null and their rows are dropped. See [cleaning contracts](/modules/data/#cleaning-and-provenance-contracts). |
| CV wrapper: `t0 contains NaT` or `t0 and t1 must both be datetimes or both be integers` | One-dimensional spans, missing datetimes, matching lengths and datetime/integer representation | Use one representation for both ends. Datetimes become nanoseconds; integer units pass through unchanged, so match them yourself. |
| CV: `samples_info_sets[i] ends before it starts` | Each span has `end >= start` | Inspect the indexed label's endpoints. Do not swap them to conceal a label-construction error. |
| CV: `n_splits must be between 2 and the number of samples` | A nonnegative fold count reaching the core must fit the sample count | Select a feasible count. A negative Python count instead fails unsigned conversion with `OverflowError`. |
| CV: `pct_embargo must be finite and in [0, 1)` | Embargo fraction, not a number of rows | Pass a fraction such as `0.01`, not `1` for one percent. Embargo size is rounded up from the total sample count. |
| `labeling.add_vertical_barrier`: `invalid datetime` | Native timestamp strings use `YYYY-MM-DD HH:MM:SS` with optional fractional seconds | Normalize timezone-aware data to the intended UTC instants before formatting. The native parser does not accept an ISO `T`/`Z` string merely because the data loader does. |
| `labeling.add_vertical_barrier`: `needs a non-zero horizon` | All four offset arguments being zero is rejected by the Python binding | Choose a bounded positive holding horizon and enough subsequent bars. Missing late barriers are omitted, not shortened. |
| Risk: `returns must not be empty` or `confidence level must be in [0, 1]` | Nonempty input and a valid quantile level | Check the series after filtering. VaR/ES use lower-tail probability, e.g. `0.05`; CDaR uses upper-tail level, e.g. `0.95`, and requires a cumulative series. |
| Risk: `prices matrix must be rectangular` or `inputs have different lengths` | Shared matrix conversion checks rectangular shape; variance also checks square shape and matching weight count | The generic `prices matrix` wording can refer to a covariance matrix here. Keep asset ordering identical across rows, columns and weights. Shape checks do not establish a valid covariance. |

See [data cleaning](/modules/data/), [labeling](/modules/labeling/),
[cross-validation](/modules/cross-validation/), and [risk metrics](/modules/risk-metrics/)
for the full conventions.

## Reproduce expected errors safely

This example exercises ordinary validation paths. It catches only the expected
exception type and checks a stable part of each message; an unexpected failure
still propagates.

```python
import polars as pl
from openquant import cross_validation as cv, data, labeling, risk


def expect_value_error(call, fragment):
    try:
        call()
    except ValueError as error:
        assert fragment in str(error), str(error)
    else:
        raise AssertionError("expected ValueError")


expect_value_error(
    lambda: data.clean_ohlcv(pl.DataFrame({"close": [1.0]})),
    "missing required OHLCV columns",
)
expect_value_error(
    lambda: cv.purged_kfold_splits([0, 2], [1, 1], 2),
    "ends before it starts",
)
expect_value_error(
    lambda: labeling.add_vertical_barrier([], []),
    "needs a non-zero horizon",
)
expect_value_error(
    lambda: risk.calculate_value_at_risk([], 0.05),
    "returns must not be empty",
)
expect_value_error(
    lambda: risk.calculate_variance([[1.0, 0.0], [1.0]], [0.5, 0.5]),
    "must be rectangular",
)
print("five expected ValueErrors")
```

```text
five expected ValueErrors
```

## A successful call can still require investigation

**Cleaning can remove rows without raising.** Invalid timestamps become null and
rows with null canonical fields are dropped. Request the report and compare input
and output row counts. NaN is distinct from null: it is not removed by that rule.
This deliberately invalid-price example demonstrates the distinction, not usable
market data.

```python
import polars as pl
from openquant import data

raw = pl.DataFrame({
    "ts": ["bad", "2026-01-02"], "symbol": ["X", "X"],
    "open": [1.0, 1.0], "high": [1.0, 1.0], "low": [1.0, 1.0],
    "close": [1.0, float("nan")], "volume": [1.0, 1.0],
})
cleaned, report = data.clean_ohlcv(raw, return_report=True)
assert cleaned.height == 1
assert report["null_counts"]["ts"] == 1
assert cleaned["close"].is_nan().to_list() == [True]
```

**Risk output is not universally finite or economically valid.** The empirical ES
implementation returns NaN when no observation is strictly below its VaR cutoff.
Variance checks dimensions but does not validate positive semidefiniteness. An
invalid covariance can therefore produce a negative number without an exception:

```python
import math
from openquant import risk

assert math.isnan(risk.calculate_expected_shortfall([0.1, 0.1], 0.05))
invalid_covariance = [[1.0, -2.0], [-2.0, 1.0]]
assert risk.calculate_variance(invalid_covariance, [0.5, 0.5]) == -0.5
```

The second result is evidence of unsuitable input, not a valid negative portfolio
risk. Check finite observations and an appropriate covariance estimate before the
call; do not turn NaN into a reported zero-risk result.

**Ordering and alignment are not universal validation checks.** CV accepts span
order as supplied; folds and embargo use row positions. Jointly sort labels,
features, predictions and weights by label start before splitting. Labeling joins
events, targets and sides by exact timestamp and can omit unmatched events. Derive
later arrays from retained labels rather than assuming one output row per request.

## An event disappears after timestamp conversion

The loader and adapters normalize datetimes to timezone-naive UTC **microseconds**.
The CV wrapper converts datetime arrays to **nanoseconds**, while the native labeling
wire format can preserve nanoseconds but does not parse timezone offsets. These are
compatible only after you choose a common time representation. Precision lost earlier
cannot be restored by converting back to nanoseconds.

Here the first bar's original instant is 123 nanoseconds past the second. Cleaning
normalizes its UTC offset and loses that sub-microsecond remainder. Supplying the
original high-precision event to the cleaned bars therefore yields no event row:

```python
import polars as pl
from openquant import data, labeling

raw = pl.DataFrame({
    "ts": ["2026-01-02T09:30:00.000000123-05:00", "2026-01-02T09:30:01-05:00"],
    "symbol": ["X", "X"], "open": [100.0, 102.0], "high": [100.0, 102.0],
    "low": [100.0, 102.0], "close": [100.0, 102.0], "volume": [1.0, 1.0],
})
cleaned = data.clean_ohlcv(raw)
bar_times = [str(t) for t in cleaned["ts"]]
prices = cleaned["close"].to_list()
assert bar_times == ["2026-01-02 14:30:00", "2026-01-02 14:30:01"]
original_event = "2026-01-02 14:30:00.000000123"  # already normalized to UTC
assert labeling.triple_barrier_events(
    bar_times, prices, [original_event], [original_event], [0.01]
) == []

# For this example, generate the event and its target on the cleaned bar clock.
event = bar_times[0]
retained = labeling.triple_barrier_events(bar_times, prices, [event], [event], [0.01])
assert len(retained) == 1 and retained[0][0] == event
```

Do not round externally generated events merely to make a join succeed. Decide whether
the data's time resolution is adequate, then generate or explicitly align events, targets
and sides on that clock. Two distinct raw bars of the same symbol can become one
`(symbol, ts)` key after conversion and then be subject to the cleaner's duplicate policy.

Precision can also change purging. In this deliberately tiny diagnostic, two distinct
zero-duration label spans do not overlap at nanosecond resolution. Converting them to
microseconds collapses their starts and ends; closed-interval purging then removes the
other row. This illustrates time representation, not a useful two-row research split.

```python
import numpy as np
from openquant import cross_validation as cv

starts = np.array([
    "2026-01-02T14:30:00.000000123", "2026-01-02T14:30:00.000000124",
], dtype="datetime64[ns]")
precise = cv.purged_kfold_splits(starts, starts, 2)
assert precise[0][0].tolist() == [1]
coarse = starts.astype("datetime64[us]")
collapsed = cv.purged_kfold_splits(coarse, coarse, 2)
assert collapsed[0][0].tolist() == []
```

Keep nanosecond timestamps in a representation that preserves them if that precision
matters. Converting a frame column to ordinary Python `datetime` objects also limits
precision to microseconds. The [data](/modules/data/), [adapters](/modules/adapters/)
and [labeling](/modules/labeling/) guides describe each API's local contract; use the
same chosen clock across their boundaries.

## Rust errors and unchecked boundaries

Fallible core APIs such as `PurgedKFold::new` and the `RiskMetrics` methods return
typed `Result` errors; their Python bindings map those errors to `ValueError`.
This does not mean every Rust function or invalid input uses that path.

In particular, the core `labeling::add_vertical_barrier` returns a vector directly.
It constructs Chrono durations and adds them to event timestamps without checked
arithmetic. Extreme offsets or dates can panic instead of returning a typed error.
The Python binding rejects an all-zero horizon, but does not validate the full
arithmetic range or require the combined offset to be positive. Use bounded,
positive horizons whose resulting dates are representable; do not probe extreme
values in a long-running research process or treat catching `ValueError` as a
complete guard. This limitation is identified from source; no panic-inducing
example is included here.

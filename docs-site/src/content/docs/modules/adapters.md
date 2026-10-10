---
title: "adapters"
description: "Polars DataFrame adapters for signals, events, weights, backtest curves, and streaming buffers."
status: authored
last_authored: '2026-10-10'
audience:
  - quant-dev
  - platform-engineering
module: "adapters"
api_surface: "python-only"
python_api:
  - "adapters.to_polars_signal_frame"
  - "adapters.to_polars_event_frame"
  - "adapters.to_polars_backtest_frame"
  - "adapters.to_polars_weights_frame"
  - "adapters.to_polars_indicator_matrix"
  - "adapters.to_polars_frontier_frame"
  - "adapters.SignalStreamBuffer"
  - "adapters.to_pandas"
citation:
  - "OpenQuant implementation: python/openquant/adapters.py."
  - "OpenQuant regression tests: python/tests/test_adapters_viz.py."
sidebar:
  badge: Module
---

The adapters turn parallel Python sequences into Polars DataFrames. They preserve
input order, check corresponding lengths and name columns consistently for notebooks.
They do not align rows by timestamp, sort data, normalize weights or validate a
trading strategy. The [implementation](https://github.com/Open-Quant/openquant/blob/main/python/openquant/adapters.py)
and [adapter tests](https://github.com/Open-Quant/openquant/blob/main/python/tests/test_adapters_viz.py)
are the contract for these Python-only helpers.

## Choose the frame shape

| Function | Required input sequences | Output columns |
| --- | --- | --- |
| `to_polars_signal_frame` | `timestamps`, `signal` | `ts`, `signal`; optional `side`, repeated `symbol` |
| `to_polars_event_frame` | `starts`, `ends`, `probs` | `start_ts`, `end_ts`, `prob`; optional `side`, `label` |
| `to_polars_backtest_frame` | `timestamps`, `equity_curve` | `ts`, `equity`; optional `returns`, `position` |
| `to_polars_weights_frame` | `asset_names`, `weights` | `asset`, `weight`; optional repeated `as_of` |
| `to_polars_frontier_frame` | `volatility`, `returns` | `volatility`, `return`, `point_id`; optional `sharpe` |
| `to_polars_indicator_matrix` | rectangular `ind_mat` | `bar_index`, one integer column per label |

Frontier point IDs default to `p0`, `p1`, …; indicator row indices default to
0, 1, … and labels to `label_0`, `label_1`, …. The matrix is row-major (bars ×
labels), and entries are cast to integers. An empty matrix returns only an empty
`bar_index` column. Optional sequences must also match their corresponding rows.

## Convert and buffer signals

```python
from openquant.adapters import SignalStreamBuffer, to_polars_signal_frame

signals = to_polars_signal_frame(
    timestamps=["2024-01-02 09:30:00", "2024-01-02 09:31:00"],
    signal=[0.5, -0.3], side=[1.0, -1.0], symbol="CL",
)
print("columns:", signals.columns)
print("timestamp type:", signals.schema["ts"])
print("signals:", signals["signal"].to_list())

buffer = SignalStreamBuffer()
buffer.append(["2024-01-02 09:32:00"], [0.1], symbol="CL")
buffer.append(["2024-01-02 09:33:00"], [-0.2], symbol="CL")
print("buffer rows:", buffer.frame().height)
print("symbols:", buffer.frame()["symbol"].to_list())
buffer.clear()
print("after clear:", buffer.frame().height)
```

```text
columns: ['ts', 'signal', 'side', 'symbol']
timestamp type: Datetime(time_unit='us', time_zone=None)
signals: [0.5, -0.3]
buffer rows: 2
symbols: ['CL', 'CL']
after clear: 0
```

`SignalStreamBuffer.append` constructs each batch immediately, so a length mismatch
fails at append time. `frame()` vertically concatenates every stored batch in append
order: use consistent optional columns and compatible dtypes across batches. It
has no time window, eviction, deduplication or persistence; call `clear()` to release
stored batches. An empty buffer returns an empty `ts`, `signal` frame.

## Parsing and error handling

Timestamp columns use the [shared parser](https://github.com/Open-Quant/openquant/blob/main/python/openquant/data.py).
The parser attempts datetime parsing with `strict=False`; unparseable entries in
a recognizable date series become null. If no format can be inferred (for example,
a series containing only invalid strings), Polars can raise a parsing error. A frame with matching lengths can therefore still contain unusable timestamps.
Inspect nulls before joining or plotting; no adapter checks chronological order or
whether an event's end follows its start.

```python
from openquant.adapters import to_polars_signal_frame, to_polars_weights_frame

frame = to_polars_signal_frame(
    ["2024-01-02 09:30:00", "bad timestamp"], [0.1, 0.2]
)
print("null timestamps:", frame["ts"].null_count())
try:
    to_polars_weights_frame(["CL", "GC"], [1.0])
except ValueError as error:
    print(type(error).__name__)
```

```text
null timestamps: 1
ValueError
```

`to_pandas(frame)` delegates to Polars' pandas conversion and raises a helpful
`RuntimeError` when pandas is absent. Install pandas before using the bridge; PyArrow is already a package dependency. Conversion
can require a copy; it is not a streaming interface.

[pipeline](/modules/pipeline/)'s `run_mid_frequency_pipeline_frames` already calls
these adapters. Use them directly for custom stage outputs, and
[viz](/modules/viz/) to build plotting payloads from sequence columns.

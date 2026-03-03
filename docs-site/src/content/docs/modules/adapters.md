---
title: "adapters"
description: "Polars DataFrame adapters for signals, events, weights, backtest curves, and streaming buffers."
status: validated
last_validated: '2026-03-02'
audience:
  - quant-dev
  - platform-engineering
module: "adapters"
api_surface: "python-only"
risk_notes:
  - "All adapter functions validate input length alignment before constructing frames."
  - "SignalStreamBuffer supports incremental append for streaming research notebooks."
  - "to_pandas() is available for downstream tools that require pandas; requires pandas to be installed."
rust_api:
  - "to_polars_signal_frame"
  - "to_polars_event_frame"
  - "to_polars_backtest_frame"
  - "to_polars_weights_frame"
  - "SignalStreamBuffer"
sidebar:
  badge: Module
---

## Concept Overview

The Rust core returns results as plain dicts and lists. The adapters module converts these into typed Polars DataFrames with proper datetime parsing, column naming, and validation. This is the standard way to move data between the Rust computation engine and Python analysis/visualization code.

`SignalStreamBuffer` provides an incremental append interface for streaming workflows where signals arrive in chunks — common in live research notebooks or paper-trading loops.

## When to Use

Use adapters whenever you receive output from the Rust core or pipeline module and need DataFrames for analysis, visualization, or further processing. The pipeline module's `_frames` variant calls these adapters internally.

**Alternatives**: Manual Polars DataFrame construction from dicts, but you lose validation and timestamp parsing.

## Usage Examples

### Python

#### Convert pipeline outputs to typed DataFrames

```python
from openquant.adapters import (
    to_polars_signal_frame,
    to_polars_weights_frame,
    SignalStreamBuffer,
)

# Signal frame from raw timestamps + values
signals = to_polars_signal_frame(
    timestamps=["2024-01-02T09:30:00", "2024-01-02T09:31:00"],
    signal=[0.5, -0.3],
    side=[1.0, -1.0],
    symbol="CL",
)

# Streaming buffer for incremental signal updates
buf = SignalStreamBuffer()
buf.append(timestamps=["2024-01-02T09:32:00"], signal=[0.1])
buf.append(timestamps=["2024-01-02T09:33:00"], signal=[-0.2])
all_signals = buf.frame()  # concat into single DataFrame
```

## API Reference

### Python API

- `adapters.to_polars_signal_frame`
- `adapters.to_polars_event_frame`
- `adapters.to_polars_backtest_frame`
- `adapters.to_polars_weights_frame`
- `adapters.to_polars_indicator_matrix`
- `adapters.to_polars_frontier_frame`
- `adapters.SignalStreamBuffer`
- `adapters.to_pandas`

### Key Functions

- `to_polars_signal_frame`
- `to_polars_event_frame`
- `to_polars_backtest_frame`
- `to_polars_weights_frame`
- `SignalStreamBuffer`

## Implementation Notes

- All adapter functions validate input length alignment before constructing frames.
- SignalStreamBuffer supports incremental append for streaming research notebooks.
- to_pandas() is available for downstream tools that require pandas; requires pandas to be installed.

## Related Modules

- [`pipeline`](/modules/pipeline/)
- [`data`](/modules/data/)

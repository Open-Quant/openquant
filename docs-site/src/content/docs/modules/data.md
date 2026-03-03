---
title: "data"
description: "OHLCV loading, cleaning, calendar alignment, and data quality reporting."
status: validated
last_validated: '2026-03-02'
audience:
  - quant-dev
  - platform-engineering
module: "data"
api_surface: "python-only"
risk_notes:
  - "Column aliases are resolved automatically (e.g., 'timestamp' → 'ts', 'ticker' → 'symbol')."
  - "clean_ohlcv deduplicates by (symbol, ts) and sorts chronologically."
  - "align_calendar marks missing bars with is_missing_bar=True for downstream imputation logic."
rust_api:
  - "load_ohlcv"
  - "clean_ohlcv"
  - "align_calendar"
  - "data_quality_report"
sidebar:
  badge: Module
---

## Concept Overview

Before any AFML workflow begins, raw market data must be loaded into a consistent schema, cleaned of duplicates and formatting issues, and aligned to a regular time grid. This module handles that ingestion layer.

It accepts CSV or Parquet files with flexible column naming (e.g., "timestamp", "datetime", "date" all map to "ts"; "ticker" or "asset" map to "symbol") and produces a standardized Polars DataFrame with canonical OHLCV columns. Deduplication handles duplicate (symbol, timestamp) keys, and calendar alignment generates a regular grid with explicit gap markers.

The data quality report provides diagnostics — row counts, symbol counts, duplicate counts, gap intervals, and null counts — that should be inspected before feeding data into bars, labeling, or any downstream module.

## When to Use

Use this module as the first step when working with pre-aggregated OHLCV data (daily bars, minute bars from a vendor). If you have raw tick/trade data instead, use the `data_structures` module to construct bars first.

**Prerequisites**: A CSV or Parquet file, or an existing Polars DataFrame with OHLCV-like columns.

**Alternatives**: Direct Polars/pandas loading if you handle column normalization and cleaning yourself.

## Key Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `path` | `str | Path` | File path to CSV or Parquet OHLCV data | — |
| `symbol` | `str | None` | Symbol name if not present as a column in the data | None |
| `interval` | `str` | Calendar alignment interval (e.g., '1d', '1h', '5m') | '1d' |
| `dedupe_keep` | `str` | Which duplicate to keep: 'first' or 'last' | 'last' |

## Usage Examples

### Python

#### Load, clean, and inspect OHLCV data

```python
from openquant.data import load_ohlcv, data_quality_report, align_calendar

# Load from CSV/Parquet with auto column normalization
df, report = load_ohlcv("prices.csv", symbol="AAPL", return_report=True)
print(report)
# {'row_count': 5040, 'symbol_count': 1, 'duplicate_key_count': 0, ...}

# Align to regular calendar (fills gaps with nulls + is_missing_bar flag)
aligned = align_calendar(df, interval="1d")

# Quality report on any DataFrame
quality = data_quality_report(df)
```

## Common Pitfalls

- Forgetting to check the quality report for gaps — missing bars silently create NaN features downstream.
- Using align_calendar with an interval shorter than the data's actual frequency — this creates many synthetic missing-bar rows.

## API Reference

### Python API

- `data.load_ohlcv`
- `data.clean_ohlcv`
- `data.align_calendar`
- `data.data_quality_report`

### Key Functions

- `load_ohlcv`
- `clean_ohlcv`
- `align_calendar`
- `data_quality_report`

## Implementation Notes

- Column aliases are resolved automatically (e.g., 'timestamp' → 'ts', 'ticker' → 'symbol').
- clean_ohlcv deduplicates by (symbol, ts) and sorts chronologically.
- align_calendar marks missing bars with is_missing_bar=True for downstream imputation logic.

## Related Modules

- [`data-structures`](/modules/data-structures/)

---
title: "data"
description: "Fetching daily OHLCV through a cache, content hashes for run manifests, and OHLCV loading, cleaning, calendar alignment and quality reporting."
status: generated
generated_from: src/data/moduleDocs.ts
last_generated: '2026-09-25'
audience:
  - quant-dev
  - platform-engineering
module: "data"
api_surface: "both"
rust_api:
  - "load_ohlcv"
  - "clean_ohlcv"
  - "align_calendar"
  - "data_quality_report"
sidebar:
  badge: Module
---

## Concept Overview

Before any AFML workflow begins, raw market data must be fetched, loaded into a consistent schema, cleaned of duplicates and formatting issues, and aligned to a regular time grid. This module handles that ingestion layer.

`fetch(symbols, start, end, source=..., cache_dir=...)` gets daily OHLCV bars from a pluggable source. The source is any object with a `name` and a `fetch_symbol(symbol, start, end)` method (the `DataSource` protocol). Three sources ship with the module. `LocalSampleSource` reads the bundled synthetic sample and is the default. `LocalFileSource` reads your own CSV or Parquet file. `CallableSource` wraps your own function, which calls your vendor with your key. Each (source, symbol, date range) request is cached as a Parquet file and a JSON sidecar under `<cache_dir>/<source>[@<version>]/<symbol>/<start>_<end>.parquet`. A repeated request is served from disk without calling the source, so it works offline, and `offline=True` makes that a guarantee. The returned frame is the canonical `clean_ohlcv` frame, and it has passed `data_quality_report`.

`dataset_hash(df)` is a deterministic SHA-256 of a table's contents. `record_dataset_hash(manifest, df, **meta)` writes that hash, with the source, terms and request, into a run manifest, so every result can name the exact data behind it. `experiments/run_pipeline.py` records it in every `run_manifest.json`.

The loaders accept CSV or Parquet files with flexible column naming (e.g., "timestamp", "datetime", "date" all map to "ts"; "ticker" or "asset" map to "symbol") and produce a standardized Polars DataFrame with canonical OHLCV columns. Deduplication handles duplicate (symbol, timestamp) keys, and calendar alignment generates a regular grid with explicit gap markers.

The data quality report provides diagnostics — row counts, symbol counts, duplicate counts, gap intervals, and null counts — that should be inspected before feeding data into bars, labeling, or any downstream module.

## When to Use

Use `fetch` when a study needs daily bars for a list of symbols and must be reproducible. Record the dataset hash in the run manifest. Use the loaders directly when you already hold pre-aggregated OHLCV data (daily bars, minute bars from a vendor). If you have raw tick/trade data instead, use the `data_structures` module to construct bars first.

**Prerequisites**: For `fetch`, a source: the bundled synthetic sample, your own file, or your own vendor function and key. For the loaders, a CSV or Parquet file, or an existing Polars DataFrame with OHLCV-like columns.

**Alternatives**: Direct Polars/pandas loading if you handle column normalization and cleaning yourself.

## Key Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `symbols` | `str | Iterable[str]` | Symbols to fetch; duplicates are dropped | — |
| `start, end` | `date | datetime | str` | Inclusive date range (ISO strings accepted) | — |
| `source` | `DataSource | None` | Where bars come from; None uses the synthetic LocalSampleSource | None |
| `cache_dir` | `str | Path | None` | Cache root; None uses $OPENQUANT_DATA_CACHE, $XDG_CACHE_HOME/openquant/data or ~/.cache/openquant/data | None |
| `refresh / offline` | `bool` | Refetch even if cached / never call the source (raise CacheMissError) | False |
| `return_meta` | `bool` | Also return provenance (source, terms, cache status, dataset_hash) for the run manifest | False |
| `path` | `str | Path` | File path to CSV or Parquet OHLCV data (load_ohlcv) | — |
| `symbol` | `str | None` | Symbol name if not present as a column in the data | None |
| `interval` | `str` | Calendar alignment interval (e.g., '1d', '1h', '5m') | '1d' |
| `dedupe_keep` | `str` | Which duplicate to keep: 'first' or 'last' | 'last' |

## Usage Examples

### Python

#### Fetch through the cache and record the dataset hash

```python
import tempfile

from openquant.data import data_quality_report, fetch, quality_failures, record_dataset_hash

cache = tempfile.mkdtemp()  # omit cache_dir to use ~/.cache/openquant/data

# The default source is the bundled SYNTHETIC sample (SYN_A ... SYN_E), so this runs offline.
df, meta = fetch(["SYN_A", "SYN_B"], "2023-01-01", "2023-03-31", cache_dir=cache, return_meta=True)
print(df.columns)  # ['ts', 'symbol', 'open', 'high', 'low', 'close', 'volume', 'adj_close']
print(df.height, meta["cache"])  # 130 {'SYN_A': 'miss', 'SYN_B': 'miss'}
assert quality_failures(data_quality_report(df)) == []

# The same request again is read from disk; offline=True guarantees the source is not called.
again, meta2 = fetch(["SYN_A", "SYN_B"], "2023-01-01", "2023-03-31", cache_dir=cache, return_meta=True, offline=True)
print(meta2["cache"])  # {'SYN_A': 'hit', 'SYN_B': 'hit'}
assert again.equals(df) and meta2["dataset_hash"] == meta["dataset_hash"]

# Put the content hash, source, terms and request into the run manifest.
manifest = record_dataset_hash({"run_name": "demo"}, df, **meta)
print(manifest["dataset_hash"].startswith("sha256:"))  # True
```

#### Plug in your own data vendor (fetch-only, your own key)

```python doc-check=skip
import os

import polars as pl
from openquant.data import CallableSource, fetch


def my_vendor(symbol, start, end):
    key = os.environ["MY_VENDOR_API_KEY"]  # your own key, from the environment; never commit it
    rows = my_vendor_client.daily_bars(symbol, start, end, api_key=key)  # your vendor's client
    return pl.DataFrame(rows)  # columns such as date/open/high/low/close/volume[/adj_close]


source = CallableSource(my_vendor, name="my-vendor", terms="https://my-vendor.example/terms")
df, meta = fetch(["SPY", "TLT"], "2020-01-01", "2024-12-31", source=source, return_meta=True)
```

#### Load, clean, and inspect OHLCV data

```python doc-check=skip
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

- Treating the bundled sample as market data. It is synthetic, so any result computed on it says nothing about real markets.
- Committing a cache directory or fetched data to a repository. Almost every vendor's terms forbid that; see DATA_SOURCES.md.
- Expecting a cached range to serve a sub-range. The cache key is the exact (source, symbol, start, end), so a different range is a new fetch.
- Forgetting to check the quality report for gaps — missing bars silently create NaN features downstream.
- Using align_calendar with an interval shorter than the data's actual frequency — this creates many synthetic missing-bar rows.
- Using align_calendar with bars that are not on the grid. Each symbol's grid starts at its first bar and steps by `interval`, so a bar stamped off that grid (a daily bar at a different time of day, an irregular intraday print) is not in the output. `align_calendar(df, interval=..., return_report=True)` returns `(aligned, report)` with those bars in `report["off_grid_bars"]` and their number in `report["off_grid_bar_count"]`; without `return_report` a `UserWarning` says how many were dropped. The `_core.data.align_calendar` and `align_calendar_df` bindings take the same `return_report` flag.

## API Reference

### Python API

- `data.fetch`
- `data.dataset_hash`
- `data.record_dataset_hash`
- `data.quality_failures`
- `data.default_cache_dir`
- `data.DataSource`
- `data.LocalSampleSource`
- `data.LocalFileSource`
- `data.CallableSource`
- `data.CacheMissError`
- `data.load_ohlcv`
- `data.clean_ohlcv`
- `data.align_calendar`
- `data.data_quality_report`
- `data.clean_ohlcv_df`
- `data.quality_report_df`
- `data.align_calendar_df`

### Rust API

- `load_ohlcv`
- `clean_ohlcv`
- `align_calendar`
- `data_quality_report`

## Risk Notes and Caveats

- The bundled sample is SYNTHETIC, not market data. Its symbols (SYN_A ... SYN_E) are not real tickers. DATA_SOURCES.md at the repository root records the terms of every source and why no real sample is committed yet.
- Your vendor's terms govern data you fetch with your own adapter. Several vendors forbid redistribution, and some forbid persistent storage on free plans. The cache is for you only: never commit it.
- dataset_hash (oq-dataset-sha256-v1) ignores row and column order and Parquet layout. It changes with any value, column name, column type or row, so equal hashes mean the same data.
- Column aliases are resolved automatically (e.g., 'timestamp' → 'ts', 'ticker' → 'symbol').
- clean_ohlcv deduplicates by (symbol, ts) and sorts chronologically.
- align_calendar marks missing bars with is_missing_bar=True for downstream imputation logic.
- `gap_interval_count` in the quality report depends on the bar spacing, which the report infers and returns as `inferred_interval_us`: the most common spacing between consecutive bars of one symbol, pooled over symbols (the smallest on a tie). For daily data (a spacing within an hour of one day) a gap is a skipped weekday, so weekends are not gaps; exchange holidays are, since no holiday calendar is applied. For any other spacing a gap is a spacing longer than the inferred one, so intraday data counts every overnight and weekend break as a gap. Before #168 any spacing over one day was a gap whatever the frequency: weekends counted on daily data, and nothing counted on intraday data.

## Related Modules

- [`data-structures`](/modules/data-structures/)
- [`research`](/modules/research/)

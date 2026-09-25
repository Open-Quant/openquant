## ADDED Requirements

### Requirement: Fetch daily OHLCV through a pluggable source
The system SHALL provide `openquant.data.fetch(symbols, start, end, *, source=None, cache_dir=None,
refresh=False, offline=False, return_meta=False)`. It SHALL return daily bars for every requested symbol with
dates in `[start, end]` inclusive. The frame SHALL have the canonical `clean_ohlcv` columns (`ts, symbol, open,
high, low, close, volume, adj_close`) and be sorted by symbol, then ts. `source` SHALL be any object with a
`name` string and a `fetch_symbol(symbol, start, end)` method (the `DataSource` protocol). When `source` is
omitted, the SYNTHETIC `LocalSampleSource` SHALL be used.

#### Scenario: Default source returns canonical bars
- **WHEN** `fetch(["SYN_A", "SYN_B"], "2022-01-01", "2022-06-30", cache_dir=d)` is called
- **THEN** the result has exactly the canonical OHLCV columns
- **AND** it contains only `SYN_A` and `SYN_B` rows dated within the range

#### Scenario: A user-supplied adapter
- **WHEN** `source` is a `CallableSource(fn, name="vendor-x")` or any object with `name` and `fetch_symbol`
- **THEN** `fetch` calls it once per uncached symbol with `datetime.date` bounds
- **AND** it accepts any frame or mapping whose columns `clean_ohlcv` recognizes

#### Scenario: Invalid source or request
- **WHEN** `source` lacks `name` or `fetch_symbol`, `start` is after `end`, or `symbols` is empty
- **THEN** `fetch` raises `TypeError` or `ValueError` and writes nothing to the cache

### Requirement: Fetched data passes the quality report
Before returning, `fetch` SHALL run the source's output through `clean_ohlcv` and clip it to the requested
range. It SHALL reject output that has rows for another symbol or no rows in range. The returned frame SHALL
pass `data_quality_report`: at least one row, no duplicate `(symbol, ts)` keys and no nulls.

#### Scenario: Messy source output is cleaned
- **WHEN** a source returns duplicate dates and dates outside the range
- **THEN** the result has no duplicate keys and only in-range dates
- **AND** `quality_failures(data_quality_report(result))` is empty

#### Scenario: Source returns nothing usable
- **WHEN** a source returns no rows in the range, or rows for a different symbol
- **THEN** `fetch` raises `ValueError` and caches nothing

### Requirement: On-disk cache serves repeated requests offline
`fetch` SHALL cache each (source, symbol, start, end) request as one Parquet file and one JSON sidecar at
`<cache_dir>/<quoted source name>[@<quoted source version>]/<quoted symbol>/<start>_<end>.parquet` (and
`.json`). The sidecar format is `openquant-ohlcv-cache-v1` and records the source, version, terms, symbol,
range, row count, `dataset_hash` and fetch time. A cached entry whose Parquet content still matches the
recorded hash SHALL be returned without calling the source. `cache_dir` SHALL default to
`$OPENQUANT_DATA_CACHE`, else `$XDG_CACHE_HOME/openquant/data`, else `~/.cache/openquant/data`.

#### Scenario: Second call is served from cache
- **WHEN** the same request is made a second time and the source would now raise (offline)
- **THEN** `fetch` returns a frame equal to the first result without calling the source
- **AND** `return_meta=True` reports each symbol's cache status as `hit`

#### Scenario: Offline mode never fetches
- **WHEN** `offline=True` and a requested symbol/range is not cached
- **THEN** `fetch` raises `CacheMissError` (a `LookupError`) without calling the source

#### Scenario: Stale or tampered cache entries
- **WHEN** `refresh=True`, the source's `version` changes, or a cached Parquet file no longer matches its recorded hash
- **THEN** `fetch` calls the source again and rewrites the entry (with `offline=True`, a mismatched entry raises `CacheMissError`)

### Requirement: Deterministic dataset content hash
The system SHALL provide `dataset_hash(frame)`, which returns `"sha256:<64 hex>"` computed by the versioned
definition `oq-dataset-sha256-v1`. The input is: columns in sorted name order; each column typed as int,
float, str, bool, date or datetime[us,<tz>]; rows sorted by all columns with nulls last; fields encoded as
text, with floats as their exact round-trip `repr`. The hash SHALL NOT depend on row order, column order or
Parquet round-trips. It SHALL change when any value, column name, column type or the set of rows changes.
Unsupported column types SHALL raise `TypeError`.

#### Scenario: Hash is stable
- **WHEN** a frame is reversed, shuffled, has its columns reordered or is written to and read from Parquet
- **THEN** `dataset_hash` returns the same value

#### Scenario: Changing the data changes the hash
- **WHEN** one value changes by 1e-9, a row is removed or duplicated, a column is renamed, retyped or dropped, or the timestamps shift
- **THEN** `dataset_hash` returns a different value

### Requirement: Run manifests record the dataset hash
The system SHALL provide `record_dataset_hash(manifest, frame=None, *, digest=None, **provenance)`. It SHALL set
`manifest["dataset_hash"]` and merge the hash, the hash version, the row count and the provenance into
`manifest["dataset"]`. Every `run_manifest.json` written by `experiments/run_pipeline.py` (single runs, grid
runs and each grid sub-run) SHALL contain the `dataset_hash` of the data that run used.

#### Scenario: Manifest carries the hash
- **WHEN** `experiments/run_pipeline.py` runs a config or a grid
- **THEN** every `run_manifest.json` has a `dataset_hash` equal to `dataset_hash` of that run's dataset

### Requirement: Only redistributable data is committed
The repository SHALL commit only data whose terms permit redistribution. Each source and candidate source
SHALL have its terms recorded in `DATA_SOURCES.md`. Data fetched through user adapters SHALL be cached outside
the repository and never committed. Until the owner chooses a real redistributable sample, CI and the docs
SHALL use the SYNTHETIC sample. That sample is produced by `scripts/data/make_synthetic_sample.py` from a fixed
seed, and its symbols (`SYN_A` to `SYN_E`) are not real tickers.

#### Scenario: Sample is reproducible and labelled
- **WHEN** the generator script is run
- **THEN** its output is byte-identical to the committed sample
- **AND** `LocalSampleSource().terms` states that the data is SYNTHETIC

## 1. Implementation

- [x] 1.1 `DataSource` protocol, `LocalFileSource`, `LocalSampleSource`, `CallableSource`
- [x] 1.2 `fetch` with the on-disk Parquet cache, `refresh`, `offline` and `return_meta`
- [x] 1.3 `dataset_hash` and `record_dataset_hash`; record the hash in `experiments/run_pipeline.py` manifests
- [x] 1.4 Seeded generator script and the committed SYNTHETIC sample
- [x] 1.5 `DATA_SOURCES.md` with the terms of each source and candidate
- [x] 1.6 Tests in `python/tests/test_data_fetch.py`
- [x] 1.7 Data module docs page, with a runnable `fetch` example

## 2. Owner decisions

- [x] 2.1 Choose a real redistributable sample from the candidates in `DATA_SOURCES.md`, or keep the synthetic one
  Decided when #43 closed (2026-09-26): keep the SYNTHETIC sample in CI and the docs. Quandl WIKI and IEX HIST
  stay documented as candidates in `DATA_SOURCES.md` if this is revisited.
- [x] 2.2 Route the runbooks through `fetch` once 2.1 is decided (they already read the SYNTHETIC sample through `fetch`; only the source changes, via `OPENQUANT_RUNBOOK_SOURCE` or `SOURCE`). Notebook 06, which this task first named, was removed in #47.
  Decided with 2.1: no source change. The runbooks keep reading the SYNTHETIC sample through `fetch`, and users
  run them on their own data via `fetch()` (`LocalFileSource`/`CallableSource`, or `OPENQUANT_RUNBOOK_SOURCE`).

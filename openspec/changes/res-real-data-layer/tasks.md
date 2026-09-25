## 1. Implementation

- [x] 1.1 `DataSource` protocol, `LocalFileSource`, `LocalSampleSource`, `CallableSource`
- [x] 1.2 `fetch` with the on-disk Parquet cache, `refresh`, `offline` and `return_meta`
- [x] 1.3 `dataset_hash` and `record_dataset_hash`; record the hash in `experiments/run_pipeline.py` manifests
- [x] 1.4 Seeded generator script and the committed SYNTHETIC sample
- [x] 1.5 `DATA_SOURCES.md` with the terms of each source and candidate
- [x] 1.6 Tests in `python/tests/test_data_fetch.py`
- [x] 1.7 Data module docs page, with a runnable `fetch` example

## 2. Owner decisions (open)

- [ ] 2.1 Choose a real redistributable sample from the candidates in `DATA_SOURCES.md`, or keep the synthetic one
- [ ] 2.2 Route notebook 06 through `fetch` once 2.1 is decided (it still scrapes Stooq)

# Filters fixtures

The input bars are `tests/fixtures/shared/dollar_bar_sample.csv`, copied from mlfinlab v0.8.0
`tests/test_data/dollar_bar_sample.csv` (BSD-3-Clause; see `tests/FIXTURES.md`) and shared
with the other fixtures that read it.

- `events.json`: CUSUM and z-score filter events on `shared/dollar_bar_sample.csv`, written by
  `generate.py` (AFML snippet 2.4 and a rolling z-score rule, in pandas; imports neither
  openquant nor mlfinlab):

  ```bash
  uv run --with pandas python tests/fixtures/filters/generate.py
  ```

Used by `crates/openquant/tests/filters.rs`:
- CUSUM: event sequences for thresholds 0.005-0.04 and the dynamic threshold `close * 1e-5`.
- Z-score: event sequences for window/lag 100, threshold 2.

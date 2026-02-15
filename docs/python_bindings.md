# Python Bindings (PyO3 + maturin)

This repository includes a Python package built from Rust using PyO3 and maturin, with `uv` as the Python package and virtual environment manager.

## Quickstart

Prerequisites:
- Rust stable toolchain
- `uv` (`https://docs.astral.sh/uv/`)

From repo root:

```bash
uv venv --python 3.13 .venv
uv sync --group dev
uv run --python .venv/bin/python maturin develop --manifest-path crates/pyopenquant/Cargo.toml
uv run --python .venv/bin/python python -c "import openquant; print('openquant import ok')"
uv run --python .venv/bin/python pytest python/tests -q
```

Quick performance showcase:

```bash
uv run --python .venv/bin/python python python/benchmarks/benchmark_pipeline.py --iterations 30 --bars 2048
```

Build a wheel:

```bash
uv run --python .venv/bin/python maturin build --manifest-path crates/pyopenquant/Cargo.toml --out dist
```

## API Mapping (Initial Surface)

### `openquant.risk`
- `calculate_value_at_risk(returns, confidence_level)`
- `calculate_expected_shortfall(returns, confidence_level)`
- `calculate_conditional_drawdown_risk(returns, confidence_level)`

Input conventions:
- `returns`: list of floats
- `confidence_level`: float in `[0, 1]`

### `openquant.filters`
- `cusum_filter_indices(close, threshold)`
- `cusum_filter_timestamps(close, timestamps, threshold)`
- `z_score_filter_indices(close, mean_window, std_window, threshold)`
- `z_score_filter_timestamps(close, timestamps, mean_window, std_window, threshold)`

Input conventions:
- `close`: list of floats
- `timestamps`: list of strings formatted as `%Y-%m-%d %H:%M:%S`
- timestamp variants require `len(close) == len(timestamps)`

### `openquant.bars` (AFML Ch.2 event-driven bars; Rust core via PyO3)
- `build_time_bars(df, interval="1d")`
- `build_tick_bars(df, ticks_per_bar=50)`
- `build_volume_bars(df, volume_per_bar=100_000.0)`
- `build_dollar_bars(df, dollar_value_per_bar=5_000_000.0)`
- `bar_diagnostics(df)`

Input conventions:
- `df`: polars DataFrame with canonical OHLCV columns (`ts,symbol,open,high,low,close,volume,adj_close`)

### `openquant.data` (canonicalization + Rust-backed processing via PyO3)
- `load_ohlcv(path, symbol=None, return_report=False)`
- `clean_ohlcv(df, dedupe_keep="last", return_report=False)`
- `data_quality_report(df)`
- `align_calendar(df, interval="1d")`

Notes:
- File IO and column alias canonicalization happen in Python for ergonomics.
- Core cleaning, deduplication, quality reporting, and calendar alignment are executed in Rust through `_core.data`.

### `openquant.labeling`
- `triple_barrier_events(...)`
- `triple_barrier_labels(...)`
- `meta_labels(...)`

Input conventions:
- `close_timestamps`: list of `%Y-%m-%d %H:%M:%S` strings
- `close_prices`: list of floats
- `t_events`: event timestamps
- `target_timestamps` + `target_values`: target/volatility inputs
- barriers: `pt`, `sl`, `min_ret`, optional `vertical_barrier_times=[(start_ts, end_ts)]`
- optional side model for meta-labeling: `side_prediction=[(timestamp, side)]`

Label regimes:
- `triple_barrier_labels`: `{-1, 0, 1}`
- `meta_labels`: `{0, 1}` with side-adjusted returns

### `openquant.sampling`
- `get_ind_matrix(label_endtime, bar_index)`
- `get_ind_mat_average_uniqueness(ind_mat)`
- `seq_bootstrap(ind_mat, sample_length=None, warmup_samples=None)`

Input conventions:
- `label_endtime`: list of `(start_idx, end_idx)` tuples
- `bar_index`: list of integer bar indices
- `ind_mat`: nested list of 0/1 integers

### `openquant.bet_sizing`
- `get_signal(prob, num_classes, pred=None)`
- `discrete_signal(signal0, step_size)`
- `bet_size(w_param, price_div, func)`

Input conventions:
- `func` for `bet_size` must be `sigmoid` or `power`

### `openquant.portfolio`
- `allocate_inverse_variance(prices)`
- `allocate_min_vol(prices)`
- `allocate_max_sharpe(prices, risk_free_rate=0.0)`

Return conventions:
- tuple `(weights, portfolio_risk, portfolio_return, portfolio_sharpe)`

Input conventions:
- `prices`: rectangular nested list of floats (`rows=time`, `cols=assets`)

### `openquant.adapters` (polars-first research adapters)
- `to_polars_signal_frame(...)`
- `to_polars_event_frame(...)`
- `to_polars_indicator_matrix(...)`
- `to_polars_weights_frame(...)`
- `to_polars_frontier_frame(...)`
- `to_polars_backtest_frame(...)`
- `SignalStreamBuffer` for incremental signal updates in streaming/online notebook loops

### `openquant.pipeline` (mid-frequency research orchestration)
- `run_mid_frequency_pipeline(...)`
- `run_mid_frequency_pipeline_frames(...)`
- `summarize_pipeline(...)`

`run_mid_frequency_pipeline` contract:
- Inputs:
  - `timestamps`, `close`, `model_probabilities` (aligned 1:1)
  - `asset_prices` (`rows=time`, `cols=assets`)
  - optional `model_sides`, `asset_names`
  - params: `cusum_threshold`, `num_classes`, `step_size`, `risk_free_rate`, `confidence_level`
- Outputs:
  - `events`: event indices/timestamps/probabilities/sides
  - `signals`: event signal + aligned timeline signal
  - `portfolio`: max-sharpe allocation summary
  - `risk`: VaR/ES/CDaR + realized Sharpe
  - `backtest`: strategy returns, equity curve, drawdown metrics
  - `leakage_checks`: alignment and ordering guards

`run_mid_frequency_pipeline_frames` adds notebook-ready polars frames for `signals`, `events`, `backtest`, and `weights`.

### `openquant.research` (flywheel iteration helpers)
- `make_synthetic_futures_dataset(n_bars=..., seed=..., asset_names=...)`
- `run_flywheel_iteration(dataset, config=...)`

`run_flywheel_iteration` extends pipeline output with:
- cost model summary (turnover + vol/spread proxy),
- net/gross return and net Sharpe,
- promotion decision gates (statistical + economic).

### `openquant.viz` (backend-agnostic plotting payloads)
- `prepare_feature_importance_payload(...)`
- `prepare_drawdown_payload(...)`
- `prepare_regime_payload(...)`
- `prepare_frontier_payload(...)`
- `prepare_cluster_payload(...)`

Example:

```python
import openquant

signal_df = openquant.adapters.to_polars_signal_frame(
    timestamps=["2024-01-01 09:30:00", "2024-01-01 09:31:00"],
    signal=[0.2, -0.1],
    symbol="ES",
)
drawdown_payload = openquant.viz.prepare_drawdown_payload(
    timestamps=["2024-01-01 09:30:00", "2024-01-01 09:31:00"],
    equity_curve=[100.0, 99.5],
)

pipe = openquant.pipeline.run_mid_frequency_pipeline_frames(
    timestamps=["2024-01-01 09:30:00", "2024-01-01 09:31:00", "2024-01-01 09:32:00"],
    close=[100.0, 100.2, 100.1],
    model_probabilities=[0.55, 0.58, 0.52],
    asset_prices=[[100.0, 100.0], [100.2, 99.8], [100.1, 100.1]],
)
summary = openquant.pipeline.summarize_pipeline(pipe)
```

## Common Errors

- `ValueError: prices matrix must be rectangular`
  - Ensure every row in `prices` has identical length.
- `ValueError: close/timestamps length mismatch`
  - Align prices and timestamps one-to-one before calling timestamp APIs.
- `ValueError: invalid datetime ...`
  - Use `%Y-%m-%d %H:%M:%S` timestamp strings.
- `ModuleNotFoundError: No module named 'openquant'`
  - Re-run `uv run --python .venv/bin/python maturin develop --manifest-path crates/pyopenquant/Cargo.toml`.

## Notes

- The binding layer is intentionally thin: Rust `openquant` remains the source of truth.
- Polars-first adapters, plotting payload builders, and notebook flywheel helpers are included for research UX.

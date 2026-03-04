---
name: flywheel-runner
description: Run OpenQuant notebook-to-experiment flywheel iterations (single run and grid run), capture artifacts, and summarize ranked outcomes for promotion decisions.
---

# Flywheel Runner

Use this skill when you need to execute the research flywheel end-to-end and produce deterministic artifacts.

## What this skill does

1. Runs a baseline single experiment from TOML config.
2. Runs a grid (algo wheel) experiment from a grid TOML.
3. Produces artifact directories and leaderboard output.
4. Surfaces top run metrics and promotion flags.

## Commands

```bash
# Single run
uv run --python .venv/bin/python python experiments/run_pipeline.py \
  --config experiments/configs/futures_oil_baseline.toml \
  --out experiments/artifacts

# Grid run (parameter wheel)
uv run --python .venv/bin/python python experiments/run_pipeline.py \
  --config experiments/configs/futures_oil_baseline.toml \
  --grid-config experiments/configs/futures_oil_grid.toml \
  --out experiments/artifacts
```

## Expected outputs

- Single run folder containing:
  - `run_manifest.json`
  - `metrics.parquet`
  - `events.parquet`
  - `signals.parquet`
  - `weights.parquet`
  - `backtest.parquet`
  - `decision.md`
- Grid run folder containing:
  - `leaderboard.parquet`
  - one subfolder per run with the single-run files above

## File targets

- Runner: `experiments/run_pipeline.py`
- Base config: `experiments/configs/futures_oil_baseline.toml`
- Grid config: `experiments/configs/futures_oil_grid.toml`
- Output root: `experiments/artifacts/`

## Quality checks

```bash
uv run --python .venv/bin/python pytest python/tests/test_experiment_scaffold.py -q
```

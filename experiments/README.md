# Experiments Scaffold

This directory provides config-driven, reproducible research runs for OpenQuant.

## Run

```bash
uv run --python .venv/bin/python python experiments/run_pipeline.py \
  --config experiments/configs/futures_oil_baseline.toml \
  --out experiments/artifacts
```

Grid run (algo-wheel style sweep):

```bash
uv run --python .venv/bin/python python experiments/run_pipeline.py \
  --config experiments/configs/futures_oil_baseline.toml \
  --grid-config experiments/configs/futures_oil_grid.toml \
  --out experiments/artifacts
```

## Artifacts

Each run writes a deterministic folder:

- `run_manifest.json` (config hash, git sha, versions, seed)
- `metrics.parquet` (summary table)
- `events.parquet` (event stage output)
- `signals.parquet` (signal timeline)
- `weights.parquet` (portfolio weights)
- `backtest.parquet` (equity/returns/position)
- `decision.md` (promotion verdict and rationale)

Grid runs additionally include:
- `leaderboard.parquet` (ranked cross-run summary)
- one artifact subdirectory per grid entry with the same single-run files

## Notes

- Default config is futures-centric and oil-inclusive.
- Runner uses `openquant.research.run_flywheel_iteration` for notebook/script parity.

# Experiments Scaffold

This directory provides config-driven, reproducible research runs for OpenQuant.

## Run

```bash
uv run --python .venv/bin/python python experiments/run_pipeline.py \
  --config experiments/configs/futures_oil_baseline.toml \
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
- `equity_curve.svg` (deterministic equity line chart from `backtest.parquet`)
- `drawdown.svg` (deterministic drawdown line chart from `backtest.parquet`)
- `decision.md` (promotion verdict and rationale)

Plot artifacts are generated on every run from the deterministic backtest frame using
fixed-size SVG output (same config + seed -> same data + chart geometry).

## Notes

- Default config is futures-centric and oil-inclusive.
- Runner uses `openquant.research.run_flywheel_iteration` for notebook/script parity.

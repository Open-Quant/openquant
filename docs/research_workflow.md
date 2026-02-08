# Notebook-First Research Workflow

This guide defines the promotion path from hypothesis to candidate strategy in OpenQuant.

## Promotion Path

1. Exploratory notebook (`notebooks/python/*.ipynb`)
2. Scripted experiment (`experiments/run_pipeline.py` + TOML config)
3. Artifact + parity checks (`python/tests/test_experiment_scaffold.py`)
4. Candidate decision (`decision.md` in artifact run dir)

## Required Controls

- Use event-based sampling and leakage-safe validation paths.
- Use deterministic seeds and config-hashed run directories.
- Report both gross and net metrics; include turnover and cost estimate.
- Require statistical and economic gates for promotion.

## Local Commands

```bash
# setup
uv venv --python 3.11 .venv
uv sync --group dev
uv run --python .venv/bin/python maturin develop --manifest-path crates/pyopenquant/Cargo.toml

# notebook logic smoke
uv run --python .venv/bin/python python notebooks/python/scripts/smoke_all.py

# reproducible experiment run
uv run --python .venv/bin/python python experiments/run_pipeline.py --config experiments/configs/futures_oil_baseline.toml --out experiments/artifacts

# parity + scaffold tests
uv run --python .venv/bin/python pytest python/tests/test_experiment_scaffold.py -q
```

## Promotion Checklist

- [ ] Hypothesis has explicit economic rationale and invalidation condition.
- [ ] Eventing/labeling choices documented in notebook and config.
- [ ] Leakage checks passed (`inputs_aligned`, `event_indices_sorted`).
- [ ] Net Sharpe and realized Sharpe passed thresholds.
- [ ] Artifact bundle includes manifest, parquet outputs, and decision note.

## Anti-Patterns

- Random CV splits on overlapping financial labels.
- Promotion on gross-only metrics without costs.
- Notebook-only logic that cannot be reproduced from config.
- Hidden mutable state across notebook cells.

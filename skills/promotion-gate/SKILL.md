---
name: promotion-gate
description: Evaluate promotion readiness from flywheel outputs using statistical and economic gates, then produce a concise go/no-go decision note.
---

# Promotion Gate

Use this skill when deciding whether a strategy candidate moves from research iteration to deeper validation.

## What this skill does

1. Reads flywheel output (`summary`, `promotion`, `costs`).
2. Verifies required gates:
   - realized Sharpe
   - net Sharpe
   - alignment guard
   - event ordering guard
3. Produces a decision summary with blocking reasons.

## Core APIs

- `openquant.research.run_flywheel_iteration(...)`
- `openquant.pipeline.summarize_pipeline(...)`

## Decision contract

Required outputs:
- `promote_candidate` (bool)
- failed gate list
- key metrics (`realized_sharpe`, `net_sharpe`, `turnover`, `estimated_cost`)
- short rationale

## Artifact source

- `decision.md` and `metrics.parquet` in each experiment run folder.

## Quality checks

```bash
uv run --python .venv/bin/python pytest python/tests/test_experiment_scaffold.py -q
```

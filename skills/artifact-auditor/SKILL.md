---
name: artifact-auditor
description: Audit experiment artifact folders for completeness, schema consistency, and promotion traceability across single and grid runs.
---

# Artifact Auditor

Use this skill when validating whether research runs are publishable/reviewable by another quant without hidden notebook state.

## What this skill does

1. Verifies required files exist for each run.
2. Checks expected columns in `metrics.parquet` and `leaderboard.parquet`.
3. Confirms decision traceability (`decision.md` + `run_manifest.json`).
4. Reports missing files/fields with actionable remediation.

## Required files (single run)

- `run_manifest.json`
- `metrics.parquet`
- `events.parquet`
- `signals.parquet`
- `weights.parquet`
- `backtest.parquet`
- `decision.md`

## Required files (grid run)

- top-level `leaderboard.parquet`
- top-level `run_manifest.json`
- sub-run directories each containing the single-run file set

## Suggested audit commands

```bash
# quick schema check (example)
uv run --python .venv/bin/python - <<'PY'
import polars as pl
from pathlib import Path
p = Path("experiments/artifacts")
for lf in p.rglob("leaderboard.parquet"):
    df = pl.read_parquet(lf)
    required = {"run_name", "net_sharpe", "promote_candidate"}
    missing = required - set(df.columns)
    if missing:
        print(f"{lf}: missing {sorted(missing)}")
PY
```

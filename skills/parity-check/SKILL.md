---
name: parity-check
description: Verify parity between notebook-like pipeline execution and scripted experiment workflows so research findings are reproducible.
---

# Parity Check

Use this skill when confirming notebook results and experiment-runner results are consistent for core metrics.

## What this skill does

1. Runs notebook-like pipeline calls in Python.
2. Runs scripted experiment runner paths.
3. Compares key outputs:
   - event count
   - realized Sharpe
   - portfolio Sharpe
   - net Sharpe (where applicable)

## Primary test target

- `python/tests/test_experiment_scaffold.py`

## Command

```bash
uv run --python .venv/bin/python pytest python/tests/test_experiment_scaffold.py -q
```

## Escalation criteria

If metrics drift:
1. Check config mismatch (`pipeline`/`costs`/`gates`).
2. Check dataset seed and asset list.
3. Check run manifest `config_digest` and git SHA.

---
name: notebook-executor
description: Execute OpenQuant notebook code cells non-interactively, persist outputs to artifact paths, and fail fast on runtime errors.
---

# Notebook Executor

Use this skill when you need deterministic notebook execution in CI or local verification loops.

## What this skill does

1. Executes `.ipynb` code cells with plain Python.
2. Writes execution counts and stdout/error outputs back to notebook JSON.
3. Supports writing to a separate output path (`--out`) to avoid mutating source notebooks.

## Command

```bash
uv run --python .venv/bin/python notebooks/python/scripts/execute_notebook_cells.py \
  notebooks/python/08_algo_wheel_experiments.ipynb \
  --out /tmp/openquant-artifacts/08_algo_wheel_experiments.executed.ipynb
```

## File targets

- Executor: `notebooks/python/scripts/execute_notebook_cells.py`
- Notebook set: `notebooks/python/*.ipynb`

## Validation

```bash
uv run --python .venv/bin/python pytest python/tests/test_notebook_execute_script.py -q
```

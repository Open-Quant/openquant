---
name: notebook-executor
description: Execute OpenQuant notebook code cells non-interactively, persist outputs to artifact paths, and fail fast on runtime errors.
---

# Notebook Executor

Use this skill when you need notebook execution with normalized output metadata in CI or local verification loops.

## What this skill does

1. Executes `.ipynb` code cells through `nbclient` in the named `python3` Jupyter kernel, with the notebook directory as the working directory. Check that kernel's interpreter and imports; the launcher interpreter alone does not prove kernel selection.
2. Preserves execution counts, streams and rich outputs (including HTML and inline figures), and normalizes notebook metadata, positional cell IDs and execution timing metadata. This does not make unseeded calculations deterministic.
3. Writes in place by default; `--out` writes a separate notebook, overwriting an existing destination. It does not redirect files written by notebook code or isolate other side effects.
4. Stops on a cell execution error, writes the partially executed notebook, then raises an error. Kernel startup failures and other exceptions are not covered by that partial-output guarantee. `--timeout` controls the per-cell timeout (default 600 seconds).

## Command

Run from the repository root after installing the development environment and native extension. `--no-sync` preserves the prepared environment; it does not install missing dependencies.

```bash
uv run --no-sync --python .venv/bin/python notebooks/python/scripts/execute_notebook_cells.py \
  notebooks/python/08_algo_wheel_experiments.ipynb \
  --out /tmp/openquant-artifacts/08_algo_wheel_experiments.executed.ipynb
```

## File targets

- Executor: `notebooks/python/scripts/execute_notebook_cells.py`
- Notebook set: `notebooks/python/*.ipynb`

## Validation

```bash
uv run --no-sync --python .venv/bin/python pytest python/tests/test_notebook_execute_script.py -q
```

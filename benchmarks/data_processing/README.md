# Data Processing Benchmarks

This folder tracks Python-facing `openquant.data` performance metrics.

## Files

- `latest.json`: most recent benchmark run output.
- `baseline.json`: pinned comparison baseline for regression checks.

## Generate Latest Metrics

```bash
just py-bench-data
```

or directly:

```bash
uv run --python .venv/bin/python python python/benchmarks/benchmark_data_processing.py \
  --rows-per-symbol 200000 \
  --symbols 4 \
  --iterations 7 \
  --out benchmarks/data_processing/latest.json
```

## Compare Against Baseline

```bash
just py-bench-data-compare
```

## Refresh Baseline

After reviewing and accepting new performance:

```bash
cp benchmarks/data_processing/latest.json benchmarks/data_processing/baseline.json
```

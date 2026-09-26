# Data Processing Benchmarks

This folder tracks Python-facing `openquant.data` performance metrics.

## Files

- `latest.json`: output of a past run, kept as an example of the format. Timings depend on
  the machine, so no baseline is committed.

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

## Compare Against a Baseline

Timings are only comparable on the same machine, so keep a baseline locally: save a run you
accept, then pass it with `--baseline` (the script refuses a path that does not exist):

```bash
cp benchmarks/data_processing/latest.json /tmp/data_processing_baseline.json
uv run --python .venv/bin/python python python/benchmarks/benchmark_data_processing.py \
  --rows-per-symbol 200000 --symbols 4 --iterations 7 \
  --out benchmarks/data_processing/latest.json \
  --baseline /tmp/data_processing_baseline.json
```

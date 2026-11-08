# Data Processing Benchmarks

This folder tracks Python-facing `openquant.data` performance metrics.

## Files

- `latest.json`: output of a past run, kept as an example of the format. Timings depend on
  the machine, so no baseline is committed.

## Timing and input contract

The fixture is deterministic: each symbol has the same complete one-minute grid,
linearly increasing OHLC prices and constant volume. It has no random seed, missing
bars or duplicate rows. It does not benchmark difficult data-cleaning cases.
Dataset construction and writing temporary CSV/Parquet files are outside the timers.
Each operation and file path is warmed before measurement. `load_ohlcv` measures
reading **and cleaning**, not just parsing; repeated warmed reads do not establish
cold-disk throughput. Temporary input files are removed after the run.

Times are wall-clock milliseconds. Rows/s divides input rows by mean seconds.
Core MB/s uses Polars' estimated in-memory frame size; I/O MB/s uses the actual
CSV or Parquet file size, divided by 1,000,000. These byte counts differ, especially
for compressed Parquet, so the MB/s column is not a common bandwidth comparison.

The reported p95 is the sorted timing at zero-based index
`max(int(iterations * 0.95) - 1, 0)`: with five repetitions it is the fourth-smallest
value, and can be below the mean. It is not an interpolated percentile or a latency
guarantee. JSON p50 is the upper middle observation and std is population standard
deviation. Use positive row, symbol and repetition counts; a tiny run checks that
the script works, not representative performance.

## Generate Latest Metrics

Use the prepared Python environment and a release extension build described in
[Python setup](../../docs-site/src/content/docs/setup/python-bindings.md). The direct
commands below use `--no-sync` to preserve that installed build. Record the source
commit and any working-tree diff, exact command/workload, dependency lockfile,
Python/Polars/native versions, interpreter and loaded native-extension paths, and
release build profile. Also retain machine/CPU/OS, thread settings, warm-up/cache
conditions and background-load conditions: the output JSON does not capture this
receipt. The same machine alone does not control thermal state or competing work.

```bash
just py-bench-data
```

The recipe uses `uv run` with synchronization enabled. To preserve an already
prepared extension build, invoke the script directly with synchronization disabled:

```bash
uv run --no-sync --python .venv/bin/python python python/benchmarks/benchmark_data_processing.py \
  --rows-per-symbol 200000 \
  --symbols 4 \
  --iterations 7 \
  --out benchmarks/data_processing/latest.json
```

## Compare Against a Baseline

Use the same machine, input sizes, build, dependencies and thread settings for a
local comparison. The comparator matches function names and mean times only; it
does not check dataset dimensions or environment, and prints regressions without
failing a threshold. Save a run you accept and pass it with `--baseline` (a missing
path is rejected). The following commands overwrite `latest.json`; choose a new
`--out` path when preserving an earlier result. Before copying the baseline below,
run Generate Latest Metrics above on this machine with the same 200,000 rows per
symbol, 4 symbols and 7 iterations. Do not copy the committed example as a local
baseline: it used 100,000 rows per symbol and 5 iterations.

```bash
cp benchmarks/data_processing/latest.json /tmp/data_processing_baseline.json
uv run --no-sync --python .venv/bin/python python python/benchmarks/benchmark_data_processing.py \
  --rows-per-symbol 200000 --symbols 4 --iterations 7 \
  --out benchmarks/data_processing/latest.json \
  --baseline /tmp/data_processing_baseline.json
```

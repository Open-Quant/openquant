---
title: Python Bindings Setup
description: Build the PyO3 extension, install it into a virtual environment, and prove it imports.
status: authored
last_authored: '2026-11-05'
audience:
  - quant-dev
  - platform-engineering
sidebar:
  order: 3
---

The Python package is a pure-Python layer (`python/openquant/`) over a compiled
PyO3 extension (`crates/pyopenquant/`, built as `openquant._core`). This guide
builds both from one checkout. The distribution name is `pyopenquant`; the import
name is `openquant`. Do not `pip install openquant`: that name belongs to an
unrelated project, so the command can succeed and install the wrong package.
For packaged releases, check [release status](/project/governance/#versioning-and-releases).

Before you start: [Prerequisites](/setup/prerequisites/). You need a Rust
toolchain, a linker, and `uv`.

## Setup

Run from the repository root. Python 3.13 matches `just py-setup`; CI tests
the bindings on both Python 3.11 and 3.13.

```bash
uv venv --python 3.13 .venv
uv sync --group dev
uv run --python .venv/bin/python maturin develop --manifest-path crates/pyopenquant/Cargo.toml
uv run --python .venv/bin/python python -c "import openquant; print('openquant bindings import ok')"
uv run --python .venv/bin/python pytest python/tests -q
```

The first build compiles the Rust extension and its dependencies, including
`polars` and `nalgebra`. Budget **10–20 minutes on a cold cache**; later builds
are incremental. The import check prints `openquant bindings import ok` and
pytest should report no failures. The test count and runtime change with the
checkout and machine, so they are not a fixed acceptance value.

Re-run `maturin develop` after any Rust change before testing Python. Otherwise,
you can accidentally test an older extension against newer Python source.

### The `just` equivalents

The same flow is wrapped in the `justfile`, if you have
[`just`](https://github.com/casey/just):

```bash
just py-setup          # uv venv --python 3.13 .venv && uv sync --group dev
just py-develop        # maturin develop (debug build)
just py-develop-release  # maturin develop --release
just py-import-smoke   # import openquant; print('openquant bindings OK')
just py-test           # pytest python/tests -q
```

:::note[When environment setup fails]
`uv sync` also builds the project through maturin's PEP 517 hook. If it fails
before showing a useful compiler error, follow the explicit diagnostic build in
[Troubleshooting](/setup/troubleshooting/#uv-sync-fails-with-an-opaque-pep517-build-wheel-error).
Use the pinned toolchain rather than changing dependency versions to suppress
an error.
:::

### Debug or release

`maturin develop` compiles a **debug** extension by default: quick to build, and
correct, but unoptimised. That is fine for tests and most of the API. Numerically
heavy calls are a different matter — a full-series
[SADF](/modules/structural-breaks/) run can take minutes per model in a debug build.
Before timing anything or running a long research job, rebuild with
`maturin develop --release` (or `just py-develop-release`). For distributable wheels, pass `--release` to `maturin build` explicitly.

## Building a wheel

`maturin develop` installs in place, which is what you want for
development. To produce a distributable wheel:

```bash
uv run --python .venv/bin/python --with maturin \
  maturin build --release --manifest-path crates/pyopenquant/Cargo.toml --out dist
```

The wheel lands in `dist/`. It is a stable-ABI (`abi3`) wheel: `pyproject.toml`
sets `[tool.maturin] features = ["pyo3/abi3-py311"]`, so one wheel installs on
CPython 3.11 and every later version, on the platform it was built for. Build
from the repository root so maturin reads that setting from `pyproject.toml`.

## Smoke test the real surface

Once step 3 passes, this exercises the layer you actually came for. It
runs entirely on generated data, so it needs no market data feed:

```python
from openquant.research import make_synthetic_futures_dataset, run_flywheel_iteration

dataset = make_synthetic_futures_dataset(n_bars=192, seed=7)
result = run_flywheel_iteration(dataset)

print(result["summary"])
print(result["promotion"])
```

`result["summary"]` is a `polars.DataFrame`, `result["promotion"]` a
`dict` of boolean gates. For what the numbers mean, see
[Python Core Workflow](/workflows/python-core-workflow/).

## What you installed

`import openquant` re-exports two different kinds of thing
(`python/openquant/__init__.py`):

| Attribute | Kind | Comes from |
|---|---|---|
| `risk`, `filters`, `sampling`, `labeling`, `bet_sizing`, `portfolio`, `fracdiff`, `fast_ewma`, `volatility`, `codependence`, `backtest_stats`, `sample_weights`, `microstructural`, `strategy_risk`, `ensemble`, `structural_breaks`, `synthetic_bt`, `ef3m`, `streaming_hpc`, `hrp`, `hcaa`, `onc`, `cla`, `sb_bagging`, `dynamic_allocation` | compiled | `openquant._core`, i.e. the Rust crate |
| `bars`, `data`, `pipeline`, `backtesting_engine`, `cross_validation`, `feature_importance`, `hyperparameter_tuning`, `evaluation`, `feature_diagnostics`, `research`, `adapters`, `viz` | Python | `python/openquant/*.py` |

The extension registers 32 submodules: the 25 in the first row, re-exported
unchanged, plus `bars`, `data`, `pipeline`, `backtesting_engine`,
`cross_validation`, `feature_importance` and `hyperparameter_tuning`, which the
Python modules of the same name wrap (the compiled ones stay reachable as `openquant._core.<name>`).
Only the first row requires the compile step. That is why editing a file
under `python/openquant/` takes effect immediately, while editing
anything under `crates/` needs `maturin develop` re-run.

Runtime dependencies are `numpy>=1.26,<3`, `polars>=1.28.1,<2` and `pyarrow>=15,<26`
(`pyproject.toml`); `uv` installs them for you.

## Next

- [Python Core Workflow](/workflows/python-core-workflow/) — a runnable end-to-end research loop
- [Troubleshooting](/setup/troubleshooting/) — when the compile step fails

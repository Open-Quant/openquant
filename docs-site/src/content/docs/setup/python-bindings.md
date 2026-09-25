---
title: Python Bindings Setup
description: Build the PyO3 extension, install it into a virtual environment, and prove it imports.
status: authored
last_authored: '2026-09-25'
audience:
  - quant-dev
  - platform-engineering
sidebar:
  order: 3
---

OpenQuant is not on PyPI yet. The Python package is a thin pure-Python layer
(`python/openquant/`) over a compiled PyO3 extension
(`crates/pyopenquant/`, built as `openquant._core`), so getting it means
building it. It will be published as `pyopenquant` with the import name
`openquant`. Do not `pip install openquant`: that name on PyPI belongs to an
unrelated project, so the command succeeds and installs the wrong package.

Before you start: [Prerequisites](/setup/prerequisites/). You need a Rust
toolchain, a linker, and `uv`.

## Setup

Four commands, from the repository root. The `python` jobs in
`.github/workflows/ci.yml` run the same build on every PR, on Python 3.11
and 3.13 (CI installs `maturin` from the dev dependency group rather than
with `--with`; the result is the same).

```bash
# 1. Create an isolated interpreter. CI builds against 3.11 and 3.13.
uv venv --python 3.11 .venv

# 2. Compile the extension and install it into that environment.
uv run --python .venv/bin/python --with maturin \
  maturin develop --manifest-path crates/pyopenquant/Cargo.toml

# 3. Prove it imports.
uv run --python .venv/bin/python python -c "import openquant; print('openquant bindings import ok')"

# 4. Prove it works.
uv run --python .venv/bin/python --with pytest pytest python/tests -q
```

Step 2 compiles the whole Rust workspace — `polars`, `nalgebra` and their
dependency trees. Budget **10–20 minutes on a cold cache**; subsequent
runs are incremental and take seconds.

Expected output from steps 3 and 4:

```
openquant bindings import ok
```

```
................................                                         [100%]
32 passed in 2.52s
```

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

:::caution[`just py-setup` is not the same as the CI path]
`py-setup` pins Python **3.13** where CI uses **3.11**, and it uses
`uv sync` rather than `--with maturin`. `uv sync` builds the project
through maturin's PEP 517 hook, which swallows the compiler's error
output and reports only `returned non-zero exit status 1` — see
[Troubleshooting](/setup/troubleshooting/#uv-sync-fails-with-an-opaque-pep517-build-wheel-error).
Prefer the four explicit commands above when something is going wrong.
:::

### Debug or release

`maturin develop` compiles a **debug** extension by default: quick to build, and
correct, but unoptimised. That is fine for tests and most of the API. Numerically
heavy calls are a different matter — a full-series
[SADF](/modules/structural-breaks/) run can take minutes per model in a debug build.
Before timing anything or running a long research job, rebuild with
`maturin develop --release` (or `just py-develop-release`). `maturin build`, used for
wheels below, already builds in release mode.

## Building a wheel

`maturin develop` installs in place, which is what you want for
development. To produce a distributable wheel:

```bash
uv run --python .venv/bin/python --with maturin \
  maturin build --manifest-path crates/pyopenquant/Cargo.toml --out dist
```

The wheel lands in `dist/` and is installable into any interpreter of the
**same** Python minor version and platform it was built for — the
extension is ABI-specific. There is no `abi3` configuration in
`crates/pyopenquant/Cargo.toml`, so a 3.11 wheel will not load on 3.12.

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
| `risk`, `filters`, `sampling`, `labeling`, `bet_sizing`, `portfolio`, `fracdiff`, `fast_ewma`, `volatility`, `codependence`, `backtest_stats`, `sample_weights`, `microstructural`, `strategy_risk`, `ensemble`, `structural_breaks`, `synthetic_bt`, `ef3m`, `streaming_hpc`, `hrp`, `hcaa`, `onc`, `cla`, `sb_bagging` | compiled | `openquant._core`, i.e. the Rust crate |
| `bars`, `data`, `feature_diagnostics`, `pipeline`, `research`, `adapters`, `viz` | Python | `python/openquant/*.py` |

The extension registers 27 submodules: the 24 in the first row, re-exported
unchanged, plus `bars`, `data` and `pipeline`, which the Python modules of the
same name wrap (the compiled ones stay reachable as `openquant._core.<name>`).
Only the first row requires the compile step. That is why editing a file
under `python/openquant/` takes effect immediately, while editing
anything under `crates/` needs `maturin develop` re-run.

Runtime dependencies are `polars>=1.0,<2` and `pyarrow>=15,<20`
(`pyproject.toml`); `uv` installs them for you.

## Next

- [Python Core Workflow](/workflows/python-core-workflow/) — a runnable end-to-end research loop
- [Troubleshooting](/setup/troubleshooting/) — when the compile step fails

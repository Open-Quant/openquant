---
title: Python Bindings Setup
description: Build the PyO3 extension, install it into a virtual environment, and prove it imports.
status: authored
last_authored: '2026-11-07'
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
the bindings in a configured Python 3.11/3.13 matrix; that configuration alone is not
evidence of a successful run on your platform.

```bash
uv venv --python 3.13 .venv
uv sync --locked --group dev --no-install-project
uv run --no-sync --python .venv/bin/python maturin develop --release --manifest-path crates/pyopenquant/Cargo.toml
uv run --no-sync --python .venv/bin/python python -c "import openquant; print('openquant bindings import ok')"
uv run --no-sync --python .venv/bin/python pytest python/tests -q
```

The first build compiles the Rust extension and its dependencies, including
`polars` and `nalgebra`; build time depends on the machine and cache. The locked sync
installs runtime and development dependencies without building the project. The explicit
maturin command then builds and installs the extension; `--no-sync` prevents later `uv run`
commands from automatically synchronizing and rebuilding the project first. The import check prints `openquant bindings import ok` and
pytest should report no failures. The test count and runtime change with the
checkout and machine, so they are not a fixed acceptance value.

Re-run `maturin develop` after any Rust change before testing Python, then restart
any interpreter or notebook kernel that already imported OpenQuant. Rebuilding
the file does not replace the native library already loaded into a running process.
Otherwise, you can test an older extension against newer Python source.

### Existing `just` shortcuts

The repository also has shortcuts in `justfile`, if you have
[`just`](https://github.com/casey/just):

```bash
just py-setup          # uv venv --python 3.13 .venv && uv sync --group dev
just py-develop        # maturin develop (debug build)
just py-develop-release  # maturin develop --release
just py-import-smoke   # import openquant; print('openquant bindings OK')
just py-test           # pytest python/tests -q
```

These recipes are not exact equivalents of the explicit sequence above: `py-setup`
currently uses an unlocked `uv sync` that installs the project, and the `py-*` commands
above use `uv run` without `--no-sync`. They can therefore trigger environment synchronization
and an additional project build. Use the explicit commands when you need to isolate the
native build or preserve the environment just prepared from the lockfile.

:::note[When environment setup fails]
A plain `uv sync` (including `just py-setup`) builds the project through maturin's
PEP 517 hook; `--no-install-project` skips that project build. If a project build fails
before showing a useful compiler error, follow the explicit diagnostic build in
[Troubleshooting](/setup/troubleshooting/#uv-sync-fails-with-an-opaque-pep517-build-wheel-error).
Use the pinned toolchain rather than changing dependency versions to suppress
an error.
:::

### Debug or release

`maturin develop` without `--release` compiles a **debug** extension: unoptimised,
with Rust debug checks that can expose different failure behavior on invalid inputs. That is fine for tests and most of the API. Numerically
heavy calls are a different matter — a full-series
[SADF](/modules/structural-breaks/) run can take minutes per model in a debug build.
Before timing anything or running a long research job, rebuild with
`maturin develop --release` (or `just py-develop-release`). For distributable wheels, pass `--release` to `maturin build` explicitly.

## Building a wheel

`maturin develop` installs into the selected environment for development. After the
setup above has installed maturin and the dependencies, produce a distributable wheel:

```bash
uv run --no-sync --python .venv/bin/python \
  maturin build --release --manifest-path crates/pyopenquant/Cargo.toml --out dist
```

The wheel lands in `dist/`. It is a stable-ABI (`abi3`) wheel: `pyproject.toml`
sets `[tool.maturin] features = ["pyo3/abi3-py311"]`, so one wheel installs on
compatible CPython 3.11+ builds on the platform it was built for. The wheel tag
is an ABI compatibility declaration, not evidence that every later interpreter
has passed this project's tests; see the [configured test matrix](/setup/prerequisites/#version-floors-and-why). Build
from the repository root so maturin reads that setting from `pyproject.toml`.

### Verify the packaged installation separately

A successful build does not prove an import came from the wheel: the checkout or an editable
installation can mask it. Create a separate test environment, replace the quoted filename
with the compatible wheel just built, and inspect imports in isolated Python mode:

```bash
uv venv --python 3.13 .venv-wheel-check
uv pip install --python .venv-wheel-check/bin/python "dist/REPLACE_WITH_WHEEL_FILENAME.whl"
.venv-wheel-check/bin/python -I -c "import sys, openquant, openquant._core; print(sys.executable); print(openquant.__file__); print(openquant._core.__file__)"
```

Use a new environment path if `.venv-wheel-check` already exists. `-I` excludes the current
working directory and `PYTHONPATH` from imports; this environment has no editable checkout
installation. Both package paths should point into its `site-packages`, not `python/openquant`
in the source tree. Check from outside the checkout as well when investigating path problems.
The install command also installs runtime dependencies; the wheel does not bundle NumPy,
Polars or PyArrow. This is an import check, not the full test suite or cross-platform validation.

## Smoke test the real surface

Once the import check passes, this exercises the layer you actually came for. It
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
Both rows require an installed extension: package initialization imports `_core`,
and the Python wrappers call it. Editing Python source does not itself require
recompilation, but a running interpreter may retain previously imported modules.
Restart it to test those edits. Editing Rust source requires rebuilding the
extension and restarting any process that loaded the old binary.

Runtime dependencies are `numpy>=1.26,<3`, `polars>=1.28.1,<2` and `pyarrow>=15,<26`
(`pyproject.toml`); `uv` installs them for you.

## Next

- [Quickstart](/quickstart/) — interpret the first synthetic result
- [Research journey](/workflows/research-journey/) — trace fold-local fitting and saved predictions
- [Python Core Workflow](/workflows/python-core-workflow/) — the convenience pipeline and its limits
- [Troubleshooting](/setup/troubleshooting/) — when the compile step fails

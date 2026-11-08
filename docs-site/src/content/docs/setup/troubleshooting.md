---
title: Troubleshooting
description: Symptom, cause and fix for the failures you actually hit building OpenQuant.
status: authored
last_authored: '2026-11-07'
audience:
  - quant-dev
  - platform-engineering
sidebar:
  order: 4
---

Each entry is a symptom you can match against your terminal, the cause, and
the fix. Entries marked **reproduced** were triggered and fixed on a real
machine while writing this page; entries marked **not reproduced** are
derived from reading the source or the CI configuration and have not been
observed in the original troubleshooting session. **Source-checked** entries identify
implementation behavior without claiming the failure was reproduced on your machine.

## Rust toolchain and Python interpreter disagree on architecture

**Symptom** — `maturin develop` dies almost immediately, while plain
`cargo build` works fine:

```
error[E0463]: can't find crate for `core`
  |
  = note: the `aarch64-apple-darwin` target may not be installed
  = help: consider downloading the target with `rustup target add aarch64-apple-darwin`
error: could not compile `cfg-if` (lib) due to 1 previous error
💥 maturin failed
  Caused by: Failed to build a native library through cargo
```

**Cause** — `maturin` builds for the architecture of the *Python
interpreter*, not of your default Rust toolchain, so it passes
`--target aarch64-apple-darwin` explicitly. If `rustup` was ever installed
under Rosetta you have an `x86_64-apple-darwin` toolchain, which has no
arm64 standard library. `cargo build` succeeds because it uses the host
target and never crosses.

Confirm the mismatch:

```bash
rustup show | head -3        # "Default host: x86_64-apple-darwin"
uv run --no-sync --python .venv/bin/python python -c "import platform; print(platform.machine())"  # selected interpreter
```

**Fix** — add the missing standard library:

```bash
rustup target add aarch64-apple-darwin
```

This resolved the recorded missing-standard-library failure. A cross-build also needs
the platform linker and SDK; installing a Rust target does not provide those. Prefer a
native toolchain matching the selected interpreter when possible.

**Status: reproduced.** This is the error that blocked the first bindings
build on the machine this page was written on, and `rustup target add`
cleared it.

## `uv sync` fails with an opaque `pep517 build-wheel` error

**Symptom** —

```
Error: command ['maturin', 'pep517', 'build-wheel', '-i', '...', '--compatibility', 'off', '--editable']
returned non-zero exit status 1
hint: This usually indicates a problem with the package or the build environment.
```

**Cause** — a plain `uv sync` installs this project through maturin's PEP 517 hook.
The final nonzero status is a summary; inspect the earlier output for the actual cause,
which can include architecture, linker, compiler or build-environment errors.

**Fix** — capture the whole log and read upward from the bottom for the
first `error[E...]` or `error:` line:

```bash
uv sync --locked --group dev > sync.log 2>&1
rg -n "error|Caused by" sync.log
```

To isolate the project build, use the same environment as [Python bindings setup](/setup/python-bindings/).
Do not recreate an existing environment just to diagnose it:

```bash
uv sync --locked --group dev --no-install-project
uv run --no-sync --python .venv/bin/python maturin develop --release --manifest-path crates/pyopenquant/Cargo.toml
```

The first command can still fail while resolving or installing dependencies; it only
skips installing OpenQuant itself. The second exposes the native build directly without
another automatic project sync.

**Status: reproduced.**

## `maturin: command not found`

**Symptom** — a bare `maturin develop` is not found by the shell, or a
`uv run --no-sync ... maturin` invocation cannot find maturin in its environment.

**Cause** — maturin is a build-system requirement and a dev dependency, not a
guaranteed global command. A successful isolated PEP 517 build does not establish that
maturin is installed in your shell's environment. Unlike a bare shell command,
`just py-develop` uses `uv run`, which normally synchronizes the project's dependencies;
inspect its full error rather than assuming a global PATH failure.

**Fix** — from an existing checkout environment, install the locked development tools
and then invoke maturin through that environment:

```bash
uv sync --locked --group dev --no-install-project
uv run --no-sync --python .venv/bin/python maturin --version
```

If `.venv` does not exist, follow [Python bindings setup](/setup/python-bindings/) first.
There is no need to install a separate global maturin merely to use these commands.

**Status: source-checked.** The dependency declarations and invocation paths are read
from `pyproject.toml` and `justfile`; no missing-tool failure was induced in this session.

## `import openquant` fails after a successful build

**Symptom** — `maturin develop` reports success, then:

```
ModuleNotFoundError: No module named 'openquant'
```

or

```
ImportError: dynamic module does not define module export function (PyInit__core)
```

**Cause** — the first error can mean the package is absent from the selected
interpreter's environment. The second means the loaded native library lacks
the expected `PyInit__core` export; check for a stale or incorrectly named
extension instead of assuming it is the same installation-path problem.

The default maturin build enables `pyo3/abi3-py311` in `pyproject.toml`.
Its `cp311-abi3` wheel uses CPython's stable ABI for Python 3.11 and later;
a different supported CPython minor version alone does not require a rebuild.
Platform and architecture compatibility still matter. A custom build without
the stable-ABI feature may be interpreter-specific.

**Fix** — always name the interpreter explicitly, on both sides. Every
`just py-*` build/test recipe selects `.venv/bin/python`. For diagnosis, suppress
automatic syncing so the check observes the existing environment:

```bash
uv run --no-sync --python .venv/bin/python python -c "import sys; print(sys.executable); import openquant; print(openquant.__file__); import openquant._core; print(openquant._core.__file__)"
```

Printing the interpreter first still gives useful evidence when the import
fails. If the paths point to another checkout or environment, install into
the intended one and remove unintended path overrides. A recreated `.venv`
needs the package and its dependencies installed again; a compatible abi3
wheel can be reused. If the native export error remains, rebuild from this
checkout with the explicit maturin command in [bindings setup](/setup/python-bindings/)
and check which extension path is being loaded. Restart processes that imported the old binary.

**Status: not reproduced.** The reasoning is read off maturin's build
environment and the `[lib] name = "_core"` / `module-name =
"openquant._core"` wiring in `crates/pyopenquant/Cargo.toml` and
`pyproject.toml`; the failure itself was not triggered here.

## `AttributeError: 'builtins.PySeries' object has no attribute '_export'`

**Symptom** — `import openquant` works, but any `openquant._core.data.*_df`
call that takes a polars `DataFrame` raises this `AttributeError`, followed by
`while processing 'pydf'`.

**Cause** — the installed Python polars is older than 1.28.1. The extension
takes each column through `pyo3-polars` 0.28, which calls the polars
`Series._export` method; older Python polars releases do not have it.
`pyproject.toml` requires `polars>=1.28.1`, so this only happens when polars
was installed or pinned outside it.

**Fix** — upgrade polars in the environment the extension runs in:

```bash
uv pip install --python .venv/bin/python 'polars>=1.28.1,<2'
```

**Status: reproduced** with polars 1.25.2 and 1.27.1 against the extension
built from this repository; 1.28.1, 1.29.0, 1.32.3 and 1.44.2 pass
`python/tests/test_data_module.py`. (Before #220 the extension used a patched
`pyo3-polars` 0.20.0 to avoid a `compat_level` `TypeError` on polars 1.32.3
and later; that patch is gone.)

## `linker 'cc' not found` (Linux)

**Symptom** — every Rust build fails at the link step, no matter which
crate.

**Cause** — `rustc` shells out to the system C compiler to link. `rustup`
installs neither.

**Fix** —

```bash
sudo apt-get install -y build-essential      # Debian/Ubuntu
sudo dnf groupinstall -y "Development Tools" # Fedora/RHEL
```

On macOS the equivalent is `xcode-select --install`.

**Status: not reproduced** — this page was written on macOS, where the
Command Line Tools were already present.

## The build is killed, or the machine swaps to a halt

**Symptom** — `cargo` or `maturin` output ends in `signal: 9, SIGKILL` or
`Killed`, usually while compiling `polars`.

**Cause** — this workspace depends on `polars 0.55` and `nalgebra 0.35`.
Cargo compiles as many crates in parallel as you have cores, and several
polars crates are individually memory-hungry. On a machine with less than
about 2 GB of RAM per core, the peak overlaps and the OOM killer wins.

**Fix** — trade wall-clock for memory:

```bash
cargo build -j 2                # or: export CARGO_BUILD_JOBS=2
```

`maturin` inherits `CARGO_BUILD_JOBS`, so export it before
`maturin develop` rather than passing `-j` to maturin.

**Status: not reproduced.** The dependency set is read from
`crates/openquant/Cargo.toml`; the OOM was not induced here, and the
2 GB-per-core figure is a rule of thumb, not a measurement.

## Python runs out of memory building bars from a large dataset

**Symptom** — `build_time_bars` / `build_volume_bars` /
`build_dollar_bars` on a multi-million-row frame consumes far more memory
than the frame itself, and may be killed.

**Cause** — this is structural, not a leak. `openquant.bars` crosses the
PyO3 boundary with plain Python objects. In
`python/openquant/bars.py`, `_build_by_symbol` does, per symbol:

```python doc-check=skip
rows = rust_builder(
    [str(x) for x in sdf["ts"].to_list()],
    [float(x) for x in sdf["close"].to_list()],
    [float(x) for x in sdf["volume"].to_list()],
    param,
)
```

Three fully materialised Python lists go in — one of them a list of
formatted timestamp *strings* — and a Python list of 9-tuples comes back,
before any of it becomes a DataFrame. A Python `str` and `float` cost
tens of bytes each in object overhead alone, so peak memory is a large
multiple of the Arrow-backed frame you started from.

**Fix** — reduce the data retained at once without splitting a symbol's
ordered input stream. Load and process complete symbols separately, and
persist completed outputs instead of retaining every symbol's frame. This
can reduce simultaneous input/output storage, but it does not remove the
per-symbol Python-list cost.

Do not treat independent time chunks followed by concatenation as equivalent
to one call. Each call starts a new aggregation state. Tick, volume and dollar
bars can lose unfinished bars at a boundary; time bars retain a trailing
partial bar and start the next call with a new anchor. A calendar-date boundary
does not in general fix either problem. The Python wrappers do not expose
state to carry an unfinished bar into the next call.

If one complete symbol still exceeds the memory budget, use more memory or
implement state-preserving streaming and verify it against a single-pass
reference. There is no general safe chunk size for the existing wrappers.

**Status: source-checked.** The materialization and per-call state behavior
come from `python/openquant/bars.py` and the Rust bar builders; no peak-memory
profile was measured here.

## The slow structural-break test looks stalled

**Symptom** — `cargo test` sits on `test_sadf_test` for a long time.

**Cause** — it is not stalled. The test is annotated in
`crates/openquant/tests/structural_breaks.rs`:

```rust
#[ignore = "long-running hotspot; run explicitly with `cargo test -p openquant --test structural_breaks test_sadf_test -- --ignored`"]
```

It is excluded from the default run, so if you are seeing it you passed
`--ignored` or `--include-ignored`.

**Fix** — use the fast path for iteration and run the slow test
deliberately:

```bash
cargo test --workspace --lib --tests --all-features -- --skip test_sadf_test
cargo test -p openquant --test structural_breaks test_sadf_test -- --ignored
```

**Status: reproduced** (the `#[ignore]` annotation and its reason string
are verbatim from the test file).

## An optional pandas or scikit-learn import fails

The base package declares NumPy, Polars and PyArrow as runtime dependencies. Pandas
and scikit-learn are not included in the repository's development dependency group,
and no pandas/scikit-learn installation extra is declared. A successful base import
therefore does not establish that these optional integrations are available.

| Symptom | Relevant operation | Next step |
| --- | --- | --- |
| `RuntimeError: pandas is not installed` | `adapters.to_pandas(frame)` checks for pandas before delegating to Polars | Add pandas to the environment that calls the bridge, or continue using the Polars frame. |
| `ModuleNotFoundError: No module named 'sklearn'` | Your code imports an external estimator or `GridSearchCV` | Add the distribution named `scikit-learn` to that research environment. The import name is `sklearn`. |
| A custom estimator behaves differently when scikit-learn is present | `feature_importance` attempts `sklearn.base.clone`; without it, or on `TypeError`, it uses `deepcopy` | Inspect the [estimator cloning contract](/modules/feature-importance/) and ensure each fold starts from the intended unfitted state. |

Inspect the interpreter you actually use:

```python
import importlib.util
import sys

print("interpreter:", sys.executable)
for module in ("openquant", "pandas", "sklearn"):
    print(module, "available" if importlib.util.find_spec(module) else "not installed")
```

An available module spec means Python can locate it, not that importing it or loading
its compiled dependencies will succeed. If an installed package fails to import, retain
the full exception and check dependency consistency in the same environment:

```bash
uv pip check --python .venv/bin/python
```

In a separate research project managed by uv, `uv add pandas` or `uv add scikit-learn`
records the chosen optional dependency in that project's metadata and lockfile. Running
those commands in this repository changes its project files. An ad hoc environment-only
install is not recorded in `uv.lock` and a later exact sync may remove it. Record optional
versions with the result and restart an existing notebook kernel after changing its
environment. The native OpenQuant examples do not require scikit-learn merely because
some APIs also accept scikit-learn-style estimators.

## Docs-site gates

Failures in `check:links`, `check:api-drift` and `check:content-schema`
are maintainer gates, not first-run problems. They are documented where
they are run, on [Local Build Setup](/setup/local-build/#when-a-gate-fails).

## The call succeeds, but the numbers disagree

First distinguish a contract mismatch from a numerical defect. Confirm the interpreter,
package and extension paths using the import checks above, then reduce the comparison to
one function with explicit inputs. A package version alone cannot identify the loaded code.
For validation exceptions or non-finite output, start with [Diagnose input errors](/setup/input-errors/).

| Difference you see | Compare before changing the calculation |
| --- | --- |
| Sharpe differs by a fixed scale | Match return frequency, `sqrt(periods_per_year)` and risk-free units. The low-level `backtest_stats.sharpe_ratio` takes a per-period risk-free value; the pipeline accepts an annual rate. See the [executed comparison](/workflows/python-core-workflow/). |
| Sharpe differs slightly on a short sample | The low-level ratio uses sample standard deviation (`ddof=1`); NumPy's default is `ddof=0`. Compare the same observations, including whether a leading zero was added to a backtest frame. |
| Allocator Sharpe differs from strategy Sharpe | The allocator reports its fitted portfolio on supplied price/return inputs. The convenience pipeline's strategy curve is a separate calculation; its weights do not drive that curve. Check [portfolio conventions](/modules/portfolio-optimization/) and [pipeline accounting](/modules/pipeline/). |
| A saved CLA result reports positive expected returns although every supplied forecast is zero | Earlier versions perturbed the last forecast. Current code preserves the vector and solves the equal-return minimum-variance point. Check the loaded revision and the [equal-return contract](/modules/cla/#equal-return-inputs-preserve-the-forecasts), then compare reported means with the original forecasts. |
| ES differs at a quantile boundary | This implementation averages observations strictly below its VaR cutoff. Ties can reduce the selected tail or make it empty. Compare the actual selected observations, not just the requested percentage. |
| Two CV runs have different training rows | Match label spans, their units and input order, the number of splits and the embargo fraction. Purging uses closed spans; embargo is based on sample positions. Use [split diagnostics](/modules/cross-validation/) to explain exclusions. |
| A hash matches but a result differs | Content hashing normalizes row order. Check the order consumed by the calculation, duplicate policy, protocol settings and loaded binary; a data hash is not a complete execution identity. |

Use a small deterministic example and an independent reference with the same conventions.
Keep the expected formula or reference implementation separate from the function under test.
Do not loosen tolerances, fill NaN with zero, or discard inconvenient observations merely to
make the results agree. If the matched comparison still differs, report the inputs, expected
and actual values, conventions and revision with the reproducer. Statistical sampling
uncertainty and floating-point rounding require different explanations.

## Still stuck

Open an issue with the output of:

```bash
rustc --version && cargo --version && uv --version
rustup show | head -5
uv run --no-sync --python .venv/bin/python python -c "import platform, sys; print(sys.executable, platform.machine(), sys.version)"
```

Include the exact failing command and its first substantive error, plus the interpreter
and extension paths when import succeeds. Redact credentials or private source locations
from logs before sharing them.

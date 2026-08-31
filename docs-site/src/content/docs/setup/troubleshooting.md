---
title: Troubleshooting
description: Symptom, cause and fix for the failures you actually hit building OpenQuant.
status: reviewed
last_validated: '2026-08-30'
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
observed here. Believe the second kind less.

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
python3 -c "import platform; print(platform.machine())"   # "arm64"
```

**Fix** — add the missing standard library:

```bash
rustup target add aarch64-apple-darwin
```

That is enough; the x86_64 host toolchain cross-compiles to arm64 without
further setup. To fix it properly, reinstall `rustup` natively (it will
tell you to: `warn: Rustup is not running natively`).

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

**Cause** — this is almost never a `uv` problem. `uv sync` builds this
project through `maturin`, and `uv` reports only maturin's exit status;
the actual compiler error is further up in the output and easy to scroll
past. In practice it is the architecture mismatch above, a missing linker,
or a Rust compile error.

**Fix** — capture the whole log and read upward from the bottom for the
first `error[E...]` or `error:` line:

```bash
uv sync --group dev > sync.log 2>&1; grep -n "^error" sync.log | head
```

Then fix *that*, not the `uv` message. If you only want the bindings and
not the full project environment, prefer the path CI uses, which reports
compiler errors directly:

```bash
uv venv --python 3.11 .venv
uv run --python .venv/bin/python --with maturin \
  maturin develop --manifest-path crates/pyopenquant/Cargo.toml
```

**Status: reproduced.**

## `maturin: command not found`

**Symptom** — `just py-develop`, or a hand-run `maturin develop`, exits
127.

**Cause** — `maturin` is not a global tool in this project. It is a
build-system requirement (`pyproject.toml`, `[build-system] requires`) and
a dev dependency (`[dependency-groups] dev`). Nothing installs it onto
your `PATH`.

**Fix** — get it into the environment you are running from, either by
materialising the dev group:

```bash
uv venv --python 3.11 .venv
uv sync --group dev
```

or per-invocation, which is what `.github/workflows/python-bindings.yml`
does:

```bash
uv run --python .venv/bin/python --with maturin \
  maturin develop --manifest-path crates/pyopenquant/Cargo.toml
```

**Status: reproduced** (`maturin` is absent from `PATH` on a machine that
has built this project successfully).

## `import openquant` fails after a successful build

**Symptom** — `maturin develop` reports success, then:

```
ModuleNotFoundError: No module named 'openquant'
```

or

```
ImportError: dynamic module does not define module export function (PyInit__core)
```

**Cause** — two different versions of the same mistake: the interpreter
importing is not the interpreter the extension was built for. `maturin
develop` installs into whichever environment it is *run inside*. The
extension is stamped with a specific ABI —
`PYO3_ENVIRONMENT_SIGNATURE="cpython-3.11-64bit"` appears in maturin's own
cargo invocation — and CPython will not load a module built for a
different minor version.

**Fix** — always name the interpreter explicitly, on both sides. Every
`just py-*` recipe does this for exactly this reason:

```bash
uv run --python .venv/bin/python python -c "import openquant, sys; print(sys.executable)"
```

If `sys.executable` is not the `.venv/bin/python` you built against,
that is the bug. Rebuild with the interpreter you intend to use; a
`.venv` recreated on a different Python minor version needs a rebuild,
not just a re-install.

**Status: not reproduced.** The reasoning is read off maturin's build
environment and the `[lib] name = "_core"` / `module-name =
"openquant._core"` wiring in `crates/pyopenquant/Cargo.toml` and
`pyproject.toml`; the failure itself was not triggered here.

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

**Cause** — this workspace depends on `polars 0.46` and `nalgebra 0.32`.
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

**Fix** — do not hand the whole history to one call. The function already
loops per symbol, so the remaining axis is time: slice with polars (which
stays in Arrow memory) and concatenate the bar frames.

```python doc-check=skip
import polars as pl
from openquant import bars

pieces = [
    bars.build_dollar_bars(chunk, dollar_value_per_bar=5_000_000.0)
    for chunk in df.sort("ts").iter_slices(2_000_000)
]
out = pl.concat(pieces, how="vertical").sort(["symbol", "ts"])
```

Choose the slice boundary on a date, not a row count, if you need bar
edges to be exactly reproducible — a bar that straddles a chunk boundary
will be split.

**Status: not reproduced.** The memory characteristic is read directly
from `python/openquant/bars.py`; no allocation profile was measured, and
the 2-million-row slice above is a starting point rather than a tuned
value.

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

## Docs-site gates

Failures in `check:links`, `check:api-drift` and `check:content-schema`
are maintainer gates, not first-run problems. They are documented where
they are run, on [Local Build Setup](/setup/local-build/#when-a-gate-fails).

## Still stuck

Open an issue with the output of:

```bash
rustc --version && cargo --version && uv --version
rustup show | head -5
python3 -c "import platform, sys; print(platform.machine(), sys.version)"
```

The first three lines resolve most of the entries above on sight.

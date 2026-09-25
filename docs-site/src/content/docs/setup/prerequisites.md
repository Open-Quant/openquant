---
title: Prerequisites
description: Toolchain OpenQuant requires, how to install it on each platform, and why each version floor exists.
status: authored
last_authored: '2026-09-25'
audience:
  - quant-dev
  - platform-engineering
sidebar:
  order: 1
---

You need four things: a Rust toolchain, a C linker, `uv`, and Git. Node is
needed only if you are working on the documentation site.

## Install

### macOS

```bash
xcode-select --install                                     # clang + linker
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Git ships with the Command Line Tools installed by the first command.

:::caution[Apple Silicon: match the architectures]
`rustup` will happily install an `x86_64-apple-darwin` toolchain on an
arm64 Mac (this happens if you ever ran it under Rosetta). The Python
bindings will then fail to build, because `maturin` asks `cargo` for the
architecture of *your Python interpreter*, not of your Rust toolchain.
Check with:

```bash
rustup show | head -3
python3 -c "import platform; print(platform.machine())"
```

If the first says `x86_64` and the second says `arm64`, see
[Troubleshooting → Rust toolchain and Python interpreter disagree on
architecture](/setup/troubleshooting/#rust-toolchain-and-python-interpreter-disagree-on-architecture).
:::

### Debian / Ubuntu

```bash
sudo apt-get update && sudo apt-get install -y build-essential git curl
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
curl -LsSf https://astral.sh/uv/install.sh | sh
```

`build-essential` is not optional: it supplies `cc`, which `rustc` invokes
as its linker. Without it every Rust build fails at the link step with
`linker 'cc' not found`.

### Windows

Use WSL2 and follow the Debian/Ubuntu instructions. The nightly workflow
runs the core crate's tests on `windows-latest` and `macos-latest`, but
the Python extension is built and tested on `ubuntu-latest` only, so a
native Windows build of the bindings is unverified by this project.

### Verify

```bash
rustc --version && cargo --version && uv --version && git --version
```

## Version floors, and why

| Tool | Floor | Where it is declared | Why |
|---|---|---|---|
| Rust | the pinned toolchain | `rust-toolchain.toml` (`channel = "1.98.1"`) | rustup reads `rust-toolchain.toml` in the repository root, so local builds and every CI job use the same compiler and the same clippy lints. No `rust-version` (MSRV) key is declared in any `Cargo.toml`; the pinned version is the only one CI proves. |
| Python | 3.11 minimum, **3.13 recommended** | `requires-python = ">=3.11"` in `pyproject.toml` | 3.11 is the floor the package metadata enforces at install time. The `python` jobs in `.github/workflows/ci.yml` build and test the extension on both 3.11 and 3.13 on every PR; 3.13 is what the `justfile` develops on. 3.12 is not tested. |
| `uv` | any recent release | not pinned | Every `just py-*` recipe shells out to `uv`, and CI installs it via `astral-sh/setup-uv@v5`. It is the project's only supported way to create the Python environment. |
| Node | 20 | `node-version: 20` in `.github/workflows/docs-pages.yml` | Docs site only. Astro 5 requires Node 18.17+; CI uses 20. |
| Bun | latest | `oven-sh/setup-bun@v2` in `docs-pages.yml` | Docs site only, and **optional** — see below. |

:::note[Which Python versions have evidence behind them]
- `pyproject.toml` — `requires-python = ">=3.11"`
- `.github/workflows/ci.yml`, `python` job — a matrix of `3.11` and `3.13`
- `justfile`, recipe `py-setup` — `uv venv --python 3.13 .venv`

Both ends of the supported range are built and tested on every PR. 3.12
sits between them and is expected to work, but nothing runs it.
:::

:::note[Bun is not required]
`docs-site/package.json` defines `check:docs` as a chain of `bun run`
commands, and the docs deploy workflow uses Bun. Nothing in the docs site
depends on Bun as a *runtime*, though — it is used purely as a package
manager and script runner. npm works:

```bash
cd docs-site
npm install
npx astro build
```

That is the path used to verify every docs command on this site. If you do
want Bun, install it with `curl -fsSL https://bun.sh/install | bash`.
:::

## Rust dependencies that need no action

The workspace vendors a patched `pyo3-polars` 0.20.0 (`vendor/pyo3-polars`,
wired up by a `[patch.crates-io]` entry in the root `Cargo.toml`). The patch
changes one call so that polars `DataFrame` arguments still work with Python
polars 1.32.3 and later. `vendor/README.md` describes the exact change and
what would let us drop it. The directory is checked in, so `cargo build`
resolves it with no extra setup.

Only the Python extension crate (`pyopenquant`) uses it. That crate ships as
a wheel built from this repository and is never published to crates.io
(`publish = false`), because Cargo ignores `[patch]` for registry
downloads. The `openquant` crate does not depend on `pyo3-polars`.

## Next

- [Local Build Setup](/setup/local-build/) — build and test the Rust core
- [Python Bindings Setup](/setup/python-bindings/) — build and import the Python package

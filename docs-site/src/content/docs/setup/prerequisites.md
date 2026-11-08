---
title: Prerequisites
description: Toolchain OpenQuant requires, how to install it on each platform, and why each version floor exists.
status: authored
last_authored: '2026-11-07'
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

For the shell-based contributor commands on this site, use WSL2 and follow
the Debian/Ubuntu instructions. The nightly workflow runs core Rust tests
on Windows and macOS. The release workflow also defines native Windows and
macOS Python wheel builds and smoke tests, including on
packaging pull requests. A passing run is needed to establish a particular wheel's result.
That wheel smoke coverage is narrower than the full Python suite, which the
main CI matrix runs on Ubuntu.

### Verify

```bash
rustc --version && cargo --version && uv --version && git --version
```

## Version floors, and why

| Tool | Floor | Where it is declared | Why |
|---|---|---|---|
| Rust | the pinned toolchain | `rust-toolchain.toml` (`channel = "1.98.1"`) | rustup reads `rust-toolchain.toml` in the repository root, so local builds and every CI job use the same compiler and the same clippy lints. No `rust-version` (MSRV) key is declared in any `Cargo.toml`; the pinned version is the only one CI proves. |
| Python | 3.11 minimum, **3.13 recommended** | `requires-python = ">=3.11"` in `pyproject.toml` | The main CI Python suite runs on 3.11 and 3.13; 3.13 is the `justfile` development default. The release workflow additionally smoke-tests wheels on 3.12. |
| `uv` | any recent release | not pinned | The documented Python environment recipes use `uv`; CI installs it via `astral-sh/setup-uv@v7`. |
| Node | 22.12 | Astro's package engine requirement; no repository Node version pin | Docs site only. Node executes Astro; Bun is used as the package manager and script runner in CI. |
| Bun | latest | `oven-sh/setup-bun@v2` in `docs-pages.yml` | Docs site only, and **optional** — see below. |

:::note[Which Python versions the workflows target]
- `pyproject.toml` — `requires-python = ">=3.11"`
- `.github/workflows/ci.yml`, `python` job — a matrix of `3.11` and `3.13`
- `justfile`, recipe `py-setup` — `uv venv --python 3.13 .venv`

The full Python suite covers 3.11 and 3.13 in main CI. The release workflow's
wheel smoke loop covers 3.11, 3.12 and 3.13 on its platform matrix; it runs
for releases, manual dispatches and pull requests matching its packaging
path filter. These are configured checks, not a claim that every Python
version allowed by the open-ended package metadata has been tested.
:::

:::note[Bun is not required]
`docs-site/package.json` defines `check:docs` as a chain of `bun run`
commands, and the docs deploy workflow uses Bun. Nothing in the docs site
depends on Bun as a *runtime*, though — it is used purely as a package
manager and script runner. npm works:

```bash
cd docs-site
npm ci
npm run build
```

This builds the site only. It does not run the full `check:docs` gate, whose
script currently invokes Bun for its component commands. To use that aggregate
gate as written, install Bun with `curl -fsSL https://bun.sh/install | bash`.
:::

## Rust dependencies that move together

The Python extension crate (`pyopenquant`) passes polars `DataFrame`s through
`pyo3-polars` 0.28, which is built for one polars release (0.55) and one PyO3
release (0.29). The three are upgraded together, and `openquant` uses the same
polars. The extension takes a frame from Python polars through
`Series._export`, so it needs Python polars 1.28.1 or later (the
`pyproject.toml` lower bound). Everything comes from crates.io; nothing is
vendored.

## Next

- [Local Build Setup](/setup/local-build/) — build and test the Rust core
- [Python Bindings Setup](/setup/python-bindings/) — build and import the Python package

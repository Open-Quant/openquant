---
title: Prerequisites
description: Toolchain OpenQuant requires, how to install it on each platform, and why each version floor exists.
status: reviewed
last_validated: '2026-08-30'
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

Use WSL2 and follow the Debian/Ubuntu instructions. The CI matrix
(`.github/workflows/`) runs `ubuntu-latest` only, so a native Windows
build is unverified by this project.

### Verify

```bash
rustc --version && cargo --version && uv --version && git --version
```

## Version floors, and why

| Tool | Floor | Where it is declared | Why |
|---|---|---|---|
| Rust | stable | `dtolnay/rust-toolchain@stable` in every CI workflow | **No MSRV is declared anywhere in this repo** — there is no `rust-version` key in any `Cargo.toml` and no `rust-toolchain.toml`. Both crates are `edition = "2021"`, and `pyo3 0.23` / `polars 0.46` pull the real floor well above edition 2021's own 1.56. Treat "current stable" as the only supported answer, because that is the only one CI proves. |
| Python | 3.9 minimum, **3.11 recommended** | `requires-python = ">=3.9"` in `pyproject.toml` | 3.9 is the floor the package metadata will enforce at install time. 3.11 is what `.github/workflows/python-bindings.yml` actually builds and tests against, so it is the version with evidence behind it. See the caveat below. |
| `uv` | any recent release | not pinned | Every `just py-*` recipe shells out to `uv`, and CI installs it via `astral-sh/setup-uv@v5`. It is the project's only supported way to create the Python environment. |
| Node | 20 | `node-version: 20` in `.github/workflows/docs-pages.yml` | Docs site only. Astro 5 requires Node 18.17+; CI uses 20. |
| Bun | latest | `oven-sh/setup-bun@v2` in `docs-pages.yml` | Docs site only, and **optional** — see below. |

:::note[The repo disagrees with itself about the Python version]
Three different Python versions are declared in-tree:

- `pyproject.toml` — `requires-python = ">=3.9"`
- `.github/workflows/python-bindings.yml` — `python-version: "3.11"`
- `justfile`, recipe `py-setup` — `uv venv --python 3.13 .venv`

Only 3.11 is exercised by CI. This page recommends 3.11 for that reason.
3.13 was built successfully while writing this page, but 3.9 and 3.10 have
not been tested by anyone here and the `>=3.9` claim should be treated as
aspirational rather than verified.
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

The workspace vendors a patched `pyo3-polars` (`vendor/pyo3-polars`, wired
up by a `[patch.crates-io]` entry in the root `Cargo.toml`). It is checked
in, so `cargo build` resolves it with no extra setup — but it does mean
`Cargo.lock` is authoritative and you should not delete it.

## Next

- [Local Build Setup](/setup/local-build/) — build and test the Rust core
- [Python Bindings Setup](/setup/python-bindings/) — build and import the Python package

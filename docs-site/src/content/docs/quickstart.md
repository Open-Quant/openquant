---
title: Quickstart
description: First 30-minute path to validate OpenQuant locally.
status: validated
last_validated: '2026-03-02'
audience:
  - quant-dev
  - platform-engineering
sidebar:
  order: 1
---

## 1. Clone and enter repository

```bash
git clone https://github.com/Open-Quant/openquant.git
cd openquant
```

## 2. Run fast validation tests

```bash
cargo test --workspace --lib --tests --all-features -- --skip test_sadf_test
```

Expected checkpoint:
- Workspace tests pass for non-slow paths.

## 3. Run slow structural break hotspot test

```bash
cargo test -p openquant --test structural_breaks test_sadf_test -- --ignored
```

Expected checkpoint:
- The explicit slow test completes without failures.

## 4. Build docs site

```bash
cd docs-site
bun run build
```

Expected checkpoint:
- Static output generated under `docs-site/dist/`.

## 5. Next steps

- [Local Build Setup](/setup/local-build/)
- [Python Bindings Setup](/setup/python-bindings/)
- [Rust Core Workflow](/workflows/rust-core-workflow/)

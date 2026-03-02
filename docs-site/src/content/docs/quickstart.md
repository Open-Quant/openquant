---
title: Quickstart
description: Execute the first-run OpenQuant validation path in under 30 minutes.
status: validated
last_validated: '2026-03-02'
audience:
  - quant-dev
  - platform-engineering
sidebar:
  order: 1
---

## Goal

Confirm that your local environment can run OpenQuant's core tests and documentation quality gates.

## Step 1: Clone and enter repository

```bash
git clone https://github.com/Open-Quant/openquant.git
cd openquant
```

## Step 2: Run fast library validation

```bash
cargo test --workspace --lib --tests --all-features -- --skip test_sadf_test
```

Expected result:
- non-slow test suite passes.

## Step 3: Run slow structural-break hotspot

```bash
cargo test -p openquant --test structural_breaks test_sadf_test -- --ignored
```

Expected result:
- explicit slow-path test passes.

## Step 4: Validate docs build and quality gates

```bash
cd docs-site
bun run check:docs
```

Expected result:
- `build`, `check:links`, `check:api-drift`, and `check:content-schema` all pass.

## Step 5: Continue to workflow docs

- [Rust Core Workflow](/workflows/rust-core-workflow/)
- [Python Core Workflow](/workflows/python-core-workflow/)
- [Module Reference Index](/modules/)

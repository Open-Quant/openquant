---
title: Python Bindings Setup
description: Baseline guidance for Python package workflow integration.
status: validated
last_validated: '2026-03-02'
audience:
  - quant-dev
  - platform-engineering
sidebar:
  order: 3
---

OpenQuant currently exposes Python workflow APIs used in data, bars, diagnostics, and pipeline tracks.

## Validation command

```bash
cargo test --workspace --lib --tests --all-features -- --skip test_sadf_test
```

## Core Python surfaces to review

- `openquant.data`
- `openquant.bars`
- `openquant.feature_diagnostics`
- `openquant.pipeline`

Reference: [API Surfaces](/module-reference/api-surfaces/)

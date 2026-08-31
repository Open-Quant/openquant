---
title: Python Bindings Setup
description: Baseline guidance for Python package workflow integration.
status: draft
banner:
  content: '<span class="doc-status doc-status--draft">Draft</span> This page is known to be incomplete. Treat its contents as provisional.'
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

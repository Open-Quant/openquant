---
title: Local Build Setup
description: Deterministic local build and validation sequence.
status: reviewed
last_validated: '2026-08-30'
banner:
  content: '<span class="doc-status doc-status--reviewed">Reviewed</span> A human has read this page end to end. It has not been verified line by line against the code.'
audience:
  - quant-dev
  - platform-engineering
sidebar:
  order: 2
---

## Build and test sequence

```bash
cargo test --workspace --lib --tests --all-features -- --skip test_sadf_test
cargo test -p openquant --test structural_breaks test_sadf_test -- --ignored
```

## Docs quality gates

```bash
cd docs-site
bun run build
bun run check:links
bun run check:api-drift
bun run check:content-schema
```

Expected checkpoints:
- Build succeeds.
- Link checks return zero broken internal links.
- API drift check is clean.
- Content schema check passes.

---
title: Troubleshooting
description: Common first-run failures and known fixes.
status: validated
last_validated: '2026-03-02'
audience:
  - quant-dev
  - platform-engineering
sidebar:
  order: 4
---

## Slow test appears stalled

- `test_sadf_test` is intentionally slow and should be run only via the explicit command in quickstart.

## Docs link check failures

- Run `bun run build` before `bun run check:links`.
- Confirm all internal links include `/openquant` base at build output level.

## API drift failures

- Re-run inventory generation from repo root and commit updated inventory if APIs changed intentionally.

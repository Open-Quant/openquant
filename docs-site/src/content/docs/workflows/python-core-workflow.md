---
title: Python Core Workflow
description: Python workflow for ingestion, diagnostics, and research pipeline orchestration.
status: validated
last_validated: '2026-03-02'
audience:
  - quant-dev
  - platform-engineering
sidebar:
  order: 2
---

## Workflow stages

1. Load and clean OHLCV (`openquant.data`).
2. Build bars (`openquant.bars`).
3. Run feature diagnostics (`openquant.feature_diagnostics`).
4. Run pipeline and summarize outputs (`openquant.pipeline`).

Reference APIs: [API Surfaces](/module-reference/api-surfaces/)

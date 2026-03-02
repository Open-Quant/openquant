---
title: Python Core Workflow
description: Python-first workflow for ingestion, bars, diagnostics, and mid-frequency pipeline execution.
status: validated
last_validated: '2026-03-02'
audience:
  - quant-dev
  - platform-engineering
sidebar:
  order: 2
---

## Workflow Objective

Run the core research loop in Python while preserving OpenQuant's leakage and diagnostics standards.

## Stage 1: Ingestion and quality controls

Namespace: `openquant.data`

Primary APIs:
- `load_ohlcv`
- `clean_ohlcv`
- `align_calendar`
- `data_quality_report`

## Stage 2: Event-driven bars

Namespace: `openquant.bars`

Primary APIs:
- `build_time_bars`
- `build_tick_bars`
- `build_volume_bars`
- `build_dollar_bars`

## Stage 3: Feature diagnostics

Namespace: `openquant.feature_diagnostics`

Primary APIs:
- `mdi_importance`
- `mda_importance`
- `sfi_importance`
- `orthogonalize_features_pca`
- `substitution_effect_report`

## Stage 4: Pipeline execution and reporting

Namespace: `openquant.pipeline`

Primary APIs:
- `run_mid_frequency_pipeline`
- `run_mid_frequency_pipeline_frames`
- `summarize_pipeline`

## Quality Criteria

- Feature diagnostics use leakage-safe validation defaults.
- Report artifacts include trial and split metadata.
- Results are reproducible from declared data and code revisions.

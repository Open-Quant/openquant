---
title: Reproducibility and Artifact Contracts
description: Minimum artifact and metadata contract for reproducible research outcomes.
status: validated
last_validated: '2026-03-02'
audience:
  - quant-dev
  - platform-engineering
sidebar:
  order: 2
---

## Required Metadata

- code revision and branch/tag
- data snapshot/version
- split protocol details (including purge/embargo)
- model and hyper-parameter settings
- runtime environment metadata

## Required Artifacts

- summary report with key diagnostics
- module-level outputs used for promotion decisions
- benchmark/performance context where applicable
- decision memo or traceable approval reference

## Acceptance Checklist

- A peer can rerun and reproduce materially equivalent outputs.
- Inputs and assumptions are discoverable without tribal context.

---
title: Notebook Research Workflow
description: Notebook-first research flow with promotion controls for institutional settings.
status: draft
audience:
  - quant-dev
  - platform-engineering
sidebar:
  order: 3
---

## Scope

This workflow complements the `OQ-nbr` track and defines notebook behavior expected before experiment promotion.

## Required Controls

- Leakage-safe splits for overlapping labels.
- Explicit trial registry for multiple-testing awareness.
- Deterministic artifact bundle per run.

Every notebook under `notebooks/python/` follows the [notebook contract](/workflows/research-notebook-contract/):
a runbook's sections run Setup, Hypothesis, Data, Method, Results, Analysis, Promotion decision,
Self-review checklist and Reproducibility, an API tour makes no claims, and both end with a
reproducibility footer. `just notebooks-lint` enforces it in CI.

## Promotion Baseline

A notebook result is promotion-eligible only when:
- assumptions and costs are disclosed,
- split and embargo controls are documented,
- strategy diagnostics include stability evidence beyond a single Sharpe estimate.

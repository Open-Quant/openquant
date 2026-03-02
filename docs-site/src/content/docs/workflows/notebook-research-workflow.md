---
title: Notebook Research Workflow
description: Notebook-first research flow with promotion controls for institutional settings.
status: in_review
last_validated: '2026-03-02'
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

## Promotion Baseline

A notebook result is promotion-eligible only when:
- assumptions and costs are disclosed,
- split and embargo controls are documented,
- strategy diagnostics include stability evidence beyond a single Sharpe estimate.

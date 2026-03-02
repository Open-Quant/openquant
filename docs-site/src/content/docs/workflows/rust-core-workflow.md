---
title: Rust Core Workflow
description: Canonical Rust workflow from event sampling to portfolio/risk diagnostics.
status: validated
last_validated: '2026-03-02'
audience:
  - quant-dev
  - platform-engineering
afml_chapter:
  - CHAPTER 2
  - CHAPTER 3
  - CHAPTER 7
  - CHAPTER 11
  - CHAPTER 12
  - CHAPTER 14
  - CHAPTER 16
sidebar:
  order: 1
---

## Workflow Objective

Produce leakage-aware model evaluation and portfolio diagnostics using the Rust library API surface.

## Stage 1: Data structures and event filters

Modules:
- `data_structures`
- `filters`

Deliverable:
- event-based bars and filtered timestamps ready for labeling.

## Stage 2: Label engineering and sizing

Modules:
- `labeling`
- `bet_sizing`

Deliverable:
- event labels and executable sizing signals with explicit barriers and constraints.

## Stage 3: Leakage-safe validation and backtesting

Modules:
- `cross_validation`
- `backtesting_engine`

Deliverable:
- Purged/embargoed validation plus CPCV-aware distributional diagnostics.

## Stage 4: Risk and portfolio diagnostics

Modules:
- `backtest_statistics`
- `risk_metrics`
- `strategy_risk`
- `portfolio_optimization`, `hrp`, `hcaa`, `onc`, `cla`

Deliverable:
- risk-adjusted strategy diagnostics and portfolio allocation outputs.

## Quality Criteria

- Purging/embargo parameters explicitly documented.
- Multiple-testing context included when reporting Sharpe-family metrics.
- Portfolio-risk and strategy-failure-risk interpretations separated.

## Module Deep Links

- [Module Reference Index](/modules/)
- [API Surfaces](/module-reference/api-surfaces/)

---
title: Indexing and Discovery
description: Navigation model and search strategy for fast access to core OpenQuant capabilities.
status: validated
last_validated: '2026-03-02'
audience:
  - quant-dev
  - platform-engineering
sidebar:
  order: 3
---

## Primary Navigation Model

- **Getting started:** onboarding and initial validation.
- **Setup:** environment requirements, local build, troubleshooting.
- **Core workflows:** end-to-end Rust/Python paths.
- **Module reference:** detailed module pages.
- **Governance and release:** controls and operational policies.

## Recommended Discovery Paths

- New engineer: [Quickstart](/quickstart/) -> [Local Build Setup](/setup/local-build/) -> [Rust Core Workflow](/workflows/rust-core-workflow/)
- Research user: [Python Core Workflow](/workflows/python-core-workflow/) -> [By AFML Chapter](/module-reference/by-afml-chapter/) -> [Module Reference Index](/modules/)
- Reviewer/risk lead: [Methodology and Leakage Controls](/governance/methodology-and-leakage-controls/) -> [Reproducibility and Artifact Contracts](/governance/reproducibility-and-artifact-contracts/)

## Search Best Practices

- Search by module names (`cross_validation`, `backtesting_engine`) when you need APIs.
- Search by AFML concept (`purged`, `embargo`, `CPCV`, `deflated sharpe`) for methodology review.
- Search by outcome terms (`drawdown`, `labeling`, `portfolio`) for workflow entry points.

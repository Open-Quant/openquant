---
title: OpenQuant Documentation
description: Production-grade documentation for AFML-aligned quantitative research and portfolio engineering with OpenQuant.
template: splash
hero:
  title: OpenQuant Documentation
  tagline: Institutional-grade quantitative research documentation aligned to AFML chapters and production deployment controls.
  actions:
    - text: Start With Quickstart
      link: /quickstart/
      icon: right-arrow
    - text: Browse Modules
      link: /modules/
      variant: minimal
status: validated
last_validated: '2026-03-02'
audience:
  - quant-dev
  - platform-engineering
---

## Documentation Principles

- **Workflow-first navigation:** documentation is organized by practical execution path, not internal project structure.
- **Methodology traceability:** all major workflows map back to AFML chapter concepts and caveats.
- **Operational realism:** setup, performance, and governance pages include commands and decision criteria used in production.
- **Evidence over claims:** examples, formulas, and failure-mode notes are explicit on each module page.

## Recommended Reading Path

1. [Quickstart](/quickstart/)
2. [Prerequisites](/setup/prerequisites/)
3. [Local Build Setup](/setup/local-build/)
4. [Rust Core Workflow](/workflows/rust-core-workflow/)
5. [Python Core Workflow](/workflows/python-core-workflow/)
6. [By AFML Chapter](/module-reference/by-afml-chapter/)
7. [Module Reference Index](/modules/)
8. [Methodology and Leakage Controls](/governance/methodology-and-leakage-controls/)

## Core Workflow Lanes

### Event-Driven Data and Labeling (AFML Chapters 2-3)

- Event bars and filters: reduce microstructure noise and avoid naive time-bar bias.
- Triple barrier and meta-labeling: explicit event horizon and side/size separation.

### Validation and Diagnostics (AFML Chapters 4, 7, 8, 9)

- Purged and embargoed validation: leakage-resistant split policies.
- Feature diagnostics and tuning: substitution effects, orthogonalization, and disciplined search.

### Backtesting and Risk (AFML Chapters 11-16)

- CPCV and synthetic backtesting for robustness.
- Portfolio and strategy risk metrics with explicit assumptions and caveats.

## What Is Included In This V1 Institutional Baseline

- Setup and quickstart runbooks.
- Full module reference pages with APIs, formulas, examples, and notes.
- Indexes by workflow, AFML chapter, and module.
- Governance pages for leakage controls, reproducibility, and benchmark policy.

## What Is Not Included Yet

- Formal audit evidence bundles.
- Change-control approval logs integrated with external governance systems.

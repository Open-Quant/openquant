---
title: Versioning and Release Policy
description: Release and documentation versioning policy for OpenQuant.
status: validated
last_validated: '2026-03-02'
audience:
  - quant-dev
  - platform-engineering
sidebar:
  order: 4
---

## Policy

- Documentation changes ship alongside API or behavior changes.
- Breaking behavior changes require explicit migration notes.
- Benchmark and risk-method changes are called out in release notes.

## Release Readiness Gates

- tests and docs quality gates pass,
- critical module docs updated,
- benchmark regression policy evaluated.

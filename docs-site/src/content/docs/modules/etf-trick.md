---
title: "etf_trick"
description: "Synthetic ETF and futures roll utilities for realistic PnL path construction."
status: validated
last_validated: '2026-03-02'
audience:
  - quant-dev
  - platform-engineering
module: "etf_trick"
risk_notes:
  - "Verify contract calendar assumptions."
  - "Costs and rates should come from the same clock as price data."
rust_api:
  - "EtfTrick"
  - "get_futures_roll_series"
  - "FuturesRollRow"
sidebar:
  badge: Module
---

## Subject

**Position Sizing and Trade Construction**

## Why This Module Exists

Backtests must include financing, carry, and contract-roll mechanics to avoid optimistic bias.

## Key Public APIs

- `EtfTrick`
- `get_futures_roll_series`
- `FuturesRollRow`

## Mathematical Definitions

### ETF NAV Update

$$NAV_t=NAV_{t-1}(1+r_t-c_t)$$

### Roll Return

$$r^{roll}_t=\frac{F^{near}_t-F^{far}_t}{F^{far}_t}$$

## Implementation Examples

### Compute futures roll series

```rust
use openquant::etf_trick::get_futures_roll_series;

let roll = get_futures_roll_series(/* input tables */);
```

## Implementation Notes

- Verify contract calendar assumptions.
- Costs and rates should come from the same clock as price data.

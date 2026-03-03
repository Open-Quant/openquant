---
title: "etf_trick"
description: "Synthetic ETF and futures roll utilities for realistic PnL path construction."
status: validated
last_validated: '2026-03-02'
audience:
  - quant-dev
  - platform-engineering
module: "etf_trick"
api_surface: "rust-only"
risk_notes:
  - "Verify contract calendar assumptions."
  - "Costs and rates should come from the same clock as price data."
  - "This module is Rust-only — no Python bindings are currently exposed."
rust_api:
  - "EtfTrick"
  - "EtfTrick::from_tables"
  - "EtfTrick::from_csv"
  - "EtfTrick::get_etf_series"
  - "get_futures_roll_series"
  - "FuturesRollRow"
  - "Table"
sidebar:
  badge: Module
---

## Subject

**Position Sizing and Trade Construction**

## Why This Module Exists

Backtests must include financing, carry, and contract-roll mechanics to avoid optimistic bias.

## Mathematical Foundations

### ETF NAV Update

$$NAV_t=NAV_{t-1}(1+r_t-c_t)$$

### Roll Return

$$r^{roll}_t=\frac{F^{near}_t-F^{far}_t}{F^{far}_t}$$

## Usage Examples

### Rust

#### Construct synthetic ETF series

```rust
use openquant::etf_trick::{EtfTrick, Table};

// Load open/close/allocation/cost tables from CSV
let etf = EtfTrick::from_csv(
    "open.csv", "close.csv", "alloc.csv", "costs.csv", Some("rates.csv"),
).unwrap();

// Generate synthetic ETF NAV series
let series = etf.get_etf_series(252).unwrap();
// Returns Vec<(date_string, nav_value)>
```

#### Compute futures roll-adjusted series

```rust
use openquant::etf_trick::{get_futures_roll_series, FuturesRollRow};

let rows: Vec<FuturesRollRow> = vec![/* ... */];
let adjusted = get_futures_roll_series(&rows, "backward", true).unwrap();
```

## API Reference

### Rust API

- `EtfTrick`
- `EtfTrick::from_tables`
- `EtfTrick::from_csv`
- `EtfTrick::get_etf_series`
- `get_futures_roll_series`
- `FuturesRollRow`
- `Table`

## Implementation Notes

- Verify contract calendar assumptions.
- Costs and rates should come from the same clock as price data.
- This module is Rust-only — no Python bindings are currently exposed.

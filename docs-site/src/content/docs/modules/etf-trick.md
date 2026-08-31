---
title: "etf_trick"
description: "Synthetic ETF and futures roll utilities for realistic PnL path construction."
status: generated
generated_from: src/data/moduleDocs.ts
last_generated: '2026-08-31'
banner:
  content: '<span class="doc-status doc-status--generated">Generated</span> Assembled automatically from <code>moduleDocs.ts</code>. No human has reviewed this page.'
audience:
  - quant-dev
  - platform-engineering
module: "etf_trick"
api_surface: "rust-only"
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

## Concept Overview

The ETF trick turns a series of futures contracts — each with its own roll, financing cost and carry — into one continuous, reinvestable price series that a backtest can treat like a tradable instrument. `EtfTrick` consumes aligned open, close, allocation and cost tables plus optional financing rates and produces a NAV series; `get_futures_roll_series` applies backward or forward roll adjustment to a single contract chain. Both exist because naively concatenating contract prices manufactures a return at every roll.

## When to Use

Use it whenever a backtest spans a contract roll, or whenever the traded object is a basket whose weights change over time. Suspiciously smooth PnL around roll dates is the symptom of skipping it. Costs and financing rates must come from the same clock as the price data, and the contract calendar assumptions are worth verifying against the exchange rather than inferring from the data. This module is Rust-only — no Python bindings are exposed.

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

## Risk Notes and Caveats

- Verify contract calendar assumptions.
- Costs and rates should come from the same clock as price data.
- This module is Rust-only — no Python bindings are currently exposed.

## Related Modules

- [`data-structures`](/modules/data-structures/)
- [`backtesting-engine`](/modules/backtesting-engine/)
- [`backtest-statistics`](/modules/backtest-statistics/)
- [`bet-sizing`](/modules/bet-sizing/)

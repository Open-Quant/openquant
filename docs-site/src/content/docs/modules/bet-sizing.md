---
title: "bet_sizing"
description: "Transforms model confidence and constraints into executable position sizes."
status: validated
last_validated: '2026-03-02'
audience:
  - quant-dev
  - platform-engineering
module: "bet_sizing"
risk_notes:
  - "Keep sizing logic coupled to latency and fill assumptions; limit price from dynamic sizing is a decision boundary, not a guaranteed fill."
  - "Use reserve sizing when overlapping books or strategy stacking can create hidden gross exposure."
  - "Calibrate step_size to real execution granularity (lots/contracts), not arbitrary decimals."
rust_api:
  - "bet_size_probability"
  - "bet_size_dynamic"
  - "bet_size_budget"
  - "bet_size_reserve"
  - "bet_size_reserve_full"
  - "get_target_pos"
  - "limit_price"
sidebar:
  badge: Module
---

## Subject

**Position Sizing and Trade Construction**

## Why This Module Exists

A model signal is not tradable until converted into bounded, discrete, and risk-aware position sizes.

## Mathematical Foundations

### From Classification Probability to Signed Bet

$$\begin{aligned}z_t&=\frac{p_t-1/K}{\sqrt{p_t(1-p_t)}}\\m_t&=\operatorname{side}_t\left(2\Phi(z_t)-1\right)\\\tilde m_t&=\operatorname{clip}_{[-1,1]}\!\left(\Delta\,\mathrm{round}\!\left(\frac{m_t}{\Delta}\right)\right)\end{aligned}$$

### Dynamic Position Target and Limit Price

$$\begin{aligned}w&=\frac{x^2(1-m^2)}{m^2}\quad (x=f-m_p)\\m(x)&=\frac{x}{\sqrt{w+x^2}}\\\text{target}&=\operatorname{maxPos}\cdot m(f-m_p)\\\text{limitPrice}&=\frac{1}{|q^*-q|}\sum_{j=q}^{q^*}\operatorname{invPrice}(j)\end{aligned}$$

### Budget and Reserve Concurrency Sizing

$$\begin{aligned}b_t^{budget}&=\frac{L_t}{\max_s L_s}-\frac{S_t}{\max_s S_s}\\c_t&=L_t-S_t\\b_t^{reserve}&=\frac{F(c_t)-F(0)}{1-F(0)}\;\mathbf 1_{c_t\ge0}+\frac{F(c_t)-F(0)}{F(0)}\;\mathbf 1_{c_t<0}\end{aligned}$$

## Usage Examples

### Rust

#### End-to-end: Probability Forecasts -> Discrete Executable Bet Sizes

```rust
use chrono::{Duration, NaiveDateTime};
use openquant::bet_sizing::bet_size_probability;

// 1) Build event stream: (start, end, class probability, trade side)
let t0 = NaiveDateTime::parse_from_str("2024-01-01 09:30:00", "%Y-%m-%d %H:%M:%S")?;
let events = vec![
    (t0, t0 + Duration::minutes(20), 0.56,  1.0),
    (t0 + Duration::minutes(5), t0 + Duration::minutes(35), 0.62,  1.0),
    (t0 + Duration::minutes(10), t0 + Duration::minutes(30), 0.48, -1.0),
    (t0 + Duration::minutes(15), t0 + Duration::minutes(45), 0.67,  1.0),
];

// 2) Convert probabilities -> signed signal -> discretized size (step=0.1)
let sizes = bet_size_probability(&events, 2, 0.1, true);

// 3) sizes are directly executable as timestamped target exposure in [-1, 1]
assert!(!sizes.is_empty());
```

#### End-to-end: Dynamic + Reserve Sizing for Execution and Inventory Control

```rust
use chrono::{Duration, NaiveDateTime};
use openquant::bet_sizing::{bet_size_dynamic, bet_size_reserve_full};

// Dynamic sizing inputs (position, max position, market price, forecast price)
let pos = vec![0.0, 1.0, 1.0, 2.0, 1.0];
let max_pos = vec![10.0; 5];
let market = vec![100.0, 100.1, 100.0, 100.2, 100.15];
let forecast = vec![100.3, 100.4, 100.2, 100.5, 100.45];

let dynamic = bet_size_dynamic(&pos, &max_pos, &market, &forecast);
// tuple: (bet_size, target_position, limit_price)

// Reserve sizing from overlapping long/short events
let t0 = NaiveDateTime::parse_from_str("2024-01-01 09:30:00", "%Y-%m-%d %H:%M:%S")?;
let t1 = vec![
  (t0, t0 + Duration::minutes(30)),
  (t0 + Duration::minutes(10), t0 + Duration::minutes(40)),
  (t0 + Duration::minutes(20), t0 + Duration::minutes(50)),
];
let side = vec![1.0, -1.0, 1.0];
let (reserve, fit) = bet_size_reserve_full(&t1, &side, 8, 1e-6, 200, true);

assert_eq!(dynamic.len(), 5);
assert!(fit.is_some());
assert!(!reserve.is_empty());
```

## API Reference

### Rust API

- `bet_size_probability`
- `bet_size_dynamic`
- `bet_size_budget`
- `bet_size_reserve`
- `bet_size_reserve_full`
- `get_target_pos`
- `limit_price`

## Implementation Notes

- Keep sizing logic coupled to latency and fill assumptions; limit price from dynamic sizing is a decision boundary, not a guaranteed fill.
- Use reserve sizing when overlapping books or strategy stacking can create hidden gross exposure.
- Calibrate step_size to real execution granularity (lots/contracts), not arbitrary decimals.

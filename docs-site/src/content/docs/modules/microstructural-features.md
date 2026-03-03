---
title: "microstructural_features"
description: "Price-impact, spread, entropy, and flow toxicity estimators."
status: validated
last_validated: '2026-03-02'
audience:
  - quant-dev
  - platform-engineering
module: "microstructural_features"
risk_notes:
  - "Microstructure signals are highly regime-dependent; normalize and standardize within venue/time bucket before cross-asset comparison."
  - "Use shared bar definitions between training and live pipelines, otherwise feature drift is structural."
  - "Entropy features are sensitive to encoding; freeze symbol maps in production."
rust_api:
  - "get_roll_measure"
  - "get_corwin_schultz_estimator"
  - "get_bar_based_kyle_lambda"
  - "get_vpin"
  - "MicrostructuralFeaturesGenerator"
sidebar:
  badge: Module
---

## Subject

**Market Microstructure, Dependence and Regime Detection**

## Why This Module Exists

Microstructure features capture liquidity and order-flow dynamics not visible in OHLC bars alone.

## Mathematical Foundations

### Kyle / Amihud / Hasbrouck Impact Families

$$\Delta p_t=\lambda_K q_t+\epsilon_t,\qquad r_t=\lambda_A\frac{1}{DV_t}+\epsilon_t,\qquad r_t=\lambda_H\frac{q_t}{\sqrt{DV_t}}+\epsilon_t$$

### Spread and Volatility Proxies

$$\text{Roll spread}\approx 2\sqrt{-\operatorname{cov}(\Delta p_t,\Delta p_{t-1})},\qquad\sigma_{CS}=f(H_t,L_t,H_{t-1},L_{t-1})$$

### Flow Toxicity and Entropy

$$\mathrm{VPIN}_t=\frac{1}{n}\sum_{i=t-n+1}^{t}\frac{|V_i^b-V_i^s|}{V_i},\qquad H=-\sum_j p_j\log p_j$$

## Usage Examples

### Rust

#### End-to-end: Build Core Liquidity Feature Panel

```rust
use openquant::microstructural_features::{
    get_roll_measure,
    get_corwin_schultz_estimator,
    get_bar_based_kyle_lambda,
    get_bar_based_amihud_lambda,
    get_vpin,
};

// 1) Inputs from bar construction
let close = vec![100.0, 100.2, 100.1, 100.3, 100.25, 100.4];
let high = vec![100.1, 100.25, 100.2, 100.35, 100.3, 100.45];
let low = vec![99.9, 100.0, 99.95, 100.1, 100.05, 100.2];
let volume = vec![1000.0, 1200.0, 900.0, 1100.0, 1300.0, 1250.0];
let dollar_volume: Vec<f64> = close.iter().zip(volume.iter()).map(|(p, v)| p * v).collect();
let buy_volume = vec![600.0, 700.0, 480.0, 650.0, 800.0, 760.0];

// 2) Liquidity and spread proxies
let roll = get_roll_measure(&close, 3);
let cs_spread = get_corwin_schultz_estimator(&high, &low, 3);
let kyle = get_bar_based_kyle_lambda(&close, &volume, 3);
let amihud = get_bar_based_amihud_lambda(&close, &dollar_volume, 3);
let vpin = get_vpin(&volume, &buy_volume, 3);

// 3) Feature panel is ready for regime model / execution model
assert_eq!(roll.len(), close.len());
assert_eq!(vpin.len(), close.len());
```

#### From Encoded Tick Signs to Entropy Features

```rust
use openquant::microstructural_features::{
    encode_tick_rule_array,
    get_shannon_entropy,
    get_lempel_ziv_entropy,
    get_plug_in_entropy,
};

let tick_rule = vec![1, 1, -1, -1, 1, -1, 1, 1, 1, -1];
let msg = encode_tick_rule_array(&tick_rule)?;

let h_shannon = get_shannon_entropy(&msg);
let h_lz = get_lempel_ziv_entropy(&msg);
let h_plugin = get_plug_in_entropy(&msg, 2);

assert!(h_shannon.is_finite());
assert!(h_lz.is_finite());
assert!(h_plugin.is_finite());
```

## API Reference

### Rust API

- `get_roll_measure`
- `get_corwin_schultz_estimator`
- `get_bar_based_kyle_lambda`
- `get_vpin`
- `MicrostructuralFeaturesGenerator`

## Implementation Notes

- Microstructure signals are highly regime-dependent; normalize and standardize within venue/time bucket before cross-asset comparison.
- Use shared bar definitions between training and live pipelines, otherwise feature drift is structural.
- Entropy features are sensitive to encoding; freeze symbol maps in production.

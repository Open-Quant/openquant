---
title: "microstructural_features"
description: "Price-impact, spread, entropy, and flow toxicity estimators."
status: generated
generated_from: src/data/moduleDocs.ts
last_generated: '2026-08-31'
banner:
  content: '<span class="doc-status doc-status--generated">Generated</span> Assembled automatically from <code>moduleDocs.ts</code>. No human has reviewed this page.'
audience:
  - quant-dev
  - platform-engineering
module: "microstructural_features"
api_surface: "both"
rust_api:
  - "get_roll_measure"
  - "get_corwin_schultz_estimator"
  - "get_bar_based_kyle_lambda"
  - "get_vpin"
  - "MicrostructuralFeaturesGenerator"
sidebar:
  badge: Module
---

## Concept Overview

Features computed from bar-level order flow rather than from price alone, in three families: effective-spread proxies (Roll, Corwin-Schultz), price-impact coefficients (Kyle's lambda, Amihud, Hasbrouck) and flow-toxicity or entropy measures (VPIN, plus Shannon, Lempel-Ziv and plug-in entropy over encoded tick signs). Together they estimate what OHLC bars omit: how expensive the instrument is to trade, and how likely it is that the counterparty knows something you do not.

## When to Use

Use them as features when the edge or its cost depends on liquidity — execution models, regime detection, and any signal that decays with trade size. VPIN in particular is an early-warning indicator for flow toxicity ahead of liquidity events. Normalise within venue and time bucket before comparing across assets, since these are strongly regime-dependent, and freeze the symbol encoding used for entropy features or the values will not be comparable between training and production.

## Mathematical Foundations

### Kyle / Amihud / Hasbrouck Impact Families

$$\Delta p_t=\lambda_K q_t+\epsilon_t,\qquad r_t=\lambda_A\frac{1}{DV_t}+\epsilon_t,\qquad r_t=\lambda_H\frac{q_t}{\sqrt{DV_t}}+\epsilon_t$$

### Spread and Volatility Proxies

$$\text{Roll spread}\approx 2\sqrt{-\operatorname{cov}(\Delta p_t,\Delta p_{t-1})},\qquad\sigma_{CS}=f(H_t,L_t,H_{t-1},L_{t-1})$$

### Flow Toxicity and Entropy

$$\mathrm{VPIN}_t=\frac{1}{V_t}\cdot\frac{1}{n}\sum_{i=t-n+1}^{t}\left|V_i^{B}-V_i^{S}\right|,\qquad H=-\sum_j p_j\log p_j$$

where $V_i^{B}$ and $V_i^{S}$ are buy- and sell-initiated volume in bar $i$ (`get_bvc_buy_volume` will estimate the split when it is not observed), $V_t$ the current bar's total volume, and $n$ the rolling `window`. The normaliser sits *outside* the sum because bars are not equal-volume: `get_vpin` averages the imbalance over the window and then scales by the latest bar. The equal-volume-bucket form used by [`streaming-hpc`](/modules/streaming-hpc/) divides each term by the same constant bucket size instead; the two agree when bars carry equal volume. $H$ is the entropy of the tick-sign message, with $p_j$ the empirical frequency of symbol $j$.

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

### Python API

- `microstructural.get_roll_measure`
- `microstructural.get_roll_impact`
- `microstructural.get_corwin_schultz_estimator`
- `microstructural.get_bekker_parkinson_vol`
- `microstructural.get_bar_based_kyle_lambda`
- `microstructural.get_bar_based_amihud_lambda`
- `microstructural.get_bar_based_hasbrouck_lambda`
- `microstructural.get_trades_based_kyle_lambda`
- `microstructural.get_trades_based_amihud_lambda`
- `microstructural.get_trades_based_hasbrouck_lambda`
- `microstructural.vwap`
- `microstructural.get_avg_tick_size`
- `microstructural.get_vpin`
- `microstructural.get_bvc_buy_volume`
- `microstructural.encode_tick_rule_array`
- `microstructural.quantile_mapping`
- `microstructural.sigma_mapping`
- `microstructural.encode_array`
- `microstructural.get_shannon_entropy`
- `microstructural.get_lempel_ziv_entropy`
- `microstructural.get_plug_in_entropy`
- `microstructural.get_konto_entropy`

### Rust API

- `get_roll_measure`
- `get_corwin_schultz_estimator`
- `get_bar_based_kyle_lambda`
- `get_vpin`
- `MicrostructuralFeaturesGenerator`

## Risk Notes and Caveats

- Microstructure signals are highly regime-dependent; normalize and standardize within venue/time bucket before cross-asset comparison.
- Use shared bar definitions between training and live pipelines, otherwise feature drift is structural.
- Entropy features are sensitive to encoding; freeze symbol maps in production.

## Related Modules

- [`data-structures`](/modules/data-structures/)
- [`streaming-hpc`](/modules/streaming-hpc/)
- [`structural-breaks`](/modules/structural-breaks/)
- [`filters`](/modules/filters/)
- [`codependence`](/modules/codependence/)

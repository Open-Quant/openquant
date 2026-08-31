---
title: "util::volatility"
description: "Volatility estimators used across labeling and risk workflows."
status: generated
generated_from: src/data/moduleDocs.ts
last_generated: '2026-08-31'
audience:
  - quant-dev
  - platform-engineering
module: "util::volatility"
api_surface: "both"
rust_api:
  - "get_daily_vol"
  - "get_parkinson_vol"
  - "get_garman_class_vol"
  - "get_yang_zhang_vol"
sidebar:
  badge: Module
---

## Concept Overview

Four volatility estimators with different data requirements and different blind spots. `get_daily_vol` is a close-to-close EWMA over a timestamped series — the estimator AFML uses to scale triple-barrier widths. Parkinson uses the high-low range and extracts far more information per observation, but ignores overnight gaps and assumes no drift. Garman-Klass adds the open and close. Yang-Zhang combines an overnight, an open-to-close and a Rogers-Satchell term under a variance-minimising weight, and is the only one of the four that handles both opening gaps and intraday drift.

## When to Use

Use `get_daily_vol` whenever volatility is a scaling target for barriers or position sizes, and match its lookback to the event horizon — a 100-bar volatility scaling a 3-bar barrier is measuring the wrong thing. Use the range-based estimators when you have OHLC and want more precision from the same number of bars, preferring Yang-Zhang for instruments that gap. All range estimators degrade when quoted spreads are wide, because the recorded high and low then reflect microstructure noise rather than price.

## Mathematical Foundations

### Parkinson

$$\sigma_P^2=\frac{1}{4\ln 2}\cdot\frac{1}{n}\sum_{t}\left(\ln\frac{H_t}{L_t}\right)^2$$

where $H_t,L_t$ are the bar high and low and $n$ the `window` length. It uses the range rather than the close, so it is far more efficient than close-to-close on the same sample — but it ignores overnight gaps and assumes no drift.

### Yang-Zhang

$$\sigma_{YZ}^2=\sigma_o^2+k\,\sigma_c^2+(1-k)\,\sigma_{rs}^2,\qquad k=\frac{0.34}{1.34+\frac{n+1}{n-1}}$$

where $\sigma_o^2$ is the overnight (close-to-open) variance, $\sigma_c^2$ the open-to-close variance, and $\sigma_{rs}^2$ the Rogers-Satchell estimator; $n$ is the `window` length. $k$ is not a free parameter — it is the weight that minimises the estimator's variance, which is what makes Yang-Zhang the only one of these four that handles both overnight gaps and intraday drift. For a 20-bar window $k\approx0.14$, so the overnight and Rogers-Satchell terms carry most of the estimate.

## Usage Examples

### Rust

#### Compute daily and range-based volatility

```rust
use chrono::{Duration, NaiveDateTime};
use openquant::util::volatility::{get_daily_vol, get_parkinson_vol};

let t0 = NaiveDateTime::parse_from_str("2024-01-02 00:00:00", "%Y-%m-%d %H:%M:%S")?;
let close: Vec<(NaiveDateTime, f64)> = (0..300)
    .map(|i| (t0 + Duration::days(i), 100.0 + (i as f64 * 0.07).sin() * 2.0))
    .collect();
let high: Vec<f64> = close.iter().map(|(_, p)| p + 0.4).collect();
let low: Vec<f64> = close.iter().map(|(_, p)| p - 0.4).collect();

// Close-to-close EWMA vol on a timestamped series; `lookback` is the EWMA span.
let daily = get_daily_vol(&close, 100);
// Parkinson uses the high/low range, so it needs no timestamps — `window` bars.
let parkinson = get_parkinson_vol(&high, &low, 20);

println!("daily vol tail = {:?}", daily.last());
println!("parkinson vol tail = {:?}", parkinson.last());
```

## API Reference

### Python API

- `volatility.get_daily_vol`
- `volatility.get_parkinson_vol`
- `volatility.get_garman_class_vol`
- `volatility.get_yang_zhang_vol`

### Rust API

- `get_daily_vol`
- `get_parkinson_vol`
- `get_garman_class_vol`
- `get_yang_zhang_vol`

## Risk Notes and Caveats

- Choose estimator based on available fields and microstructure noise.
- Daily-vol lookback should be matched to event horizon.

## Related Modules

- [`labeling`](/modules/labeling/)
- [`filters`](/modules/filters/)
- [`util-fast-ewma`](/modules/util-fast-ewma/)
- [`bet-sizing`](/modules/bet-sizing/)
- [`microstructural-features`](/modules/microstructural-features/)

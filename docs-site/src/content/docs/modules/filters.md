---
title: "filters"
description: "CUSUM and z-score event filters for event-driven sampling."
status: validated
last_validated: '2026-03-02'
audience:
  - quant-dev
  - platform-engineering
module: "filters"
api_surface: "both"
afml_chapters:
  - 2
risk_notes:
  - "Calibrate thresholds to target event frequency, not just sensitivity."
  - "Use identical filtering in train and live pipelines."
  - "Rust API supports dynamic (per-bar) thresholds via Threshold::Dynamic; Python bindings accept only a scalar threshold."
  - "Rust _checked variants return Result<..., FilterError> for input validation; Python raises exceptions."
rust_api:
  - "cusum_filter_indices"
  - "cusum_filter_timestamps"
  - "cusum_filter_indices_checked"
  - "cusum_filter_timestamps_checked"
  - "z_score_filter_indices"
  - "z_score_filter_timestamps"
  - "z_score_filter_timestamps_checked"
  - "Threshold"
  - "FilterError"
sidebar:
  badge: Module
---

## Concept Overview

Instead of sampling at fixed intervals, AFML Chapter 2 uses structural event filters to detect when something meaningful happens in the price process. This produces training examples that correspond to real market inflection points rather than arbitrary calendar dates.

The **CUSUM filter** tracks a cumulative sum of returns (or price changes). It resets to zero when the cumulative deviation exceeds a threshold h, and the reset point becomes an event. This captures points where the price has moved "enough" since the last event. The filter is directional: it tracks both positive and negative cumulative deviations separately.

The **z-score filter** standardizes the current value against a rolling mean and standard deviation, firing when the z-score exceeds a threshold. This is useful for mean-reverting signals where you want events when the price deviates significantly from its recent average.

Both filters replace the naive approach of labeling every bar, which creates highly correlated and redundant training examples.

## When to Use

Apply event filters immediately after bar construction and before labeling. They bridge raw bars to the labeling module: bars go in, event timestamps come out.

**Prerequisites**: A price series (close prices from bars), and optionally timestamps.

**Alternatives**: Fixed-interval sampling (simpler but creates redundant events), or custom event logic for strategy-specific triggers.

## Mathematical Foundations

### CUSUM

$$S_t=\max(0, S_{t-1}+r_t),\; trigger\;if\;|S_t|>h$$

### Z-score

$$z_t=\frac{x_t-\mu_t}{\sigma_t}$$

## Key Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `close` | `list[float]` | Input price series (close prices) | — |
| `threshold` | `float` | CUSUM trigger level; controls event frequency (Python: scalar only) | — |
| `threshold` | `Threshold` | CUSUM trigger: Threshold::Scalar(f64) or Threshold::Dynamic(Vec<f64>) (Rust) | — |
| `mean_window` | `int` | Rolling mean lookback for z-score filter | — |
| `std_window` | `int` | Rolling std lookback for z-score filter | — |
| `timestamps` | `list[str]` | Optional timestamps; use _timestamps variants to get event times instead of indices | — |

## Usage Examples

### Python

#### CUSUM and z-score event detection

```python
import openquant

close = [100.0, 100.1, 99.9, 100.2, 100.05, 100.3, 99.7, 100.1]
timestamps = [
    "2024-01-02T09:30:00", "2024-01-02T09:31:00",
    "2024-01-02T09:32:00", "2024-01-02T09:33:00",
    "2024-01-02T09:34:00", "2024-01-02T09:35:00",
    "2024-01-02T09:36:00", "2024-01-02T09:37:00",
]

# CUSUM filter: fires when cumulative deviation exceeds threshold
event_indices = openquant.filters.cusum_filter_indices(close, 0.02)

# With timestamps: returns event timestamps directly
event_ts = openquant.filters.cusum_filter_timestamps(close, timestamps, 0.02)

# Z-score filter: fires when z-score exceeds threshold
z_indices = openquant.filters.z_score_filter_indices(close, mean_window=20, std_window=20, threshold=2.0)
z_ts = openquant.filters.z_score_filter_timestamps(close, timestamps, mean_window=20, std_window=20, threshold=2.0)
```

### Rust

#### CUSUM with static and dynamic thresholds

```rust
use openquant::filters::{cusum_filter_indices, cusum_filter_indices_checked, Threshold};

let close = vec![100.0, 100.1, 99.9, 100.2];

// Static threshold
let idx = cusum_filter_indices(&close, Threshold::Scalar(0.02));

// Dynamic threshold (e.g. volatility-scaled per bar)
let dynamic_h = vec![0.02, 0.025, 0.018, 0.022];
let idx = cusum_filter_indices_checked(&close, Threshold::Dynamic(dynamic_h)).unwrap();
```

## Common Pitfalls

- Setting the CUSUM threshold too tight in volatile regimes — you get too many events and labels become noisy. Scale h by recent volatility.
- Using different thresholds in training vs live inference — the event distribution shifts and the model sees a different regime.
- Applying CUSUM to non-stationary raw prices instead of returns or log-returns — the filter becomes meaningless as the price drifts.
- Python bindings only support scalar thresholds — use the Rust API directly if you need dynamic (per-bar) thresholds.

## API Reference

### Python API

- `filters.cusum_filter_indices`
- `filters.cusum_filter_timestamps`
- `filters.z_score_filter_indices`
- `filters.z_score_filter_timestamps`

### Rust API

- `cusum_filter_indices`
- `cusum_filter_timestamps`
- `cusum_filter_indices_checked`
- `cusum_filter_timestamps_checked`
- `z_score_filter_indices`
- `z_score_filter_timestamps`
- `z_score_filter_timestamps_checked`
- `Threshold`
- `FilterError`

## Implementation Notes

- Calibrate thresholds to target event frequency, not just sensitivity.
- Use identical filtering in train and live pipelines.
- Rust API supports dynamic (per-bar) thresholds via Threshold::Dynamic; Python bindings accept only a scalar threshold.
- Rust _checked variants return Result<..., FilterError> for input validation; Python raises exceptions.

## Related Modules

- [`data-structures`](/modules/data-structures/)
- [`labeling`](/modules/labeling/)
- [`sample-weights`](/modules/sample-weights/)

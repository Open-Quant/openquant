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
rust_api:
  - "cusum_filter_indices"
  - "cusum_filter_timestamps"
  - "z_score_filter_indices"
  - "Threshold"
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
| `threshold (h)` | `f64 | Threshold` | CUSUM trigger level; controls event frequency. Scalar for constant threshold, or dynamic for volatility-scaled | — |

## Usage Examples

### Python

#### Detect structural price events with CUSUM

```python
from openquant._core import filters

close = [100.0, 100.1, 99.9, 100.2, 100.05, 100.3, 99.7, 100.1]
timestamps = ["2024-01-02T09:30:00", "2024-01-02T09:31:00", ...]

# CUSUM filter: fires when cumulative deviation exceeds threshold
event_indices = filters.cusum_filter_indices(close, 0.02)
# Returns indices where the filter triggered

# With timestamps: returns event timestamps directly
event_ts = filters.cusum_filter_timestamps(close, timestamps, 0.02)
```

### Rust

#### Run CUSUM over closes

```rust
use openquant::filters::{cusum_filter_indices, Threshold};

let close = vec![100.0, 100.1, 99.9, 100.2];
let idx = cusum_filter_indices(&close, Threshold::Scalar(0.02));
```

## Common Pitfalls

- Setting the CUSUM threshold too tight in volatile regimes — you get too many events and labels become noisy. Scale h by recent volatility.
- Using different thresholds in training vs live inference — the event distribution shifts and the model sees a different regime.
- Applying CUSUM to non-stationary raw prices instead of returns or log-returns — the filter becomes meaningless as the price drifts.

## API Reference

### Python API

- `filters.cusum_filter_indices`
- `filters.cusum_filter_timestamps`
- `filters.z_score_filter_indices`

### Rust API

- `cusum_filter_indices`
- `cusum_filter_timestamps`
- `z_score_filter_indices`
- `Threshold`

## Implementation Notes

- Calibrate thresholds to target event frequency, not just sensitivity.
- Use identical filtering in train and live pipelines.

## Related Modules

- [`data-structures`](/modules/data-structures/)
- [`labeling`](/modules/labeling/)
- [`sample-weights`](/modules/sample-weights/)

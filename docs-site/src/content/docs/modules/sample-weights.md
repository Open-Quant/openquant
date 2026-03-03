---
title: "sample_weights"
description: "Sample weighting utilities for overlapping event structure."
status: validated
last_validated: '2026-03-02'
audience:
  - quant-dev
  - platform-engineering
module: "sample_weights"
api_surface: "both"
afml_chapters:
  - 4
risk_notes:
  - "Pair with sequential bootstrap for robust label sampling."
  - "Time-decay controls recency bias explicitly."
rust_api:
  - "get_weights_by_return"
  - "get_weights_by_time_decay"
sidebar:
  badge: Module
---

## Concept Overview

In AFML's event-driven framework (Chapter 4), labels are derived from overlapping price paths. When two events overlap in time, their labels share information — the price observations that determine event A's outcome also influence event B's outcome. Treating these labels as independent samples inflates effective sample size and biases model training.

**Uniqueness-based weighting** addresses this by computing how unique each sample is at each time step. If a bar contributes to 3 concurrent events, each event gets 1/3 credit for that bar. The total weight of a sample is the sum of its per-bar uniqueness scores. Samples that overlap with many others get down-weighted; isolated samples get full weight.

**Return-attribution weighting** weights samples by their absolute return, giving more training influence to economically significant events.

**Time-decay weighting** applies a power-law decay so recent observations contribute more than older ones, useful when the data-generating process evolves over time.

These weights should be passed as `sample_weight` to your classifier or loss function.

## When to Use

Apply sample weights after labeling and before model training. They correct for the non-IID structure caused by overlapping triple-barrier labels.

**Prerequisites**: Labeled events from the labeling module, with event start/end times.

**Alternatives**: Equal weights (ignores overlap, biases toward dense clusters), or sequential bootstrap (sampling-based approach instead of weighting).

## Mathematical Foundations

### Uniqueness Weight

$$w_i=\sum_t\frac{I_{t,i}}{\sum_j I_{t,j}}$$

### Time Decay

$$w_i=(\frac{i}{T})^\delta$$

## Key Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `delta` | `f64` | Time-decay exponent; 0 = uniform, 1 = linear decay, >1 = aggressive recency bias | 1.0 |

## Usage Examples

### Python

#### Compute sample weights for overlapping labels

```python
from openquant._core import sample_weights

# Returns from labeled events (used for return-attribution weighting)
returns = [0.01, -0.005, 0.007, -0.002, 0.003, 0.01, -0.008]

# Weight by absolute return (higher-impact events get more weight)
w_return = sample_weights.get_weights_by_return(returns)

# Weight by time decay (more recent events weighted higher, delta=0.5)
w_decay = sample_weights.get_weights_by_time_decay(returns, 0.5)

# Use these weights in model training:
# model.fit(X, y, sample_weight=w_return)
```

### Rust

#### Compute event weights

```rust
use openquant::sample_weights::get_weights_by_time_decay;

let w = get_weights_by_time_decay(&returns, 0.5);
```

## Common Pitfalls

- Training without any overlap correction — highly overlapping labels effectively duplicate data and overfit the dense-event regime.
- Using uniqueness weights without the indicator matrix from the sampling module — the weights require knowledge of which bars each event spans.
- Combining time-decay and uniqueness weights incorrectly — multiply them element-wise, don't add.

## API Reference

### Python API

- `sample_weights.get_weights_by_return`
- `sample_weights.get_weights_by_time_decay`

### Rust API

- `get_weights_by_return`
- `get_weights_by_time_decay`

## Implementation Notes

- Pair with sequential bootstrap for robust label sampling.
- Time-decay controls recency bias explicitly.

## Related Modules

- [`labeling`](/modules/labeling/)
- [`sampling`](/modules/sampling/)
- [`sb-bagging`](/modules/sb-bagging/)

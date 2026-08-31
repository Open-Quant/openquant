---
title: "ef3m"
description: "Moment-based mixture fitting utilities for two-normal components."
status: generated
generated_from: src/data/moduleDocs.ts
last_generated: '2026-08-30'
banner:
  content: '<span class="doc-status doc-status--generated">Generated</span> Assembled automatically from <code>moduleDocs.ts</code>. No human has reviewed this page.'
audience:
  - quant-dev
  - platform-engineering
module: "ef3m"
api_surface: "both"
rust_api:
  - "M2N"
  - "centered_moment"
  - "raw_moment"
  - "most_likely_parameters"
sidebar:
  badge: Module
---

## Concept Overview

Exact Fit of the first 3, 4 or 5 Moments: fits a mixture of two Gaussians by matching sample moments instead of by maximum likelihood. `M2N` takes the observed moments and searches over the second mean and the mixing probability, solving the remaining parameters analytically at each candidate (`iter_4` and `iter_5` for the four- and five-moment variants); `most_likely_parameters` then picks the modal solution across that search. It is fast and derivative-free, which is what makes it usable as an initialiser.

## When to Use

Use it when a return or bet-outcome distribution is visibly bimodal — two regimes, or a mixture of trades that ran and trades that were stopped — and you want the components without paying for EM. It is the standard way to obtain the mixture parameters `bet_size_reserve` needs. Because it works from higher moments it is sensitive to tail estimation noise, so on small samples treat its output as an initialisation for a heavier optimiser rather than a final answer.

## Mathematical Foundations

### Raw Moment

$$m_k=E[X^k]$$

### Mixture Mean

$$\mu=p\mu_1+(1-p)\mu_2$$

## Usage Examples

### Rust

#### Estimate moments

```rust
use openquant::ef3m::centered_moment;

let moments = vec![0.0, 1.0, 0.1, 3.0];
let m3 = centered_moment(&moments, 3);
```

## API Reference

### Python API

- `ef3m.centered_moment`
- `ef3m.raw_moment`
- `ef3m.most_likely_parameters`
- `ef3m.fit_m2n`

### Rust API

- `M2N`
- `centered_moment`
- `raw_moment`
- `most_likely_parameters`

## Risk Notes and Caveats

- Use as initialization for more expensive optimizers.
- Sensitive to higher-moment estimation noise.

## Related Modules

- [`bet-sizing`](/modules/bet-sizing/)
- [`backtest-statistics`](/modules/backtest-statistics/)
- [`strategy-risk`](/modules/strategy-risk/)

---
title: "util::fast_ewma"
description: "Fast EWMA primitive shared across feature and volatility routines."
status: authored
last_authored: '2026-10-10'
audience:
  - quant-dev
  - platform-engineering
module: "util::fast_ewma"
api_surface: "both"
rust_api:
  - "ewma"
python_api:
  - "fast_ewma.ewma"
sidebar:
  badge: Module
citation:
  - "OpenQuant crates/openquant/src/util/fast_ewma.rs: adjusted EWMA implementation and doctests."
  - "OpenQuant python/openquant/_core/fast_ewma.pyi: Python binding contract."
---

## Adjusted exponential smoothing

`ewma` turns an oldest-first numeric sequence into an equally long sequence of
weighted means. Use it to smooth a feature or threshold when recent observations
should count more than older ones. `window` is a **span**, not a rolling cutoff:
with span 3, each older observation receives half the weight of the next one.
No timestamps are accepted, so decay is per observation, even across time gaps.

The implementation in [fast_ewma.rs](https://github.com/Open-Quant/openquant/blob/main/crates/openquant/src/util/fast_ewma.rs)
normalizes by the weights available so far. For $\alpha=2/(w+1)$ and
$q=1-\alpha$,

$$
m_t=\frac{\sum_{j=0}^{t}q^{t-j}x_j}{\sum_{j=0}^{t}q^{t-j}}.
$$

Equivalently, update a numerator $N_t=qN_{t-1}+x_t$ and denominator
$D_t=qD_{t-1}+1$, starting with $N_0=x_0,D_0=1$, and return $N_t/D_t$.
This is adjusted EWMA; applying $m_t=\alpha x_t+(1-\alpha)m_{t-1}$ directly
would give different early values.

For `[3, 6, 12]` at span 3, the final numerator is
$0.25\times3+0.5\times6+12=15.75$ and the denominator is $1.75$, giving 9.
The intermediate value is $(0.5\times3+6)/1.5=5$.

<figure>
<img src="/figures/util-ewma-weights-light.svg" alt="At span 3, the third adjusted mean uses weights 0.25, 0.5 and 1, normalized by their sum 1.75." class="dark:sl-hidden" />
<img src="/figures/util-ewma-weights-dark.svg" alt="At span 3, the third adjusted mean uses weights 0.25, 0.5 and 1, normalized by their sum 1.75." class="light:sl-hidden" />
<figcaption>Analytic weights for the three-value example; regenerate with <code>python3 docs-site/scripts/figures/utility_data_figures.py</code>.</figcaption>
</figure>

## Runnable Python example

```python
import math
from openquant import fast_ewma

values = [3.0, 6.0, 12.0]
smoothed = fast_ewma.ewma(values, window=3)
assert smoothed == [3.0, 5.0, 9.0]
assert fast_ewma.ewma(values, window=1) == values
assert fast_ewma.ewma([], window=3) == []
missing = fast_ewma.ewma([3.0, float("nan"), 12.0], window=3)
print(smoothed)
print([math.isnan(x) for x in missing])
try:
    fast_ewma.ewma(values, window=0)
except ValueError:
    print("zero span rejected")
```

```text
[3.0, 5.0, 9.0]
[False, True, True]
zero span rejected
```

The numerical assertions come from the weighted fractions above, not a fitted
reference series. The missing-value example illustrates a deliberate limit:
a NaN contaminates every later output; this is not pandas' default skip-missing
behavior. Clean or explicitly segment the input before smoothing.

## Rust equivalent

```rust
use openquant::util::fast_ewma::ewma;

let smoothed = ewma(&[3.0, 6.0, 12.0], 3)?;
assert_eq!(smoothed, vec![3.0, 5.0, 9.0]);
assert!(ewma(&[], 3)?.is_empty());
assert!(ewma(&[3.0], 0).is_err());
```

## Input and output contract

| Input or case | Behavior |
| --- | --- |
| `arr` in Python / `arr_in` in Rust | Numeric observations, oldest first |
| Positive `window` | Span; larger spans give slower decay |
| Empty sequence | Empty output, provided the span is positive |
| Span 1 | Returns finite inputs unchanged |
| Span 0 | Python `ValueError`; Rust `InputError::OutOfRange` |
| Nonfinite values | Not validated; NaNs propagate |

Smoothing does not turn a nonstationary feature into a stationary one, and a
span is not a forecast horizon. Choose the observation frequency and span as
part of the feature definition, then evaluate them in the downstream workflow.

## Sources and related modules

- [Rust implementation and doctests](https://github.com/Open-Quant/openquant/blob/main/crates/openquant/src/util/fast_ewma.rs): normalized weights, span validation and empty-input behavior.
- [Python binding contract](https://github.com/Open-Quant/openquant/blob/main/python/openquant/_core/fast_ewma.pyi): argument names, return shape and NaN behavior.
- [Volatility](/modules/util-volatility/) estimates dispersion rather than a mean; [filters](/modules/filters/) sample events from prices.

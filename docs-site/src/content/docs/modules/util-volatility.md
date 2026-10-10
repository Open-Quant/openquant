---
title: "util::volatility"
description: "Volatility estimators used across labeling and risk workflows."
status: authored
last_authored: '2026-10-10'
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
python_api:
  - "volatility.get_daily_vol"
  - "volatility.get_parkinson_vol"
  - "volatility.get_garman_class_vol"
  - "volatility.get_yang_zhang_vol"
sidebar:
  badge: Module
citation:
  - "López de Prado, M. (2018). Advances in Financial Machine Learning. Wiley. §3.3, Snippet 3.1."
  - "OpenQuant crates/openquant/src/util/volatility.rs and crates/openquant/tests/volatility_features.rs: estimator formulas and reference tests."
---

## Four estimators, two window conventions

The volatility utilities return standard deviations of returns, without
annualization. Supply positive price levels ordered oldest first. Range
estimators consume parallel OHLC arrays; daily volatility consumes timestamped
closes and returns timestamped estimates. The
[implementation](https://github.com/Open-Quant/openquant/blob/main/crates/openquant/src/util/volatility.rs)
and [reference tests](https://github.com/Open-Quant/openquant/blob/main/crates/openquant/tests/volatility_features.rs)
define the conventions below.

| Function | Inputs | Smoothing | Output and warm-up |
| --- | --- | --- | --- |
| `get_daily_vol` | Timestamps and close | Adjusted, unbiased EWM standard deviation; `lookback` is a span | Omits bars without an eligible earlier close; first returned estimate is NaN |
| `get_parkinson_vol` | High, low | Rolling `window` bars | Input length; first `window - 1` values NaN |
| `get_garman_class_vol` | Open, high, low, close | Rolling `window` bars | Input length; first `window - 1` values NaN |
| `get_yang_zhang_vol` | Open, high, low, close | Rolling `window` bars plus previous close | Input length; first `window` values NaN |

Use daily volatility for a timestamped target such as a labeling barrier. Use
range estimates to describe dispersion within OHLC bars. Parkinson and
Garman-Klass omit overnight gaps; Yang-Zhang includes a previous-close-to-open
term. The public name `get_garman_class_vol` retains the historical spelling,
although the estimator is Garman-Klass.

<figure>
<img src="/figures/util-volatility-warmup-light.svg" alt="For a window of three, Parkinson and Garman-Klass first return a finite value at index two; Yang-Zhang at index three." class="dark:sl-hidden" />
<img src="/figures/util-volatility-warmup-dark.svg" alt="For a window of three, Parkinson and Garman-Klass first return a finite value at index two; Yang-Zhang at index three." class="light:sl-hidden" />
<figcaption>Warm-up positions for valid synthetic OHLC inputs; the extra Yang-Zhang bar supplies a previous close.</figcaption>
</figure>

## Daily returns and timestamp alignment

AFML §3.3, Snippet 3.1 selects the last close **strictly before** $t-1$ day:
$r_t=C_t/C_j-1$, where $t_j<t-1\text{ day}$. An exact timestamp match one day
ago is excluded. Consequently, regularly spaced daily observations produce
**two-day returns** after omitting the first two rows. This is the implemented
search convention, not an ordinary adjacent daily percentage change.

With $q=1-2/(\text{lookback}+1)$, returns receive weights $w_i=q^{t-i}$.
The weighted mean is $\bar r=\sum w_i r_i/\sum w_i$ and the unbiased variance is

$$
 s^2=\frac{\sum_i w_i(r_i-\bar r)^2}
 {\sum_i w_i-\sum_i w_i^2/\sum_i w_i}.
$$

One observation cannot estimate this variance. Span 1 also leaves only one
positive weight, so every returned estimate is NaN. Span 0 or fewer than two
input bars returns an empty sequence.

```python
import math
from datetime import datetime, timedelta
from openquant import volatility

stamps = [(datetime(2024, 1, 1) + timedelta(days=i)).strftime("%Y-%m-%d %H:%M:%S")
          for i in range(5)]
close = [100.0, 100.0, 110.0, 120.0, 121.0]
result = volatility.get_daily_vol(stamps, close, lookback=3)
# Eligible returns are 110/100-1, 120/100-1, and 121/110-1.
assert [stamp for stamp, _ in result] == stamps[2:]
assert math.isnan(result[0][1])
assert math.isclose(result[1][1], math.sqrt(0.005), abs_tol=1e-12)
for stamp, value in result:
    print(stamp, f"{value:.6f}")
```

```text
2024-01-03 00:00:00 nan
2024-01-04 00:00:00 0.070711
2024-01-05 00:00:00 0.059761
```

Always join the returned timestamps back to the input. Positional assignment
would shift the target because the output omits initial rows.

## Range formulas

For a trailing window of $n$ bars, the functions return the square roots of:

$$
\sigma_P^2=\frac{1}{4\ln 2}\operatorname{mean}\left[\ln(H/L)^2\right],
\qquad
\sigma_{GK}^2=\operatorname{mean}\left[\tfrac12\ln(H/L)^2-(2\ln2-1)\ln(C/O)^2\right].
$$

Yang-Zhang uses $o_i=\ln(O_i/C_{i-1})$, $c_i=\ln(C_i/O_i)$, and
$RS_i=\ln(H_i/C_i)\ln(H_i/O_i)+\ln(L_i/C_i)\ln(L_i/O_i)$:

$$
\sigma_{YZ}^2=\operatorname{Var}_{n-1}(o)+k\operatorname{Var}_{n-1}(c)
 +(1-k)\operatorname{mean}(RS),\qquad
k=\frac{0.34}{1.34+(n+1)/(n-1)}.
$$

Both variance terms subtract their window means and use divisor $n-1$; the
Rogers-Satchell term uses divisor $n$. These are the paper's conventions tested
in `test_yang_zhang_matches_paper`, rather than the older mlfinlab expression
using $\ln(C_i/O_{i-1})$ and uncentered sums.

```python
import math
from openquant import volatility

# Synthetic identical bars: no overnight or open-to-close movement.
open_ = close = [100.0] * 5
high, low = [101.0] * 5, [99.0] * 5
p = volatility.get_parkinson_vol(high, low, window=3)
g = volatility.get_garman_class_vol(open_, high, low, close, window=3)
y = volatility.get_yang_zhang_vol(open_, high, low, close, window=3)
expected_p = math.log(101 / 99) / math.sqrt(4 * math.log(2))
assert math.isclose(p[-1], expected_p, abs_tol=1e-12)
for name, values in [("Parkinson", p), ("Garman-Klass", g), ("Yang-Zhang", y)]:
    print(name, "warm-up", sum(math.isnan(v) for v in values), "tail", f"{values[-1]:.6f}")
```

```text
Parkinson warm-up 2 tail 0.012012
Garman-Klass warm-up 2 tail 0.014143
Yang-Zhang warm-up 3 tail 0.013404
```

The differences reflect the formulas, not evidence that one estimator forecasts
better. All bars here are synthetic and contain no market information.

## Rust range example

```rust
use openquant::util::volatility::get_parkinson_vol;

let values = get_parkinson_vol(&[101.0; 5], &[99.0; 5], 3)?;
assert!(values[..2].iter().all(|v| v.is_nan()));
let expected = (101.0_f64 / 99.0).ln() / (4.0 * 2.0_f64.ln()).sqrt();
assert!((values[4] - expected).abs() < 1e-12);
```

## Failure modes and interpretation

- Parallel array lengths must match; mismatches raise Python `ValueError` or Rust `InputError`. Python daily-volatility timestamps must parse as `YYYY-MM-DD HH:MM:SS` (fractional seconds accepted).
- Supply increasing timestamps yourself. Daily volatility does not sort or validate price positivity. The range estimators check lengths, not OHLC consistency; nonpositive or inconsistent prices can produce nonfinite results.
- Range windows containing NaNs remain NaN until the affected observations leave the required window. A close used by Yang-Zhang also affects the next bar's overnight return.
- Parkinson and Garman-Klass return all NaNs for window 0; Yang-Zhang does so for windows below 2. Windows longer than the history never finish warm-up.
- Outputs are not percentages or annualized values. Calendar gaps also mean the daily estimator's actual return horizon can exceed one day. Establish the sampling convention before scaling values or comparing estimators.

## Sources and related modules

- López de Prado (2018), *Advances in Financial Machine Learning*, §3.3, Snippet 3.1: daily volatility target.
- [Rust implementation](https://github.com/Open-Quant/openquant/blob/main/crates/openquant/src/util/volatility.rs) and [Python contract](https://github.com/Open-Quant/openquant/blob/main/python/openquant/_core/volatility.pyi): formulas, validation and output alignment.
- [Volatility reference tests](https://github.com/Open-Quant/openquant/blob/main/crates/openquant/tests/volatility_features.rs): range estimator references, centered Yang-Zhang variance and pure-drift case.
- [Labeling](/modules/labeling/) consumes volatility targets; [EWMA](/modules/util-fast-ewma/) explains adjusted means; [data](/modules/data/) prepares OHLCV inputs.

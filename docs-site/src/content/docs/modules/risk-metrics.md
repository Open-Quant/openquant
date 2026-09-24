---
title: "risk_metrics"
description: "Historical value at risk, expected shortfall, conditional drawdown at risk and portfolio variance."
status: authored
last_authored: '2026-09-24'
audience:
  - quant-dev
  - platform-engineering
module: "risk_metrics"
api_surface: "both"
citation:
  - "Artzner, P., Delbaen, F., Eber, J.-M. and Heath, D. (1999). Coherent measures of risk. Mathematical Finance 9(3), 203–228."
  - "Rockafellar, R. T. and Uryasev, S. (2000). Optimization of conditional value-at-risk. Journal of Risk 2(3), 21–41."
  - "Chekhlov, A., Uryasev, S. and Zabarankin, M. (2005). Drawdown measure in portfolio optimization. International Journal of Theoretical and Applied Finance 8(1), 13–58."
rust_api:
  - "RiskMetrics"
  - "RiskMetricsError"
python_api:
  - "risk.calculate_value_at_risk"
  - "risk.calculate_expected_shortfall"
  - "risk.calculate_conditional_drawdown_risk"
  - "risk.calculate_variance"
  - "risk.calculate_value_at_risk_from_matrix"
  - "risk.calculate_expected_shortfall_from_matrix"
  - "risk.calculate_conditional_drawdown_risk_from_matrix"
sidebar:
  badge: Module
---

Four risk numbers computed from a sample, with no distribution assumed. The module is a port
of mlfinlab's `RiskMetrics` class, and its user inside this crate is
[`pipeline`](/modules/pipeline/), which reports these figures for a strategy's returns. It is
not from AFML. In Rust the functions are methods on the unit
struct `RiskMetrics`; in Python they are functions in `openquant.risk`.

## What each one measures

**Variance** of a portfolio with weights $w$ and covariance $\Sigma$ is $w^\top\Sigma w$.

**Value at risk** at level $\alpha$ is the $\alpha$-quantile of the returns: the return that
is undercut with probability $\alpha$. It says where the tail starts and nothing about what
is in it. Two portfolios can share a 5% VaR while one loses twice as much as the other on its
bad days.

**Expected shortfall** (conditional VaR) is the mean of what lies beyond that point,

$$
\mathrm{ES}_\alpha \;=\; \mathrm{E}\bigl[\,r \mid r < \mathrm{VaR}_\alpha\,\bigr]
$$

and unlike VaR it is a coherent risk measure in the sense of Artzner et al. (1999): the
expected shortfall of a combined book never exceeds the sum of its parts', which VaR can
violate.

**Conditional drawdown at risk** (Chekhlov, Uryasev and Zabarankin, 2005) applies the same
idea to drawdowns. For a cumulative series $x$ — an equity curve, a price or a cumulative
return — the drawdown is $d_t = \max_{s\le t} x_s - x_t$, in the units of $x$, and CDaR at
level $\alpha$ is the mean of the worst $1-\alpha$ share of the $d_t$. It lies between the
$\alpha$-quantile of the drawdowns and the maximum drawdown. Here `confidence_level` is that
**upper-tail** level: 0.95 averages the worst 5% of drawdowns.

VaR and expected shortfall are returned **as returns, with their sign** — a 5% VaR of
−0.0166 means a loss of 1.66% — and their `confidence_level` is the **tail probability**,
0.05, not 0.95.

```python
import random

from openquant import risk

# 1,000 daily returns with occasional gaps down.
rng = random.Random(9)
returns = [rng.gauss(0.0005, 0.01) - (0.04 if rng.random() < 0.01 else 0.0) for _ in range(1000)]

var = risk.calculate_value_at_risk(returns, 0.05)
es = risk.calculate_expected_shortfall(returns, 0.05)
print(f"5% VaR {var:+.4f}   expected shortfall {es:+.4f}   ratio {es / var:.2f}")

# The same distribution without the gaps.
normal = [rng.gauss(0.0005, 0.01) for _ in range(1000)]
ratio = risk.calculate_expected_shortfall(normal, 0.05) / risk.calculate_value_at_risk(normal, 0.05)
print(f"same, without the gaps: ratio {ratio:.2f}")

print(f"portfolio variance {risk.calculate_variance([[0.04, 0.01], [0.01, 0.09]], [0.6, 0.4]):.4f}")
```

```text
5% VaR -0.0166   expected shortfall -0.0242   ratio 1.46
same, without the gaps: ratio 1.27
portfolio variance 0.0336
```

For a normal distribution the 5% expected shortfall is about 1.25 times the 5% VaR. The
gapping series comes out at 1.46: its VaR barely registers the gaps, because a 1% event sits
well inside a 5% tail, and its expected shortfall does.

## How the quantile is taken

The quantile uses the **"higher" rule**: with $n$ sorted returns it returns the element at
index $\lceil \alpha\,(n-1)\rceil$, never an interpolated value. That matches
`numpy.quantile(..., method="higher")`, which is what mlfinlab calls. Two consequences: VaR
is always an observed return, and on small samples it is coarse — with 20 returns the 5% VaR
is simply the second-worst.

Expected shortfall then averages the returns **strictly below** VaR. If none are, as with
constant returns or $\alpha=0$, it returns `NaN` rather than an error.

Conditional drawdown at risk takes the same "higher" quantile of the drawdowns and averages
every drawdown **at or above** it, so its tail always holds at least the maximum drawdown and
the result is never `NaN` on finite input.

## From Rust

```rust
use nalgebra::DMatrix;
use openquant::risk_metrics::{RiskMetrics, RiskMetricsError};

let risk = RiskMetrics;
let returns = [-0.08, -0.03, -0.01, 0.0, 0.01, 0.01, 0.02, 0.02, 0.03, 0.04];

// ceil(0.25 * 9) = 3: the fourth-smallest return. The three below it average -0.04.
assert_eq!(risk.calculate_value_at_risk(&returns, 0.25)?, 0.0);
assert!((risk.calculate_expected_shortfall(&returns, 0.25)? + 0.04).abs() < 1e-12);

// Nothing lies strictly below the minimum, so the tail is empty.
assert!(risk.calculate_expected_shortfall(&returns, 0.0)?.is_nan());

// Conditional drawdown at risk takes a cumulative series, and 0.9 means the worst 10%.
// Drawdowns 0 0 1 0 4 1. At 0.6 the threshold is the fourth-smallest, 1; (1, 1, 4) average 2.
let equity = [1.0, 3.0, 2.0, 5.0, 1.0, 4.0];
assert_eq!(risk.calculate_conditional_drawdown_risk(&equity, 0.6)?, 2.0);
assert_eq!(risk.calculate_conditional_drawdown_risk(&equity, 0.9)?, 4.0);

let covariance = DMatrix::from_row_slice(2, 2, &[0.04, 0.01, 0.01, 0.09]);
assert!((risk.calculate_variance(&covariance, &[0.6, 0.4])? - 0.0336).abs() < 1e-12);
assert_eq!(risk.calculate_variance(&covariance, &[1.0]), Err(RiskMetricsError::DimensionMismatch));
assert_eq!(
    risk.calculate_value_at_risk(&returns, 1.5),
    Err(RiskMetricsError::InvalidConfidenceLevel)
);
```

## What to watch for

- **`calculate_conditional_drawdown_risk` wants an equity curve, not returns**, although
  the Python keyword is still called `returns`. On per-period returns
  $\max_{s\le t} x_s - x_t$ is not a drawdown, and the function has no way to tell.
  [`pipeline`](/modules/pipeline/) passes its strategy's equity curve, at one minus its
  `confidence_level`. [`hcaa`](/modules/hcaa/) has its own drawdown measure, built on the
  drawdown series of a wealth curve.
- **Its `confidence_level` is the upper-tail level**, 0.95 for the worst 5%, the opposite of
  VaR and expected shortfall beside it. Passing 0.05 averages nearly every drawdown, which
  is not a tail statistic.
- **Versions before [#102](https://github.com/Open-Quant/openquant/issues/102) was fixed**,
  and mlfinlab, which they copied, averaged the *running maximum* of the drawdown strictly
  above its quantile. That series only rises, so at 0.95 on a realistic equity curve they
  returned `NaN`. Numbers from those versions are not comparable with today's.
- **Historical estimates cannot see what has not happened.** A 1% expected shortfall from 500
  observations is the mean of five numbers. Nothing here fits a tail, scales with horizon, or
  weights recent data.
- **No annualisation and no horizon.** A VaR of daily returns is a one-day VaR. Scaling by
  $\sqrt{h}$ assumes independent, identically distributed returns, which is the assumption the
  rest of this library exists to avoid.
- **The `_from_matrix` variants read the first column only**, silently ignoring the rest. They
  exist for mlfinlab compatibility, where the input was a one-column frame.
- **`NaN` inputs sort to the end** and count toward $n$, which shifts the quantile. Drop them
  first.

## Related modules

- [`hcaa`](/modules/hcaa/) — allocates between clusters by variance, expected shortfall or
  conditional drawdown, with its own implementations of the last two.
- [`pipeline`](/modules/pipeline/) — reports these measures for a strategy.
- [`backtest-statistics`](/modules/backtest-statistics/) — drawdown and time under water of a
  track record.
- [`strategy-risk`](/modules/strategy-risk/) — the risk that the strategy itself stops
  working, which none of these measure.

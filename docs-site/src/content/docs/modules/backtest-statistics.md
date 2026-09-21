---
title: "backtest_statistics"
description: "Sharpe ratios with their uncertainty — probabilistic and deflated Sharpe, minimum track record length — plus drawdown, concentration and holding-period statistics."
status: authored
last_authored: '2026-09-20'
audience:
  - quant-dev
  - platform-engineering
module: "backtest_statistics"
api_surface: "both"
afml_chapter:
  - "14"
citation:
  - "López de Prado, M. (2018). Advances in Financial Machine Learning. Wiley. Chapter 14: §14.3 General Characteristics (Snippets 14.1–14.2); §14.5.1 Returns Concentration (Snippet 14.3); §14.5.2 Drawdown and Time under Water (Snippet 14.4); §14.7.1 The Sharpe Ratio; §14.7.2 The Probabilistic Sharpe Ratio; §14.7.3 The Deflated Sharpe Ratio."
  - "Bailey, D. H. and López de Prado, M. (2012). The Sharpe ratio efficient frontier. Journal of Risk 15(2), 3–44."
  - "Bailey, D. H. and López de Prado, M. (2014). The deflated Sharpe ratio: correcting for selection bias, backtest overfitting, and non-normality. Journal of Portfolio Management 40(5), 94–107."
rust_api:
  - "sharpe_ratio"
  - "information_ratio"
  - "probabilistic_sharpe_ratio"
  - "deflated_sharpe_ratio"
  - "minimum_track_record_length"
  - "drawdown_and_time_under_water"
  - "bets_concentration"
  - "all_bets_concentration"
  - "average_holding_period"
  - "timing_of_flattening_and_flips"
python_api:
  - "backtest_stats.sharpe_ratio"
  - "backtest_stats.information_ratio"
  - "backtest_stats.probabilistic_sharpe_ratio"
  - "backtest_stats.deflated_sharpe_ratio"
  - "backtest_stats.minimum_track_record_length"
  - "backtest_stats.drawdown_and_time_under_water"
  - "backtest_stats.bets_concentration"
  - "backtest_stats.all_bets_concentration"
  - "backtest_stats.average_holding_period"
  - "backtest_stats.timing_of_flattening_and_flips"
sidebar:
  badge: Module
---

A Sharpe ratio computed from a backtest is an estimate, from a finite sample, of a quantity
that was then selected for being large. Both facts bias it upward, and neither is visible in
the number itself. The statistics in this module that matter most are the ones that put the
uncertainty back: how likely is it that the true Sharpe ratio is above a benchmark (PSR),
what that benchmark should be once you admit how many things you tried (DSR), and how long a
track record has to be before the question can be answered at all (MinTRL). The Python module
is `openquant.backtest_stats`.

## Sharpe ratio, with error bars

The estimated Sharpe ratio $\widehat{SR}$ of $T$ returns with skewness $\hat\gamma_3$ and
kurtosis $\hat\gamma_4$ is asymptotically normal around the true value, with a variance that
grows when returns are negatively skewed or fat-tailed (Bailey and López de Prado, 2012). The
**probabilistic Sharpe ratio** is the resulting confidence that the true Sharpe ratio exceeds
a benchmark $SR^*$:

$$
\widehat{PSR}(SR^*) \;=\; Z\!\left[\frac{(\widehat{SR}-SR^*)\sqrt{T-1}}
{\sqrt{1-\hat\gamma_3\,\widehat{SR}+\frac{\hat\gamma_4-1}{4}\,\widehat{SR}^{2}}}\right]
$$

with $Z$ the standard normal CDF. Two conventions are easy to get wrong and the function
checks neither: **$\widehat{SR}$ is per period, not annualised**, and **$\hat\gamma_4$ is raw
kurtosis, 3 for a normal distribution**, not excess.

The **minimum track record length** inverts the same expression: the number of observations
needed for $\widehat{PSR}(SR^*)$ to reach $1-\alpha$,

$$
\mathrm{MinTRL} \;=\; 1+\Bigl[1-\hat\gamma_3\,\widehat{SR}+\tfrac{\hat\gamma_4-1}{4}\,\widehat{SR}^{2}\Bigr]
\left(\frac{Z_{1-\alpha}}{\widehat{SR}-SR^*}\right)^{2}
$$

## Deflating for the trials you ran

If $N$ strategy configurations with no skill at all are backtested, the best of them still
has a positive Sharpe ratio. With $\sigma_{SR}$ the standard deviation of the trials' Sharpe
ratios, its expected value is approximately (Bailey and López de Prado, 2014)

$$
SR_0 \;=\; \sigma_{SR}\Bigl[(1-\gamma)\,Z^{-1}\!\bigl(1-\tfrac1N\bigr)
+\gamma\,Z^{-1}\!\bigl(1-\tfrac{1}{N e}\bigr)\Bigr]
$$

where $\gamma\approx0.5772$ is the Euler–Mascheroni constant. The **deflated Sharpe ratio**
is simply $\widehat{PSR}(SR_0)$: the PSR measured against the Sharpe ratio that luck alone
would have produced. `deflated_sharpe_ratio` takes the trials in one of two forms:

- `estimates_param = false`: `sr_estimates` is the list of every trial's Sharpe ratio.
  $\sigma_{SR}$ and $N$ are computed from it.
- `estimates_param = true`: `sr_estimates` is `[σ_SR, N]`, for when the individual trials
  were not kept.

With `benchmark_out = true` it returns $SR_0$ instead of the probability.

```python
import random

from openquant import backtest_stats as bs

# Three years of daily returns from a strategy that earns steadily and occasionally gaps down:
# positive mean, negative skew, fat tails.
rng = random.Random(4)
returns = [rng.gauss(0.0011, 0.009) - (0.035 if rng.random() < 0.012 else 0.0) for _ in range(756)]

n = len(returns)
mean = sum(returns) / n
m2, m3, m4 = (sum((r - mean) ** p for r in returns) / n for p in (2, 3, 4))
skew, kurt = m3 / m2**1.5, m4 / m2**2
sr = mean / (m2 * n / (n - 1)) ** 0.5  # per-period Sharpe: this is what PSR and DSR take

print(f"annualised Sharpe {bs.sharpe_ratio(returns, 252, 0.0):.2f}   per-period {sr:.4f}   "
      f"skew {skew:.2f}   kurtosis {kurt:.2f}")
print(f"PSR(0): {bs.probabilistic_sharpe_ratio(sr, 0.0, n, skew, kurt):.3f}   "
      f"if returns were normal: {bs.probabilistic_sharpe_ratio(sr, 0.0, n, 0.0, 3.0):.3f}")

# The same track record, had it been the best of N configurations whose per-period
# Sharpe ratios had a standard deviation of 0.03.
for trials in (2, 10, 100, 1000):
    hurdle = bs.deflated_sharpe_ratio(sr, [0.03, trials], n, skew, kurt, True, True)
    dsr = bs.deflated_sharpe_ratio(sr, [0.03, trials], n, skew, kurt, True, False)
    print(f"best of {trials:4d}: expected max Sharpe {hurdle:.4f}   DSR {dsr:.3f}")

days = bs.minimum_track_record_length(sr, 0.0, skew, kurt, 0.05)
print(f"minimum track record at 95%: {days:.0f} observations ({days / 252:.1f} years)")
```

```text
annualised Sharpe 0.98   per-period 0.0615   skew -0.42   kurtosis 4.08
PSR(0): 0.952   if returns were normal: 0.954
best of    2: expected max Sharpe 0.0156   DSR 0.893
best of   10: expected max Sharpe 0.0472   DSR 0.651
best of  100: expected max Sharpe 0.0759   DSR 0.348
best of 1000: expected max Sharpe 0.0977   DSR 0.164
minimum track record at 95%: 736 observations (2.9 years)
```

A Sharpe ratio of about 1 over three years clears a 95% confidence bar against zero, barely:
the minimum track record is 736 days and the sample is 756. The non-normality correction is
small at this Sharpe level, because it scales with $\widehat{SR}$ per period, which is 0.06.
The number of trials is what moves the answer. If this backtest was the pick of a hundred
variations, the luckiest of a hundred skill-less strategies would be expected to show a
per-period Sharpe of 0.076 — *higher than the one observed* — and the probability that this
strategy has any skill falls to 0.35.

## Describing the path

| Function | Returns |
| --- | --- |
| `sharpe_ratio(returns, entries_per_year, risk_free_rate)` | $(\bar r - r_f)/s \cdot\sqrt{\text{entries per year}}$, with the sample deviation; `risk_free_rate` is **per period** |
| `information_ratio(returns, benchmark, entries_per_year)` | the same on `returns − benchmark`, where `benchmark` is one per-period number |
| `drawdown_and_time_under_water(series, dollars)` | one drawdown per high-water mark that was followed by a dip, and the years spent under each |
| `bets_concentration(returns)` | normalised Herfindahl index of the returns' shares of the total: 0 if every bet contributed equally, 1 if one bet made everything |
| `all_bets_concentration(returns)` | that index for positive returns, for negative returns, and for the number of bets per day |
| `average_holding_period(positions)` | average days a unit of position is held, weighting each exit by its size (Snippet 14.2) |
| `timing_of_flattening_and_flips(positions)` | timestamps where the position went to zero or changed sign, plus the last timestamp (Snippet 14.1) |

Concentration is the one to look at after the Sharpe ratio. A strategy whose positive
concentration is 0.3 made most of its money on a handful of days, and its Sharpe ratio
describes those days.

## From Rust

```rust
use openquant::backtest_statistics::{
    bets_concentration, deflated_sharpe_ratio, minimum_track_record_length,
    probabilistic_sharpe_ratio, sharpe_ratio,
};

// Per-period Sharpe 0.1 over 500 normal returns: confident it is positive...
let psr = probabilistic_sharpe_ratio(0.1, 0.0, 500, 0.0, 3.0);
assert!((psr - 0.9871).abs() < 1e-4);
// ...less so with negative skew and fat tails, and less again after 50 trials.
assert!(probabilistic_sharpe_ratio(0.1, 0.0, 500, -2.0, 10.0) < psr);
let hurdle = deflated_sharpe_ratio(0.1, &[0.05, 50.0], 500, 0.0, 3.0, true, true)?;
assert!((hurdle - 0.1138).abs() < 1e-4);
assert!(deflated_sharpe_ratio(0.1, &[0.05, 50.0], 500, 0.0, 3.0, true, false)? < 0.5);

// Observations needed to be 95% confident that a per-period Sharpe of 0.1 beats zero.
let min_trl = minimum_track_record_length(0.1, 0.0, 0.0, 3.0, 0.05)?;
assert!((min_trl - 272.9).abs() < 0.1);

// Equal bets are not concentrated; one dominant bet is.
assert!(bets_concentration(&[1.0, 1.0, 1.0, 1.0]).unwrap().abs() < 1e-12);
assert!(bets_concentration(&[97.0, 1.0, 1.0, 1.0]).unwrap() > 0.9);

// Annualising multiplies by the square root of the periods per year.
let r = [0.01, -0.005, 0.007, 0.002, -0.001];
assert!((sharpe_ratio(&r, 252.0, 0.0) / sharpe_ratio(&r, 1.0, 0.0) - 252f64.sqrt()).abs() < 1e-9);
```

## What to watch for

- **Passing an annualised Sharpe ratio to PSR, DSR or MinTRL is silently wrong.** They expect
  the per-period value with `number_of_returns` periods. With the example's annualised 0.98
  in place of 0.0615, `probabilistic_sharpe_ratio` returns 1.000 and the track record looks
  proven. Nothing in the signature stops you.
- **`minimum_track_record_length` is meaningless when the Sharpe ratio is at or below the
  benchmark**, and it does not say so: the difference is squared, so an underperforming
  strategy gets a finite positive answer, and an equal one gets infinity.
- **`drawdown_and_time_under_water` takes a cumulative series, not returns** — equity, NAV or
  cumulative PnL — despite its parameter being named `returns`. With `dollars = false` it
  computes $1-\text{trough}/\text{peak}$, which needs a positive series; use `dollars = true`
  for a PnL that starts at or crosses zero.
- **Time under water is measured to the next high-water mark *that was itself followed by a
  drawdown***, not to the recovery. After the series `100, 110, 105, 112, 115, 120, 118, 121`
  at daily steps, the dip below 110 recovers in two days but is reported as four, because 112
  and 115 were new highs with no dip after them. This reproduces Snippet 14.4 and mlfinlab;
  it overstates time under water whenever a recovery is followed by a run of new highs.
- **`all_bets_concentration` counts bets per calendar day.** Snippet 14.3 and mlfinlab group by
  month. Daily grouping over a span with weekends puts many zeros in the count and raises the
  index; compare it only with itself.
- **The expected maximum assumes the trials are independent and centred on zero.** Variations
  of one idea are highly correlated, so the *effective* number of trials is smaller than the
  count; AFML suggests clustering the trials ([`onc`](/modules/onc/)) and counting clusters.
  Counting too few trials is the more common error.
- **`sharpe_ratio` does not validate.** An empty slice gives `NaN`, one return gives `NaN`,
  constant returns give infinity.

## Related modules

- [`backtesting-engine`](/modules/backtesting-engine/) — produces the out-of-sample returns
  and carries the trial count these statistics need.
- [`hyperparameter-tuning`](/modules/hyperparameter-tuning/) — every search trial counts
  toward $N$.
- [`strategy-risk`](/modules/strategy-risk/) — the Sharpe ratio implied by precision and bet
  frequency.
- [`risk-metrics`](/modules/risk-metrics/) — VaR, expected shortfall and drawdown-at-risk.

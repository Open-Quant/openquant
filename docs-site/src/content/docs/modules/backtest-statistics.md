---
title: "backtest_statistics"
description: "Performance diagnostics for strategy returns and position trajectories."
status: generated
generated_from: src/data/moduleDocs.ts
last_generated: '2026-08-30'
banner:
  content: '<span class="doc-status doc-status--generated">Generated</span> Assembled automatically from <code>moduleDocs.ts</code>. No human has reviewed this page.'
audience:
  - quant-dev
  - platform-engineering
module: "backtest_statistics"
api_surface: "both"
rust_api:
  - "sharpe_ratio"
  - "deflated_sharpe_ratio"
  - "probabilistic_sharpe_ratio"
  - "drawdown_and_time_under_water"
  - "average_holding_period"
sidebar:
  badge: Module
---

## Concept Overview

Turns a return or equity series into the handful of statistics a strategy is actually judged on: annualised Sharpe, information ratio, the drawdown and time-under-water profile, average holding period, bet concentration, and the multiple-testing corrections — probabilistic and deflated Sharpe — that say whether a Sharpe is real. Those corrections are why this module exists rather than a two-line Sharpe helper: AFML Chapter 14's point is that a Sharpe reported without the number of trials behind it is uninterpretable.

## When to Use

Reach for it after a backtest run, at model-selection time, and again in production monitoring. Use `deflated_sharpe_ratio` whenever the strategy is the survivor of a search — a grid, a parameter sweep, a family of variants — and pass the trial count honestly; `sharpe_ratio` alone flatters every one of them. Note that `drawdown_and_time_under_water` consumes a timestamped equity curve, not a return vector, and that every annualisation constant must match your bar frequency.

## Mathematical Foundations

### Sharpe Ratio

$$\mathrm{SR}=\frac{\mu-r_f}{\sigma}\sqrt{n}$$

where $\mu$ and $\sigma$ are the mean and standard deviation of the per-bar returns, $r_f$ the per-bar risk-free rate, and $n$ the number of bars per year (`entries_per_year`) — the annualisation constant must match your bar frequency.

### Information Ratio

$$\mathrm{IR}=\frac{\mu-r_b}{\sigma_{(r-r_b)}}$$

where $r_b$ is the benchmark return and $\sigma_{(r-r_b)}$ the tracking error, i.e. the standard deviation of the *excess* return series.

### Probabilistic Sharpe Ratio

$$\mathrm{PSR}(\mathrm{SR}^*)=Z\left[\frac{(\widehat{\mathrm{SR}}-\mathrm{SR}^*)\sqrt{T-1}}{\sqrt{1-\hat\gamma_3\widehat{\mathrm{SR}}+\frac{\hat\gamma_4-1}{4}\widehat{\mathrm{SR}}^2}}\right]$$

where $Z[\cdot]$ is the standard normal CDF, $\widehat{\mathrm{SR}}$ the observed (non-annualised) Sharpe ratio, $\mathrm{SR}^*$ the benchmark being tested against, $T$ the number of returns, and $\hat\gamma_3,\hat\gamma_4$ the sample skewness and kurtosis. Non-normal returns lower the confidence a given Sharpe deserves.

### Deflated Sharpe Ratio

$$\mathrm{DSR}=\mathrm{PSR}(\mathrm{SR}_0),\qquad \mathrm{SR}_0=\sqrt{V[\{\widehat{\mathrm{SR}}_n\}]}\left((1-\gamma)Z^{-1}\!\left[1-\tfrac{1}{N}\right]+\gamma Z^{-1}\!\left[1-\tfrac{e^{-1}}{N}\right]\right)$$

where $N$ is the number of strategy variants you tried, $V[\{\widehat{\mathrm{SR}}_n\}]$ the variance of their Sharpe ratios, $\gamma\approx0.5772$ the Euler-Mascheroni constant, and $Z^{-1}$ the normal quantile function. $\mathrm{SR}_0$ is the Sharpe you would *expect* the best of $N$ independent worthless strategies to post, so DSR is the PSR measured against that bar instead of against zero. `deflated_sharpe_ratio` accepts either the raw $\{\widehat{\mathrm{SR}}_n\}$ or the $(\text{sd}, N)$ pair via `estimates_param`.

## Usage Examples

### Rust

#### Compute Sharpe and drawdown

```rust
use chrono::{Duration, NaiveDateTime};
use openquant::backtest_statistics::{drawdown_and_time_under_water, sharpe_ratio};

let returns = vec![0.01, -0.005, 0.007, -0.002, 0.003];
let sharpe = sharpe_ratio(&returns, 252.0, 0.0);

// Drawdown and time-under-water are computed on a *timestamped equity curve*,
// not on the return series: the function needs the timestamps to measure how
// long each high-water mark went un-recovered.
let t0 = NaiveDateTime::parse_from_str("2024-01-02 00:00:00", "%Y-%m-%d %H:%M:%S")?;
let mut equity = 1.0;
let curve: Vec<(NaiveDateTime, f64)> = returns
    .iter()
    .enumerate()
    .map(|(i, r)| {
        equity *= 1.0 + r;
        (t0 + Duration::days(i as i64), equity)
    })
    .collect();

// dollars = false reports each drawdown as a fraction of its high-water mark.
let (drawdowns, time_under_water) = drawdown_and_time_under_water(&curve, false);
println!("sharpe={sharpe:.3} drawdowns={drawdowns:?} tuw={time_under_water:?}");
```

## API Reference

### Python API

- `backtest_stats.sharpe_ratio`
- `backtest_stats.information_ratio`
- `backtest_stats.probabilistic_sharpe_ratio`
- `backtest_stats.deflated_sharpe_ratio`
- `backtest_stats.minimum_track_record_length`
- `backtest_stats.timing_of_flattening_and_flips`
- `backtest_stats.average_holding_period`
- `backtest_stats.bets_concentration`
- `backtest_stats.all_bets_concentration`
- `backtest_stats.drawdown_and_time_under_water`

### Rust API

- `sharpe_ratio`
- `deflated_sharpe_ratio`
- `probabilistic_sharpe_ratio`
- `drawdown_and_time_under_water`
- `average_holding_period`

## Risk Notes and Caveats

- Use annualization constants consistent with your bar frequency.
- Deflated Sharpe is useful when strategy mining many variants.

## Related Modules

- [`backtesting-engine`](/modules/backtesting-engine/)
- [`strategy-risk`](/modules/strategy-risk/)
- [`risk-metrics`](/modules/risk-metrics/)
- [`synthetic-backtesting`](/modules/synthetic-backtesting/)

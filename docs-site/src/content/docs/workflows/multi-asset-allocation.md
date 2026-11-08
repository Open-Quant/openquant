---
title: Align observations and calculate portfolio weights
description: Keep timestamps, asset columns, covariance and named weights aligned through a small allocation workflow.
status: authored
last_authored: '2026-11-07'
audience:
  - quant-dev
  - platform-engineering
sidebar:
  order: 5
---

This walkthrough starts with long-format OHLCV observations and ends with named
inverse-variance weights. It makes the timestamp and asset ordering explicit at each
handoff. The [portfolio module](/modules/portfolio-optimization/) explains the solvers;
this page shows how to prepare their inputs from one coherent frame.

The observations are synthetic and deliberately tiny. Three return rows cannot establish
a reliable covariance estimate or a useful allocation. This is an integration check,
not an allocation recommendation, execution simulation or out-of-sample backtest.
Complete [Python bindings setup](/setup/python-bindings/) first; the example uses the
installed NumPy and Polars dependencies and needs no external data feed.

## Choose a clock and an asset order

The fixture has two assets over five specified dates, with the first date missing for
BETA. Rows arrive in reverse order. Cleaning normalizes the frame; pivoting then makes
one row per timestamp and one column per asset. We explicitly retain only dates observed
for both assets and fix the column order to `names` before constructing a matrix.

This intersection policy discards one date in the fixture. It is not automatically the
right missing-data policy for a study. Report which dates were removed; missing interior
dates can change return horizons, and selecting assets based on later availability can
introduce survivorship bias. Do not fill missing prices with zero or future observations.
The library does not infer exchange calendars, common close times or corporate-action
adjustments from these rows. Choose a consistent price basis and availability convention.

## Compute inputs, allocate and restore the names

```python
import numpy as np
import polars as pl
from openquant import adapters, data, portfolio

names = ["ALPHA", "BETA"]  # explicit asset order for every matrix/vector
rows = []
for symbol, dates, prices in (
    ("ALPHA", ["2024-01-08", "2024-01-09", "2024-01-10", "2024-01-11", "2024-01-12"], [100., 101., 100., 103., 102.]),
    ("BETA", ["2024-01-09", "2024-01-10", "2024-01-11", "2024-01-12"], [50., 51., 50.5, 52.]),
):
    for ts, price in zip(dates, prices):
        rows.append({"ts": ts, "symbol": symbol, "open": price, "high": price,
                     "low": price, "close": price, "volume": 1.0})
cleaned, report = data.clean_ohlcv(pl.DataFrame(rows[::-1]), return_report=True)
assert cleaned.height == len(rows)
wide = cleaned.pivot(on="symbol", index="ts", values="close").sort("ts")
assert set(wide.columns) == {"ts", *names}
# Common observed dates only. Do not fill an absent asset with zero or a later price.
aligned = wide.drop_nulls(names).select("ts", *names)
assert aligned.height == 4 and wide.height - aligned.height == 1
prices = aligned.select(names).to_numpy()
assert np.isfinite(prices).all() and (prices > 0).all()
returns = prices[1:] / prices[:-1] - 1.0
mu = returns.mean(axis=0)
cov = np.cov(returns, rowvar=False, ddof=1)
assert np.isfinite(cov).all() and (np.diag(cov) > 0).all()
weights, risk, expected_return, sharpe = portfolio.allocate_from_inputs(
    mu.tolist(), cov.tolist(), "inverse_variance", risk_free_rate=0.0,
)
weights = np.asarray(weights)
assert np.isclose(weights.sum(), 1.0) and (weights >= 0).all()
assert np.allclose(weights, (1 / np.diag(cov)) / (1 / np.diag(cov)).sum())
assert np.isclose(risk, np.sqrt(weights @ cov @ weights))
assert np.isclose(expected_return, weights @ mu)
frame = adapters.to_polars_weights_frame(names, weights.tolist(), as_of=str(aligned["ts"][-1]))
assert frame["asset"].to_list() == names
print("observations:", cleaned.height, "common dates:", aligned.height)
print("matrix: dates x assets =", prices.shape, "return rows:", len(returns))
print("weights:", [(name, round(float(weight), 4)) for name, weight in zip(names, weights)])
```

```text
observations: 9 common dates: 4
matrix: dates x assets = (4, 2) return rows: 3
weights: [('ALPHA', 0.4452), ('BETA', 0.5548)]
```

The assertions check retained row counts, positive finite prices, matrix orientation,
positive variances, normalized weights and the reported risk/return identities. They
also compare the allocation with the independent reciprocal-variance formula for this
unconstrained long-only case. These checks cannot establish reliable estimates from such
a short sample or prove the source was available historically.

`prices` is **dates × assets**. NumPy's `rowvar=False` computes covariance between columns,
and `ddof=1` selects sample covariance. `mu`, both covariance axes, the returned weights
and the names all share the explicit order. The adapter preserves that order; it does
not discover asset names from the covariance or repair a mismatched vector.

## Keep units and the fitting boundary explicit

The example supplies means and covariance directly to `allocate_from_inputs`, without
annualizing. Expected return and risk are per retained observation interval; risk is a
standard deviation, not variance. Here the retained dates represent four successive
chosen daily observations. In general, an irregular intersection does not define a
uniform daily return horizon. Revisit alignment before interpreting those estimates.

The supplied risk-free value is zero in the same units. If you annualize the mean and
covariance by a common period count, scale the risk-free value consistently. This differs
from the price-taking allocator, which uses its documented `252 / step` convention; see
[allocation from prices](/modules/portfolio-optimization/).

The fourth returned item is the portfolio Sharpe computed from the supplied moments;
it is not a realized trading result. The sample fits moments and weights using all
retained history, so these weights are available only after the final observation. The
`as_of` field is a timestamp string passed to the adapter and parsed into a repeated
Datetime column. It records the chosen label; it does not enforce when a trade could occur.

To evaluate an allocation rule, refit it using each training window, hold the chosen
weights fixed according to a specified rebalance/execution policy, then score later data
with explicit costs. Preserve the window's asset order and missing-data decisions. The
[HRP/IVP/CLA runbook](/runbooks/hrp-vs-ivp-cla-oos/) illustrates out-of-sample allocation
comparisons under its synthetic assumptions.

## Continue with the relevant contract

- [Data cleaning and provenance](/modules/data/) — duplicate policy, timestamps and hashes.
- [Portfolio optimization](/modules/portfolio-optimization/) — methods, bounds and units.
- [Adapters](/modules/adapters/) — named frames and timestamp-string inputs.
- [Input errors](/setup/input-errors/) — shape, covariance and timestamp diagnostics.

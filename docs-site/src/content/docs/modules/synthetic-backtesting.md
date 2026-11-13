---
title: "synthetic_backtesting"
description: "Choose profit-taking and stop-loss levels on simulated paths of a fitted mean-reverting process, instead of on the one historical path."
status: authored
last_authored: '2026-11-09'
audience:
  - quant-dev
  - platform-engineering
module: "synthetic_backtesting"
api_surface: "both"
afml_chapter:
  - "13"
citation:
  - "López de Prado, M. (2018). Advances in Financial Machine Learning. Wiley. Chapter 13: §13.2 Trading Rules; §13.3 The Problem; §13.4 Our Framework; §13.5 Numerical Determination of Optimal Trading Rules (§13.5.1 The Algorithm, Snippets 13.1–13.2); §13.6 Experimental Results."
  - "Bailey, D. H. and López de Prado, M. (2013). Drawdown-based stop-outs and the triple penance rule. Journal of Risk 18(2), 61–75."
rust_api:
  - "run_synthetic_otr_workflow"
  - "calibrate_ou_params"
  - "generate_ou_paths"
  - "evaluate_rule_on_paths"
  - "search_optimal_trading_rule"
  - "detect_no_stable_optimum"
  - "OuProcessParams"
  - "TradingRule"
  - "RuleSurfacePoint"
  - "StabilityCriteria"
  - "StabilityDiagnostics"
  - "OtrSearchResult"
  - "SyntheticBacktestConfig"
  - "SyntheticBacktestError"
python_api:
  - "synthetic_bt.run_synthetic_otr_workflow"
  - "synthetic_bt.calibrate_ou_params"
  - "synthetic_bt.generate_ou_paths"
  - "synthetic_bt.evaluate_rule_on_paths"
  - "synthetic_bt.search_optimal_trading_rule"
  - "synthetic_bt.detect_no_stable_optimum"
sidebar:
  badge: Module
---

Where to take profit and where to stop out are usually chosen by trying many pairs of levels
on history and keeping the best. With two free parameters and one price path, that is about
the easiest way there is to overfit a backtest (AFML §13.3). Chapter 13's alternative is to
fit a low-dimensional model of the price process, simulate many paths from it, and choose
the exit rule on those. This separates rule selection from replaying one historical path,
but does not remove estimation error or model misspecification. The selected rule is
conditional on the fitted model and the finite simulation sample. The Python module is `openquant.synthetic_bt`.

## The procedure

**1. Fit.** `calibrate_ou_params(prices)` fits a discrete Ornstein–Uhlenbeck process, an
AR(1), by least squares:

$$
P_t \;=\; (1-\varphi)\,\mathrm{E}[P] + \varphi\,P_{t-1} + \sigma\,\varepsilon_t,
\qquad \varepsilon_t\sim N(0,1)
$$

For $0<\varphi<1$, the reversion half-life is $-\ln 2/\ln\varphi$: 6.6 steps at
$\varphi=0.9$, 138 at $0.995$. Negative stationary coefficients produce alternating
deviations, so this formula does not apply directly. At $\varphi=1$, the regression
is a random walk with drift given by its intercept; it has no finite equilibrium.
`OuProcessParams` carries `phi`, `equilibrium`, `sigma`, the regression's `r_squared`, and
`stationary`, true when $\lvert\varphi\rvert<1$.

**2. Simulate.** `generate_ou_paths(params, initial_price, n_paths, horizon, seed)` draws the
paths. `initial_price` is where the trade is entered, and its distance from `equilibrium` is
the mean-reversion forecast when the fitted process is stationary. The expected change
at step $h$ is $(\mathrm{E}[P]-P_0)(1-\varphi^h)$ when the equilibrium equals
`intercept / (1 − phi)`; the near-unit fallback described below does not satisfy that
identity. A barrier exit changes the payoff distribution.

**3. Evaluate.** `evaluate_rule_on_paths` enters **long at the first price of every path**
and exits at the first step where the profit reaches `profit_taking` or the loss reaches
`stop_loss`, or at `max_holding_steps` (the final available step for a shorter path). It returns the mean, deviation and Sharpe ratio of the
exit PnL across paths, the win rate and the average holding time. Both widths must be
finite and strictly positive. A nonfinite width raises a field-specific Rust `Invalid`
error (Python `ValueError`); it cannot be used to disable a barrier.

**4. Search.** `search_optimal_trading_rule` evaluates every pair on a grid — on the same
paths, so rules are compared on identical draws — and returns the surface sorted by Sharpe
ratio. Each grid width must also be finite and strictly positive.
`run_synthetic_otr_workflow` does all four steps.

## Inspect the simulated rule surface

The experiments in AFML §13.6 motivate comparing the shape of the Sharpe surface across
processes. In the example below, the mean-reverting fit produces a pronounced peak while
the near-random-walk fit gives a flatter surface. `detect_no_stable_optimum` combines
coefficient and surface thresholds to flag weak or flat results in `diagnostics`. It is a
heuristic, not a proof that a profitable rule exists or that none can exist.

```python
import random

from openquant import synthetic_bt as sb

def history(phi, equilibrium=100.0, sigma=1.0, n=1500, seed=3):
    rng, p, out = random.Random(seed), equilibrium, []
    for _ in range(n):
        p = (1 - phi) * equilibrium + phi * p + sigma * rng.gauss(0, 1)
        out.append(p)
    return out

grid = [0.5, 1.0, 2.0, 4.0, 8.0]  # barrier widths in price units; sigma is 1
for label, phi in (("mean-reverting", 0.90), ("near random walk", 0.995)):
    res = sb.run_synthetic_otr_workflow(
        history(phi), initial_price=97.0, n_paths=4000, horizon=60, seed=11,
        profit_taking_grid=grid, stop_loss_grid=grid, max_holding_steps=59,
        annualization_factor=1.0)
    p, best, d = res["params"], res["best_point"], res["diagnostics"]
    print(label)
    print(f"  fitted phi {p['phi']:.3f}, equilibrium {p['equilibrium']:.1f}, sigma {p['sigma']:.2f}")
    print(f"  best rule: take profit {best['profit_taking']}, stop loss {best['stop_loss']}, "
          f"Sharpe {best['sharpe']:.2f}, win rate {best['win_rate']:.0%}")
    print(f"  surface: best - median {d['peak_margin']:.2f}; no stable optimum: {d['no_stable_optimum']}")
```

```text
mean-reverting
  fitted phi 0.903, equilibrium 99.8, sigma 0.99
  best rule: take profit 2.0, stop loss 8.0, Sharpe 4.95, win rate 100%
  surface: best - median 4.17; no stable optimum: False
near random walk
  fitted phi 0.992, equilibrium 95.9, sigma 0.99
  best rule: take profit 0.5, stop loss 8.0, Sharpe 0.06, win rate 89%
  surface: best - median 0.08; no stable optimum: True
```

Both trades are entered at 97. In the first case the fitted equilibrium is 99.8, so the
position has about three points of expected reversion in its favour and the surface shows
where to collect it. In the second, 1,500 observations of a near random walk put the
"equilibrium" at 95.9 — *below* the entry, though the data were generated around 100 — and
no rule does better than a Sharpe ratio of 0.06. The search still names a best rule. The
diagnostic flags this weak surface for further scrutiny.

<figure>
<img class="dark:sl-hidden" src="/figures/ch13-otr-surface-light.svg" alt="Two five-by-five grids of Sharpe ratios over profit-taking width and stop-loss width. For the mean-reverting process the values rise toward wide stops and a profit target of 2 to 4, peaking at 5.0. For the near random walk every cell is between minus 0.1 and 0.1." />
<img class="light:sl-hidden" src="/figures/ch13-otr-surface-dark.svg" alt="Two five-by-five grids of Sharpe ratios over profit-taking width and stop-loss width. For the mean-reverting process the values rise toward wide stops and a profit target of 2 to 4, peaking at 5.0. For the near random walk every cell is between minus 0.1 and 0.1." />
<figcaption>Same grid, same shading scale. Wider stops perform better in this fitted mean-reverting example; the near-random-walk example has a much flatter surface.</figcaption>
</figure>

These results depend on the fitted forecast, noise, horizon and candidate grid. Mean
reversion does not guarantee recovery before the holding limit, and a wider stop permits
larger losses. Compare surfaces across independently simulated paths and plausible fits.

## Barrier crossings are observed prices

The evaluator exits at the sampled price that crosses a threshold, without interpolation
or a fill-price model. With two-point paths and barriers of 2, the following exits earn +5
and −10. A stop-loss threshold is therefore not a maximum loss.

```python
from openquant import synthetic_bt as sb

crossings = sb.evaluate_rule_on_paths(
    [[100.0, 105.0], [100.0, 90.0]], 2.0, 2.0, 1, 1.0)
assert crossings["mean_return"] == -2.5
assert crossings["win_rate"] == 0.5
constant_profit = sb.evaluate_rule_on_paths(
    [[100.0, 105.0], [100.0, 105.0]], 2.0, 2.0, 1, 1.0)
assert constant_profit["mean_return"] == 5.0
assert constant_profit["std_return"] == 0.0
assert constant_profit["sharpe"] == 0.0
```

## From Rust

```rust
use openquant::synthetic_backtesting::{
    calibrate_ou_params, evaluate_rule_on_paths, generate_ou_paths, OuProcessParams, TradingRule,
};

// An exact AR(1) recursion with no noise term would have sigma = 0 and be refused, so
// perturb it slightly: phi = 0.8 around 50.
let mut prices = vec![40.0];
for i in 1..400 {
    let wobble = if i % 2 == 0 { 0.05 } else { -0.05 };
    prices.push(0.2 * 50.0 + 0.8 * prices[i - 1] + wobble);
}
let fitted = calibrate_ou_params(&prices)?;
assert!((fitted.phi - 0.8).abs() < 0.01 && (fitted.equilibrium - 50.0).abs() < 0.01);
assert!(fitted.stationary);

// Simulation is reproducible for a seed, and every path starts at the entry price.
let params = OuProcessParams { phi: 0.9, intercept: 10.0, equilibrium: 100.0, sigma: 1.0, r_squared: 0.0, stationary: true };
let paths = generate_ou_paths(params, 97.0, 2_000, 60, 11)?;
assert_eq!(paths, generate_ou_paths(params, 97.0, 2_000, 60, 11)?);
assert!(paths.iter().all(|p| p.len() == 60 && p[0] == 97.0));

// Entered three points below equilibrium, a wide stop beats a tight one.
let wide = evaluate_rule_on_paths(&paths, TradingRule { profit_taking: 2.0, stop_loss: 8.0 }, 59, 1.0)?;
let tight = evaluate_rule_on_paths(&paths, TradingRule { profit_taking: 2.0, stop_loss: 0.5 }, 59, 1.0)?;
assert!(wide.sharpe > tight.sharpe && wide.win_rate > 0.95);
```

`generate_ou_paths` simulates from `intercept` and `phi`; `equilibrium` is carried for
reference, so when building `OuProcessParams` by hand keep them consistent:
`intercept = (1 − phi) × equilibrium`.

## What to watch for

- **Barriers are in price units, not multiples of volatility.** AFML's experiments set
  $\sigma=1$, so its grids read as multiples of $\sigma$. Here a grid of `[0.5, …, 8.0]` on an
  instrument with $\sigma=25$ is all noise-width barriers. Scale the grids by the fitted
  `sigma`.
- **Long only.** Every path is a long entry at `initial_price`. For a short, negate the
  series and the entry.
- **High simulated Sharpe can reflect concentrated exit PnL.** The best rule in the example
  has a rounded win rate of 100% and relatively little dispersion in its exit PnL. Barrier
  crossings can overshoot, so the profit target does not cap the payoff. The reported Sharpe
  does not include uncertainty about whether the real process follows the fitted model.
  If exit PnL has exactly zero dispersion, the implementation returns Sharpe **0**, including
  when all paths earn the same positive profit.
- **`annualization_factor` multiplies by its square root and ignores holding time.** A rule
  that exits in 3 steps and one that holds for 59 are scaled alike. Use 1.0 and read the
  Sharpe ratio as per trade, alongside `avg_holding_steps`.
- **The stability thresholds are heuristics.** `StabilityCriteria::default()` is
  $\lvert\varphi\rvert\ge0.97$, margin 0.20, surface deviation 0.10, best Sharpe 0.30, and
  `run_synthetic_otr_workflow` uses the same four when its threshold arguments are left as
  `None`. They are this library's numbers, not AFML's — the book shows the flattening but
  sets no thresholds. Look at the surface.
- **The model is the assumption.** Everything downstream is conditional on the price being
  fitted AR(1) with constant parameters. A high `r_squared` does not establish stationarity;
  `stationary` records $\lvert\varphi\rvert<1$, including 0.999. The path generator does not
  enforce that condition or use the flag to reject a manually supplied nonstationary model.
  When $\lvert1-\varphi\rvert\le10^{-12}$, calibration stores the response sample mean as
  `equilibrium`; that fallback is not a long-run equilibrium. Fit on several windows and
  check coefficient and equilibrium stability.
- **The forecast is an input.** `initial_price` relative to `equilibrium` decides the
  expected change of a stationary process at a fixed horizon. Entering at equilibrium
  removes that directional forecast; it does not make every finite simulated rule score
  exactly zero.

## Related modules

- [`labeling`](/modules/labeling/) — the triple barrier these two levels feed.
- [`backtesting-engine`](/modules/backtesting-engine/) — resampling history, the other
  alternative to a single walk-forward.
- [`backtest-statistics`](/modules/backtest-statistics/) — deflate whatever rule is finally
  traded.
- [`structural-breaks`](/modules/structural-breaks/) — test whether the fitted regime is
  still in force.

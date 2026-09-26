---
title: "Runbook: bet sizing from predicted probabilities"
description: "Does sizing meta-labeled bets by their predicted probability (AFML Snippets 10.1-10.3) beat a flat 0.5 filter, net of costs and deflated by every sizing choice tried? A pre-registered test of runbook 11's post hoc finding, on fresh SYNTHETIC planted-signal paths and no-signal controls."
status: authored
last_authored: '2026-09-25'
audience:
  - quant-dev
afml_chapter:
  - "3"
  - "7"
  - "10"
  - "14"
citation:
  - "López de Prado, M. (2018). Advances in Financial Machine Learning. Wiley. Chapter 10: §10.3 Bet Sizing from Predicted Probabilities (Snippet 10.1); §10.4 Averaging Active Bets (Snippet 10.2); §10.5 Size Discretization (Snippet 10.3); §10.6 Dynamic Bet Sizes and Limit Prices (Snippet 10.4). Chapter 3: §3.6 meta-labeling. Chapter 7: Snippet 7.3 (purged k-fold). Chapter 14: §14.7 (PSR, DSR)."
  - "Bailey, D. H. and López de Prado, M. (2014). The deflated Sharpe ratio: correcting for selection bias, backtest overfitting, and non-normality. Journal of Portfolio Management 40(5), 94–107."
examples:
  - "notebooks/python/13_bet_sizing_from_probabilities.ipynb"
sidebar:
  order: 5
---

:::caution[SYNTHETIC data]
Every number and figure on this page comes from simulated price paths with a **planted,
documented signal**, or from the committed SYNTHETIC sample `SYN_A`..`SYN_E` (seeded random walks,
see `DATA_SOURCES.md`). None of it describes a real market. Read it as a description of **what the
procedure does on data where the right answer is known**.
:::

**Notebook:** [`notebooks/python/13_bet_sizing_from_probabilities.ipynb`](https://github.com/Open-Quant/openquant/blob/main/notebooks/python/13_bet_sizing_from_probabilities.ipynb),
committed with its outputs and executed in CI by `just notebooks-run`.
Module background: [`bet_sizing`](/modules/bet-sizing/), [`evaluation`](/modules/evaluation/),
[`cross_validation`](/modules/cross-validation/). Builds on
[runbook 11](/runbooks/meta-labeling-triple-barrier/).

## Where the question comes from

[Runbook 11](/runbooks/meta-labeling-triple-barrier/) trained a meta-model to decide whether to
take a moving-average crossover's bets. After seeing its results, it noticed that sizing the bets
by the meta-model's probability (Snippet 10.1) had done best: Sharpe ratio 1.28 on its headline
path, and +0.36 over the unfiltered primary model in its Monte Carlo. It had not planned that
comparison, so it labelled the result a candidate.

This runbook tests the candidate properly:

- **Same pipeline.** Runbook 11's generator, primary model, meta-model and purged k-fold are
  copied unchanged.
- **Fresh paths.** Seed 50 instead of 47, so the test does not reuse the data the candidate was
  found on.
- **Hypotheses written first.** They and the trial grid were committed before the first run.

## Hypotheses (pre-registered)

The two rules share the same out-of-fold probability $p$ and both pay **5 bps per unit of one-way
turnover**:

- **Filter:** take the primary model's bet at full size when $p \ge 0.5$ (runbook 11's rule).
- **Sized:** Snippet 10.1 with $K = 2$, floored at zero, then the average of live bets
  (Snippet 10.2), rounded to steps of 0.1 (Snippet 10.3). This is
  `bet_sizing.bet_size_probability`.

| | Claim | Result |
|---|---|---|
| H1 | At $\kappa = 0.25$, sized has a higher net Sharpe ratio than filter (paired, 30 paths) | **supported**: 0.31 → 0.59, +0.27, $t$ = 7.0 |
| H2 | On a fresh headline path, sized's DSR over all 21 trials is at least 0.95 | **supported**: DSR 1.000 (on a favourable path) |
| H3a | At $\kappa = 0$, sized does not beat filter | **passes**: +0.01, $t$ = 0.4 |
| H3b | At $\kappa = 0$, the best-of-grid DSR is ≥ 0.95 on at most 5% of paths | **passes**: 0 of 30 |
| H3c | On `SYN_*`, neither sized nor the best of the grid has DSR ≥ 0.95 | **passes**: 0.004 and 0.075 |

## Method

- **Probabilities.** Runbook 11's pipeline, unchanged:
  - CUSUM events at 2σ;
  - triple-barrier meta-labels at ±1.5σ, with a 14-day vertical barrier;
  - three past-only features and average-uniqueness weights;
  - an L2 logistic meta-model whose probabilities are out-of-fold from
    `cross_validation.purged_kfold_splits` (5 folds, 1% embargo).
- **Sizing rules.** Every rule uses the same probabilities. With $s$ the primary model's side and
  $z = (p - 1/K)/\sqrt{p(1-p)}$:

| Rule | Size of one bet |
|---|---|
| `fixed` (primary) | $s$ |
| `binary` (filter) | $s \cdot 1\{p \ge 0.5\}$ |
| `prob` | $s \cdot 1\{p \ge 0.5\} \cdot (2\Phi(z) - 1)$ |
| `prob-signed` | $s \cdot (2\Phi(z) - 1)$: bets *against* the primary model when $p < 0.5$ |

- **From sizes to a position.**
  - `avg`: the position is the mean size of the live bets (`avg_active_signals`, Snippet 10.2).
  - `latest`: the position is the size of the most recently opened live bet.
  - Either way, the position is then rounded to the step (Snippet 10.3).
  - The notebook asserts that `bet_size_probability` equals `get_signal` → `avg_active_signals`
    → `discrete_signal`.
- **Returns.** A position set at the close of bar $t$ earns bar $t + 1$, less 5 bps per unit of
  turnover.
  - The *break-even cost* is mean gross return divided by mean turnover.
  - Maximum drawdown comes from `backtest_stats.drawdown_and_time_under_water`.
- **Trial grid: 21 configurations per path.** Every one is recorded before any DSR is read:
  - primary and filter;
  - `prob` with $K \in \{2, 3\}$ × step $\in \{0, 0.05, 0.1, 0.2\}$ × {`avg`, `latest`};
  - `prob-signed`;
  - the filter and the headline sized rule, walk-forward.
- **No-bet trials.** `TrialRegistry.record` rejects constant returns (#187; the fix, PR #190, was
  not merged when this ran). A trial that takes no position is counted with a Sharpe ratio of 0.

## Results (SYNTHETIC)

**Headline path** (fresh seed, 5,000 bars, 695 events, 21 trials, SR0 = 0.52 annualised):

| Rule | Bets | Net Sharpe (ann.) | Exposure | Turnover / day | Max DD | Break-even (bps) | DSR |
|---|---:|---:|---:|---:|---:|---:|---:|
| Primary | 695 | 0.30 | 0.73 | 0.155 | 0.49 | 17 | 0.165 |
| Filter ($p \ge 0.5$) | 402 | 0.78 | 0.43 | 0.100 | 0.24 | 38 | 0.870 |
| **Sized** ($K$ = 2, step 0.1, avg) | 402 | **1.32** | 0.09 | 0.021 | 0.03 | 62 | **1.000** |
| Oracle filter (not tradable) | 314 | 1.31 | 0.37 | 0.074 | 0.16 | 53 | — |

<figure>
<img class="dark:sl-hidden" src="/figures/notebooks/nb13-headline-light.svg" alt="Left: cumulative net log return on the fresh planted-signal path for the primary crossover, the flat 0.5 meta filter, the probability-sized rule and the oracle filter, each scaled to the filter's volatility. Right: the Snippet 10.1 bet-size curves for two and three classes and the stepped two-class curve, over a histogram of the meta-model's out-of-fold probabilities." />
<img class="light:sl-hidden" src="/figures/notebooks/nb13-headline-dark.svg" alt="Left: cumulative net log return on the fresh planted-signal path for the primary crossover, the flat 0.5 meta filter, the probability-sized rule and the oracle filter, each scaled to the filter's volatility. Right: the Snippet 10.1 bet-size curves for two and three classes and the stepped two-class curve, over a histogram of the meta-model's out-of-fold probabilities." />
<figcaption>Left: headline path, net of costs, every rule scaled to the filter's volatility (display only; the sized book is about a fifth of the filter's). Right: how Snippet 10.1 maps the meta-model's probabilities to sizes.</figcaption>
</figure>

**Monte Carlo, 30 fresh paths per strength** (mean annualised net Sharpe ratio; paired
difference, sized − filter):

| $\kappa_{\text{trend}}$ | Primary | Filter | Sized | Sized − filter | Oracle filter |
|---:|---:|---:|---:|---:|---:|
| 0 (no signal) | −0.19 | −0.20 | −0.18 | +0.01 ($t$ = 0.4) | −0.19 |
| 0.12 | 0.06 | 0.05 | −0.03 | **−0.08 ($t$ = −2.4)** | 0.38 |
| 0.25 | 0.25 | 0.31 | 0.59 | **+0.27 ($t$ = 7.0)** | 1.18 |

<figure>
<img class="dark:sl-hidden" src="/figures/notebooks/nb13-monte-carlo-light.svg" alt="Box plots over simulated paths at signal strengths 0, 0.12 and 0.25. Left: the net Sharpe ratio of the probability-sized rule minus that of the flat 0.5 filter. Right: the flat filter minus the unfiltered primary model." />
<img class="light:sl-hidden" src="/figures/notebooks/nb13-monte-carlo-dark.svg" alt="Box plots over simulated paths at signal strengths 0, 0.12 and 0.25. Left: the net Sharpe ratio of the probability-sized rule minus that of the flat 0.5 filter. Right: the flat filter minus the unfiltered primary model." />
<figcaption>Per-path gains. Sizing adds to filtering only where the filter itself has something to work with.</figcaption>
</figure>

<figure>
<img class="dark:sl-hidden" src="/figures/notebooks/nb13-cost-sensitivity-light.svg" alt="Mean net Sharpe ratio over simulated paths at the headline signal strength for the primary model, the flat 0.5 filter and the probability-sized rule, as the cost per unit of turnover rises from 0 to 40 basis points." />
<img class="light:sl-hidden" src="/figures/notebooks/nb13-cost-sensitivity-dark.svg" alt="Mean net Sharpe ratio over simulated paths at the headline signal strength for the primary model, the flat 0.5 filter and the probability-sized rule, as the cost per unit of turnover rises from 0 to 40 basis points." />
<figcaption>Cost sensitivity at κ = 0.25. The sized rule breaks even at about 30 bps per unit turnover, the filter at about 18.</figcaption>
</figure>

## What it means

- **The candidate replicates on fresh data.** Sizing beats filtering by +0.27 of Sharpe ratio at
  the headline strength. Walk-forward, the gain is +0.19 ($t$ = 4.3).
- **The headline path is a good draw.** Its DSR is 1.000, but only 11 of 30 paths at this
  strength give the sized rule a DSR of at least 0.95.
- **Sizing amplifies what the meta-model knows, including nothing.** At $\kappa = 0.12$ the
  meta-model has no detectable skill: the filter gains nothing over the primary model. There,
  sizing *loses* to filtering ($t$ = −2.4). Size by probability only after the meta-model has
  shown skill.
- **$K$ matters; step size and averaging did not, here.**
  - $K = 3$ on a binary model gives every $p \ge 0.5$ bet at least a 0.26 position, and scores
    0.44 against 0.59.
  - Steps from 0 to 0.2, averaged or latest-signal, all land within 0.585–0.612. Bets rarely
    overlap on this generator (mean uniqueness 0.87), so it cannot test averaging.
  - Betting against the primary model when $p < 0.5$ scores 0.47.
- **Capacity.** The sized book is small (exposure 0.075 against the filter's 0.55). Its turnover
  per unit of exposure is about the same as the filter's, and its break-even cost is higher (30
  against 18 bps). Matching the filter's risk needs about 5 to 7 times the position, which a
  nonlinear impact model could penalise. One synthetic asset with a linear cost cannot say.
- **Controls.** On the null paths no configuration, even the best of 21 picked after the fact,
  reaches DSR 0.95. On `SYN_*` every rule loses.

## Decision

**Promote probability sizing as the default sizing stage after a meta-model that has shown skill;
promote no strategy.**

- **What is promoted.**
  - Snippet 10.1 with $K$ equal to the number of classes, floored at zero (follow or stand aside).
  - Averaging live bets and a 0.1 step, through `bet_size_probability`. These are kept for the
    book's reasons; they were not shown to help here.
  - Every sizing choice registered as a trial.
- **Conditions.**
  - Gate on skill first: judge the filter against a null run, as runbook 11 requires.
  - Report the break-even cost and exposure next to the Sharpe ratio.
- **Not promoted.** $K = 3$ for a binary model, signed sizing, any claim about step size or
  averaging, and any market claim.

## Checks

The notebook asserts or prints all seven:

1. Features, sides, volatility and events are unchanged when the future after a cut is replaced.
2. Labels do not change when closes after $t_1$ change, and do change when closes inside the span
   change.
3. Purged folds have zero train/test overlaps, and no embargoed event is trained on.
4. A fold's fitted model is bit-identical when its test fold is replaced with noise.
5. For every `prob` configuration, positions up to each of 20 cuts are bit-identical when later
   probabilities and the ends of live bets are re-drawn. Re-drawing one live bet moves the
   position.
6. **Shift test.** Trading one bar early, which peeks, lifts the Sharpe ratio from 1.32 to 2.32.
   One bar late gives 1.20.
7. **Shuffle test.** Permuting the probabilities across events gives a mean Sharpe ratio of 0.18,
   with a 95th percentile of 0.50. None of 200 permutations reaches 1.32.

**Trial count.** 21 per path, all deflated by. Two trials on null Monte Carlo paths took no
position; they were counted with a Sharpe ratio of 0. The Monte Carlo paths, the cost sweep, the
oracle, the shift and shuffle tests and the Snippet 10.4 illustration are not trials.

**Illustration (not a hypothesis).** The notebook also sizes the last six headline events from a
price forecast, using `get_w`, `get_target_pos` and the #163-corrected `limit_price`
(Snippet 10.4). The forecast is the expected barrier price implied by $p$.

## Run it on your own data

The control study reads three environment variables; the planted-signal study does not depend on
the data source. Write the executed copy and figures outside the repository:

```bash
OPENQUANT_RUNBOOK_SOURCE=/path/to/daily_ohlcv.parquet \
OPENQUANT_RUNBOOK_SYMBOLS=ES,NQ,CL \
OPENQUANT_RUNBOOK_RANGE=2010-01-01:2024-12-31 \
OPENQUANT_RUNBOOK_REGISTRY=$HOME/.openquant/trials-bet-sizing.json \
OPENQUANT_FIGURE_DIR=/tmp/bet-sizing-figures \
  uv run --python .venv/bin/python python notebooks/python/scripts/execute_notebook_cells.py \
    notebooks/python/13_bet_sizing_from_probabilities.ipynb --out /tmp/13_on_my_data.ipynb
```

`OPENQUANT_RUNBOOK_REGISTRY` keeps the control's trial registry between sessions, so every sizing
configuration you try on your data counts toward its deflated Sharpe ratio. For a vendor API, set
`SOURCE` in the notebook's parameters cell to an `openquant.data.CallableSource`.
`OPENQUANT_RUNBOOK_REPS` sets the number of Monte Carlo paths per strength (default 30).

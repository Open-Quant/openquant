---
title: "Runbook: CPCV backtest with PSR and deflated Sharpe"
description: "Best-of-74 strategy selection judged by the naive PSR, the deflated Sharpe ratio, a walk-forward path and CPCV path distributions, on SYNTHETIC data with a no-signal control and a planted signal."
status: authored
last_authored: '2026-09-25'
audience:
  - quant-dev
afml_chapter:
  - "11"
  - "12"
  - "14"
citation:
  - "López de Prado, M. (2018). Advances in Financial Machine Learning. Wiley. Chapter 11: §11.6 Backtest Overfitting. Chapter 12: §12.2 The Walk-Forward Method; §12.4 The Combinatorial Purged Cross-Validation Method; §12.5 How CPCV Addresses Backtest Overfitting. Chapter 14: §14.7 The Probabilistic and Deflated Sharpe Ratios."
  - "Bailey, D. H. and López de Prado, M. (2014). The deflated Sharpe ratio: correcting for selection bias, backtest overfitting and non-normality. Journal of Portfolio Management 40(5), 94–107."
  - "Bailey, D. H. and López de Prado, M. (2012). The Sharpe ratio efficient frontier. Journal of Risk 15(2), 3–44."
examples:
  - "notebooks/python/12_cpcv_deflated_sharpe.ipynb"
sidebar:
  order: 4
---

:::caution[SYNTHETIC data]
Every number and figure on this page comes from simulated return series with a known answer, or
from the committed SYNTHETIC sample `SYN_A`..`SYN_E`. None of it describes a real market.
:::

**Notebook:** [`notebooks/python/12_cpcv_deflated_sharpe.ipynb`](https://github.com/Open-Quant/openquant/blob/main/notebooks/python/12_cpcv_deflated_sharpe.ipynb),
committed with its outputs and executed in CI. Run it with `just notebooks-run --only 12`; set
`OPENQUANT_RUNBOOK_RUNS=1000` for more datasets per strength (CI runs 200).

## The question

A researcher backtests all 74 configurations of a moving-average crossover family, reports the
best one, and asks whether its Sharpe ratio is evidence of skill. AFML §11.6 says a best-of-N
backtest mostly reports luck. §12.4 offers combinatorial purged cross-validation (CPCV) as a
distribution of out-of-sample paths instead of one. §14.7 offers the deflated Sharpe ratio (DSR),
which tests the best Sharpe ratio against SR0, the best that N skill-less trials would reach. This
runbook checks those claims on data where the right answer is known.

## Design

- **Data.** Each dataset is 10 years of daily returns with Student-$t_5$ noise and, optionally, a
  persistent latent drift (an AR(1) with coefficient 0.99). The drift's strength is the signal.
  - At strength 0 there is no drift: **the no-signal control**.
  - At strength 0.2 (**the planted signal**) the best configuration has a true annualised Sharpe
    ratio of 1.73. Each configuration's true Sharpe ratio is measured on one 250,000-bar path.
  - A power scan runs six strengths × 200 datasets.
- **Strategies.** A crossover of a fast moving average (2–34 bars) and a slow one (10–120 bars),
  long/short or long/flat: 74 configurations. Costs are 5 bps per unit of position change.
  - Every configuration is recorded in an
    [`evaluation.TrialRegistry`](/modules/evaluation/) before anything is selected.
  - DSR takes N and σ(SR) from the registry.
- **Validation.**
  - [`cross_validation.cpcv_splits`](/modules/cross-validation/) with N = 10 groups and k = 3 test
    groups gives 120 splits and φ[10, 3] = 36 paths.
  - In each split the procedure selects the configuration with the best training Sharpe ratio and
    trades it on the test groups.
  - [`backtesting_engine.run_cpcv`](/modules/backtesting-engine/) scores the splits and paths.
    `assemble_cpcv_paths` stitches them together.
  - For comparison there is one anchored walk-forward path: 2 years to start, re-selected yearly.
- **No lookahead, tested.**
  - Labels span one bar. A 126-bar embargo follows each test block, which covers the 120-bar
    lookback (AFML Snippet 7.3, as fixed in #170).
  - A test cell corrupts every test-period return in every split and rebuilds all 74 strategies.
    It asserts that every training sample's net return is bit-identical, and that the same check
    fails with the embargo off.

## Hypotheses and results (200 datasets per strength, SYNTHETIC)

| | Rule, fixed before the run | Result | |
| --- | --- | --- | --- |
| **H1a** | With no signal, naive PSR(0) > 0.95 on the in-sample best fires in more than 5% of datasets | 60 / 200 = 30% | supported |
| **H1b** | With no signal, DSR > 0.95 fires in no more than 5% | 0 / 200 | supported |
| **H2** | With the planted signal (true SR 1.73), DSR > 0.95 in more than half | 161 / 200 = 80.5% | supported |

Power scan, means over 200 datasets per strength (Sharpe ratios annualised):

| Best config's true SR | Naive PSR > 0.95 | DSR > 0.95 | In-sample best SR | True SR of the selected config | CPCV mean path SR | Walk-forward SR |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.00 (control) | 30% | 0% | 0.39 | −0.09 | −0.09 | −0.08 |
| 0.17 | 51% | 0% | 0.52 | 0.06 | 0.04 | 0.03 |
| 0.74 | 91% | 20% | 0.95 | 0.56 | 0.50 | 0.49 |
| 1.19 | 99.5% | 49% | 1.37 | 1.02 | 0.93 | 0.91 |
| 1.73 (planted) | 100% | 81% | 1.91 | 1.58 | 1.51 | 1.49 |
| 2.87 | 100% | 95% | 2.94 | 2.68 | 2.52 | 2.53 |

<figure>
<img class="dark:sl-hidden" src="/figures/notebooks/nb12-path-sharpe-light.svg" alt="Two panels. No-signal control: the CPCV path Sharpe ratios centre slightly below zero while the in-sample best-of-74 Sharpe ratios centre near 0.4. Planted signal: the path Sharpe ratios centre near 1.5 and the in-sample best near 1.9, with the best configuration's true Sharpe ratio of 1.73 marked." />
<img class="light:sl-hidden" src="/figures/notebooks/nb12-path-sharpe-dark.svg" alt="Two panels. No-signal control: the CPCV path Sharpe ratios centre slightly below zero while the in-sample best-of-74 Sharpe ratios centre near 0.4. Planted signal: the path Sharpe ratios centre near 1.5 and the in-sample best near 1.9, with the best configuration's true Sharpe ratio of 1.73 marked." />
<figcaption>CPCV path Sharpe ratios (36 paths × 200 datasets) against the in-sample best a naive backtest reports. The ticks are dataset 0's 36 paths. SYNTHETIC.</figcaption>
</figure>

<figure>
<img class="dark:sl-hidden" src="/figures/notebooks/nb12-dsr-vs-trials-light.svg" alt="Left: on the no-signal dataset 0, the naive PSR of the best configuration so far climbs from 0.43 to 0.85 as configurations are registered, while its DSR stays between 0.5 and 0.6. Right: the best Sharpe ratio so far and SR0 for the no-signal and planted datasets; on the no-signal dataset the two stay close, on the planted dataset the best stays far above SR0." />
<img class="light:sl-hidden" src="/figures/notebooks/nb12-dsr-vs-trials-dark.svg" alt="Left: on the no-signal dataset 0, the naive PSR of the best configuration so far climbs from 0.43 to 0.85 as configurations are registered, while its DSR stays between 0.5 and 0.6. Right: the best Sharpe ratio so far and SR0 for the no-signal and planted datasets; on the no-signal dataset the two stay close, on the planted dataset the best stays far above SR0." />
<figcaption>Deflation as the trial registry grows, on the two headline datasets.</figcaption>
</figure>

<figure>
<img class="dark:sl-hidden" src="/figures/notebooks/nb12-power-bias-light.svg" alt="Left: share of datasets in which the naive PSR and the DSR accept the best configuration, against its true Sharpe ratio; the naive PSR accepts 30 percent of no-signal datasets, the DSR none, and the DSR reaches 81 percent at a true Sharpe ratio of 1.73. Right: the in-sample best Sharpe ratio sits 0.3 to 0.5 above the true Sharpe ratio of the selected configuration, while the CPCV and walk-forward estimates sit close to it." />
<img class="light:sl-hidden" src="/figures/notebooks/nb12-power-bias-dark.svg" alt="Left: share of datasets in which the naive PSR and the DSR accept the best configuration, against its true Sharpe ratio; the naive PSR accepts 30 percent of no-signal datasets, the DSR none, and the DSR reaches 81 percent at a true Sharpe ratio of 1.73. Right: the in-sample best Sharpe ratio sits 0.3 to 0.5 above the true Sharpe ratio of the selected configuration, while the CPCV and walk-forward estimates sit close to it." />
<figcaption>False discoveries and power (left); what each kind of backtest reports against the truth (right).</figcaption>
</figure>

## What it shows

- **Deflation works, and on correlated trials it is conservative.**
  - The naive test finds skill in 30% of no-signal datasets. DSR finds it in none, where 5% would
    be allowed.
  - The 74 configurations are highly correlated, so the effective number of trials is far below
    the N that DSR uses.
  - The cost is power: 20% at a true Sharpe ratio of 0.74, and 49% at 1.19.
  - With a real signal, the trials' Sharpe ratios also differ for real reasons. That raises σ(SR),
    and so SR0 (0.93 against 0.27 on the headline datasets).
- **CPCV and walk-forward both remove the selection bias.**
  - Both land within about 0.1 of the true Sharpe ratio of what was selected. The in-sample best
    overstates it by 0.3 to 0.5.
  - CPCV's mean varies a little less across datasets than the single walk-forward path: s.d. 0.31
    against 0.38 on the controls.
- **The spread of CPCV paths is not a confidence interval.**
  - Within one dataset, the 36 paths have a standard deviation of about 0.21 at every strength.
  - The CPCV mean varies more than that *across* datasets: 0.30 to 0.77.
  - The paths reuse the same returns and differ only in which training set picked each group's
    configuration. Their spread measures how unstable the selection is, not the sampling error.
- **Headline datasets.**
  - **Control, dataset 0.** The in-sample best has SR 0.33, PSR(0) 0.85 and DSR 0.57. MinTRL is
    24.7 years, against 10 in the sample. The CPCV median is −0.33, with 3 of 36 paths positive,
    and the walk-forward is −0.48.
  - **Planted, dataset 0.** The in-sample best has SR 2.23 (true 1.69), and DSR is 1.00. The CPCV
    median is 1.80, with all 36 paths positive, and the walk-forward is 1.92.

This notebook also delivers the **CPCV Sharpe-distribution** figure (OQ-nbr.8) that the notebook
runner ([#45](https://github.com/Open-Quant/openquant/issues/45)) left pending until the CPCV
bindings existed.

## Decision

**Promote the evaluation protocol, not a strategy.** Every strategy-selection runbook should:

- register every configuration in a `TrialRegistry`;
- report DSR with the registry's N next to any PSR;
- report a CPCV (or walk-forward) Sharpe ratio instead of the in-sample best;
- state MinTRL.

Keep two caveats with it. A DSR below 0.95 on a correlated grid is weak evidence of *no* skill,
and CPCV path spread is not a confidence interval. An effective-number-of-trials estimate for the
registry is a worthwhile follow-up. The MA-crossover family itself has no edge on the data-layer
sample, which is random walks.

## Run it on your own data

The notebook's last section runs the same pipeline on bars read by
[`openquant.data.fetch`](/modules/data/). The environment variables are the same as
[runbook 09's](/runbooks/fracdiff-stationarity-memory/). Keep a trial registry across runs:

```bash
OPENQUANT_RUNBOOK_SOURCE=/path/to/daily_ohlcv.parquet \
OPENQUANT_RUNBOOK_SYMBOLS=ES,NQ,CL \
OPENQUANT_RUNBOOK_RANGE=2010-01-01:2024-12-31 \
OPENQUANT_RUNBOOK_REGISTRY_DIR=$HOME/.openquant/trials \
OPENQUANT_FIGURE_DIR=/tmp/cpcv-figures \
  uv run --python .venv/bin/python python notebooks/python/scripts/execute_notebook_cells.py \
    notebooks/python/12_cpcv_deflated_sharpe.ipynb --out /tmp/12_on_my_data.ipynb
```

That section uses N = 6, k = 2 and an embargo of at least 120 bars. The committed sample covers
only two years, so most of each training set is embargoed: use many years of data. Every grid you
try on the same data goes into the same registry, so the deflation counts every trial.

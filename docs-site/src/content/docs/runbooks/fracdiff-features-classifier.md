---
title: "Runbook: fracdiff features against returns in a classifier"
description: "Do fractionally differentiated features beat plain returns and integer differences as inputs of a purged-CV classifier, net of costs and deflated by every feature set tried? Runbook 09's open question, on SYNTHETIC paths with a planted signal of known strength and no-signal controls."
status: authored
last_authored: '2026-09-26'
audience:
  - quant-dev
afml_chapter:
  - "3"
  - "5"
  - "7"
  - "14"
citation:
  - "López de Prado, M. (2018). Advances in Financial Machine Learning. Wiley. Chapter 5: §5.5 Fixed-Width Window Fracdiff (Snippet 5.3), §5.6 Stationarity with Maximum Memory Preservation. Chapter 3: §3.2 fixed-time horizon labels. Chapter 7: Snippet 7.3 (purged k-fold with embargo). Chapter 14: §14.7 (PSR, DSR)."
  - "Bailey, D. H. and López de Prado, M. (2014). The deflated Sharpe ratio: correcting for selection bias, backtest overfitting, and non-normality. Journal of Portfolio Management 40(5), 94–107."
  - "MacKinnon, J. G. (2010). Critical values for cointegration tests. Queen's Economics Department Working Paper 1227."
examples:
  - "notebooks/python/14_fracdiff_features_classifier.ipynb"
sidebar:
  order: 6
---

:::caution[SYNTHETIC data]
Every number and figure on this page comes from simulated price paths with a **planted,
documented signal**, from the same generator with the signal switched off, or from the committed
SYNTHETIC sample `SYN_A`..`SYN_E` (seeded random walks, see `DATA_SOURCES.md`). None of it
describes a real market. Read it as a description of **what the procedure does on data where the
right answer is known**.
:::

**Notebook:** [`notebooks/python/14_fracdiff_features_classifier.ipynb`](https://github.com/Open-Quant/openquant/blob/main/notebooks/python/14_fracdiff_features_classifier.ipynb),
committed with its outputs and executed in CI by `just notebooks-run`.
Module background: [`fracdiff`](/modules/fracdiff/), [`evaluation`](/modules/evaluation/),
[`cross_validation`](/modules/cross-validation/). Follows
[runbook 09](/runbooks/fracdiff-stationarity-memory/).

## Where the question comes from

[Runbook 09](/runbooks/fracdiff-stationarity-memory/) swept the fractional differencing order $d$
and decided not to promote "the smallest $d$ passing ADF" as a default transform. Issue #49 also
asked whether FFD features beat plain returns in a purged-CV classifier, net of costs. On random
walks no feature can predict anything, so runbook 09 left that open. This runbook plants a signal
that only memory can find: the log price is a random walk plus a hidden AR(1) mispricing
($\phi = 0.97$, half-life about 23 bars), so the expected five-bar return is
$(\phi^5 - 1)\,m_t$. A classifier on the hidden $m_t$ (the oracle, not tradable) bounds what any
feature set can reach.

The generator's strength was calibrated on the oracle only, and every setting tried is disclosed
in the notebook. No candidate feature set had been run when the hypotheses were written.

## Hypotheses (pre-registered)

"FFD" is `ffd_dstar`: the FFD log price at $t, \dots, t-4$ (`thresh = 1e-3`), with $d$ chosen on
each training fold by runbook 09's ADF rule. "Returns" is the last five log returns. "Integer
differences" is the log price differenced over 1, 5, 20 and 60 bars. Each is the input of the same
L2-penalised logistic regression. Out-of-fold probabilities come from purged 5-fold CV with a 1%
embargo. Each test is a paired one-sided $t$-test over 30 paths at the headline strength
$s_m = 0.008$.

- **H1.** FFD's out-of-fold AUC is higher than returns'.
- **H2.** FFD's net Sharpe ratio (5 bps per unit turnover) is higher than returns'.
- **H3.** FFD's AUC is higher than the integer differences'.
- **H4.** On the headline path, FFD's deflated Sharpe ratio over all 8 registered trials is at
  least 0.95.
- **Controls.** At $s_m = 0$ the H1-H3 differences must not be significant ($|t| < 1.96$), and on
  the `SYN_*` sample no trial may reach a DSR of 0.95.
- **Decision rule.** Promote only if H1 and H2 hold and both controls pass.

## Method

- **Labels:** one event per bar, $y_t = 1$ if the five-bar forward log return is positive
  (fixed-horizon labels, AFML §3.2). The label spans $[t, t+5]$.
- **Feature sets (8 trials):** `returns`, `int_diff`, `level` ($d = 0$), FFD at fixed
  $d = 0.2, 0.4, 0.6, 0.8$, and `ffd_dstar`. Every one uses closes up to $t$ only.
- **Classifier:** L2-penalised logistic regression on features standardised with the training
  fold's statistics. It is written in numpy because scikit-learn is not a dependency, and it is
  the same model as [runbook 11](/runbooks/meta-labeling-triple-barrier/).
- **Bets:** $\text{sign}(p - 0.5)$ held for five bars, overlapping bets averaged. Net returns are
  $w_{t-1} r_t - 5\,\text{bps} \times |w_t - w_{t-1}|$, annualised by $\sqrt{252}$ because the
  bars are simulated business days.
- **Registry:** every trial is recorded in an `openquant.evaluation.TrialRegistry`. The headline
  path and the `SYN_*` control each have their own registry.

## Results (SYNTHETIC)

| Test (30 paths) | $s_m = 0.008$ (signal) | $s_m = 0$ (null) |
|---|---|---|
| H1: AUC, FFD − returns | +0.040, $t$ = 7.4 | +0.024, $t$ = 4.6 (**control fails**) |
| H2: net Sharpe, FFD − returns | +0.47, $t$ = 6.1 | +0.27, $t$ = 3.4 (**control fails**) |
| H3: AUC, FFD − integer differences | +0.029, $t$ = 4.7 | +0.023, $t$ = 2.9 (**control fails**) |

H4: on the headline path FFD's net Sharpe ratio is 0.60 (PSR 0.951) and its DSR over 8 trials is
0.779, so H4 is **rejected**. The `SYN_*` control passes as registered, with a highest DSR of 0.863.

<img class="dark:sl-hidden" src="/figures/notebooks/nb14-headline-equity-light.svg" alt="Cumulative net log return on the planted-signal path for the classifier on returns, on integer differences, on FFD features with d chosen per fold, and on the hidden mispricing (oracle). FFD ends near 0.8, returns and integer differences near 0.2, the oracle near 1.2." />
<img class="light:sl-hidden" src="/figures/notebooks/nb14-headline-equity-dark.svg" alt="Cumulative net log return on the planted-signal path for the classifier on returns, on integer differences, on FFD features with d chosen per fold, and on the hidden mispricing (oracle). FFD ends near 0.8, returns and integer differences near 0.2, the oracle near 1.2." />

<img class="dark:sl-hidden" src="/figures/notebooks/nb14-monte-carlo-light.svg" alt="Mean out-of-fold AUC and net Sharpe ratio over 30 paths for each feature set at three signal strengths. Even at zero signal, the level and low-d FFD sets sit above returns in both panels." />
<img class="light:sl-hidden" src="/figures/notebooks/nb14-monte-carlo-dark.svg" alt="Mean out-of-fold AUC and net Sharpe ratio over 30 paths for each feature set at three signal strengths. Even at zero signal, the level and low-d FFD sets sit above returns in both panels." />

## What it means

- **Most of the gap is there without a signal.** H1-H3 pass at the planted strength, but with no
  signal FFD is already ahead by most of the same margin. Over the null, the planted signal adds
  only +0.016 of AUC and +0.20 of Sharpe ratio.
- **Purged k-fold flatters memory and handicaps returns (post hoc, walk-forward).** On random
  walks, k-fold gives the `level` classifier a net Sharpe ratio of +0.22 and `ffd_dstar` +0.17. The
  model for a middle fold is fitted on bars after it too, and a feature that carries the price
  level lets it use where the path went. Purging removes label overlap, not this. Returns sit
  below chance (AUC 0.478, the base-rate effect of runbook 11). Walk-forward removes the Sharpe
  bias: FFD minus returns is −0.01 ($t$ = −0.1) at $s_m = 0$, and +0.35 ($t$ = 3.3) at the planted
  strength. The AUC gap survives the walk-forward null (+0.017, $t$ = 3.4), so pooled out-of-fold
  AUC is not a clean yardstick for memory features.
- **FFD is the level in all but name.** The ADF rule picks $d \approx 0.2$, where the truncated
  weights pass 37% of the log price through. `ffd_dstar` and `level` score the same: k-fold AUC
  0.530 and 0.529, walk-forward net Sharpe ratio 0.18 and 0.16.
- **On random walks, k-fold nearly made a false discovery.** On the `SYN_*` sample, `level` has a
  net Sharpe ratio of 1.85 and `ffd_d0.2` 1.44. Only the deflation keeps them below 0.95.

## Decision

**Do not promote.** FFD features with $d$ chosen per training fold do not become the default
classifier input over returns or over integer differences. The pre-registered null control
fails. Later runbooks should:

- evaluate memory-bearing features walk-forward, or with CPCV *and* a null run through the same
  splits;
- judge them on net returns against that null, not on pooled AUC;
- compare FFD with the level itself, not only with returns;
- pre-register the walk-forward comparison (FFD vs returns vs level, net Sharpe ratio, with the
  null) as the next test, on real data.

## Checks

The notebook asserts, as executed code:

1. **Features are causal.** No feature, and no FFD series for any $d$, changes when every close
   after a cut is replaced.
2. **Purge.** Train/test label overlaps are zero in every fold; unpurged k-fold has 5-10.
3. **The test fold is unseen.** Each fold's chosen $d$ and fitted model are bit-identical when the
   test fold's labels are replaced with noise.
4. **Shift test.** On a zero-signal path, the same features moved one bar into the future lift
   the AUC from 0.50 to 0.69 (returns) and from 0.51 to 0.68 (FFD at 0.4).

**Trial count.** 8 per registry, all pre-registered, all deflated by. The Monte Carlo paths, the
oracle and the post hoc walk-forward runs, which were made on the Monte Carlo paths only, are not
trials.

## Run it on your own data

The control study reads four environment variables; the planted-signal study does not depend on
the data source. Write the executed copy and figures outside the repository:

```bash
OPENQUANT_RUNBOOK_SOURCE=/path/to/daily_ohlcv.parquet \
OPENQUANT_RUNBOOK_SYMBOLS=ES,NQ,CL \
OPENQUANT_RUNBOOK_RANGE=2010-01-01:2024-12-31 \
OPENQUANT_RUNBOOK_REGISTRY=$HOME/.openquant/trials-fracdiff-classifier.json \
OPENQUANT_FIGURE_DIR=/tmp/fracdiff-classifier-figures \
  uv run --python .venv/bin/python python notebooks/python/scripts/execute_notebook_cells.py \
    notebooks/python/14_fracdiff_features_classifier.ipynb --out /tmp/14_on_my_data.ipynb
```

`OPENQUANT_RUNBOOK_REGISTRY` keeps the control's trial registry between sessions, so every feature
set you try on your data counts toward its deflated Sharpe ratio. `OPENQUANT_RUNBOOK_REPS` sets
the number of Monte Carlo paths per strength (default 30). On your data, read the control table
next to the zero-signal Monte Carlo: purged k-fold favours level-like features even when there is
nothing to find.

## Reproducibility

The notebook's last cell prints the `dataset_hash` of the sample (`sha256:b311c219…`), the seed
(209), a digest of the configuration and the trial counts (8 on the planted path, 8 on the
control, 90 Monte Carlo paths). The git commit, the package versions and the hash of the simulated
path go to stderr.

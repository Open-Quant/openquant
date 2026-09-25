---
title: "Runbook: fracdiff, stationarity versus memory"
description: "Sweeps the fractional differencing order d per series, picks the smallest d that passes ADF, and checks that rule against simulated series whose memory is known. Committed results are on SYNTHETIC data."
status: authored
last_authored: '2026-09-25'
audience:
  - quant-dev
afml_chapter:
  - "5"
citation:
  - "López de Prado, M. (2018). Advances in Financial Machine Learning. Wiley. Chapter 5: §5.4-5.6, Snippets 5.1-5.4, Fig. 5.5."
  - "MacKinnon, J. G. (2010). Critical values for cointegration tests. Queen's Economics Department Working Paper 1227."
sidebar:
  order: 1
---

:::caution[SYNTHETIC data]
Every number and figure on this page comes from synthetic series. The five symbols `SYN_A` to
`SYN_E` are seeded log-normal random walks from the committed sample (see `DATA_SOURCES.md`), and
the controls are simulated in the notebook. None of it describes a real market. Read the page as a
description of **what the procedure does on data whose properties are known**.
:::

**Notebook:** [`notebooks/python/09_fracdiff_stationarity_memory.ipynb`](https://github.com/Open-Quant/openquant/blob/main/notebooks/python/09_fracdiff_stationarity_memory.ipynb),
committed with its outputs and executed in CI by `just notebooks-run`. Module background:
[`fracdiff`](/modules/fracdiff/).

## Hypothesis

**H1.** For each series there is a minimum $d^* < 1$ at which the fixed-width-window fractionally
differenced (FFD) log price rejects a unit root (ADF, constant, one lag, 5%), and at that $d^*$
the FFD series still correlates at least **0.5** with the log price. This is the AFML Fig. 5.5
claim. The threshold of 0.5 was fixed before the first run.

Theory gives a known answer to check against. A series that is fractionally integrated of order
$\delta$, differenced $d$ times, is stationary only once $\delta - d < 0.5$. For a random walk
($\delta=1$) that means $d^* \approx 0.5$.

## Method

- **Grid:** $d = 0, 0.05, \dots, 1$ (21 values). FFD by `openquant.fracdiff.frac_diff_ffd` with
  `thresh = 1e-3` (Snippet 5.3).
- **ADF:** $\Delta y_t = \alpha + \gamma y_{t-1} + \phi\,\Delta y_{t-1} + e_t$, the statistic is
  $t_\gamma$, and the 5% critical value comes from MacKinnon's (2010) finite-sample surface. This
  is Snippet 5.4's `adfuller(maxlag=1, regression="c", autolag=None)`, written in numpy because
  `statsmodels` is not a dependency. It matches `statsmodels` to 1e-12.
- **Memory:** the Pearson correlation between the FFD series and the log price.
- **Controls:** 200 simulated paths each of fractionally integrated noise with
  $\delta = 0.4, 0.7, 1.0$, 520 bars like the sample, run through the same sweep.
- **Checks:** $d^*$ chosen on the first half and tested on the second; a shift test showing that
  no FFD output changes when later bars are replaced.

## Results

| Symbol | $d^*$ | corr at $d^*$ | corr at $d=1$ (returns) | FFD weight sum at $d^*$ | H1 |
|---|---|---|---|---|---|
| `SYN_A` | 0.25 | 0.918 | 0.053 | 0.28 | supported |
| `SYN_B` | 0.30 | 0.929 | 0.040 | 0.22 | supported |
| `SYN_C` | 0.25 | 0.916 | 0.104 | 0.28 | supported |
| `SYN_D` | 0.00 | 1.000 | 0.131 | 1.00 | supported |
| `SYN_E` | 0.20 | 0.935 | 0.082 | 0.37 | supported |

<figure>
<img class="dark:sl-hidden" src="/figures/notebooks/nb09-adf-corr-sample-light.svg" alt="ADF statistic and correlation with the log price against d for each synthetic sample symbol. The ADF curves cross the 5% critical value between d = 0.2 and 0.3, where the correlation is still above 0.9." />
<img class="light:sl-hidden" src="/figures/notebooks/nb09-adf-corr-sample-dark.svg" alt="ADF statistic and correlation with the log price against d for each synthetic sample symbol. The ADF curves cross the 5% critical value between d = 0.2 and 0.3, where the correlation is still above 0.9." />
<figcaption>The Fig. 5.5 shape on the synthetic sample. The diamonds mark each symbol's d*.</figcaption>
</figure>

The rule is then run on the controls, where the right answer is known:

| $\delta$ | theory $d^*$ | median $d^*$ | $d^*$ below theory | same, ADF with 5 lags |
|---|---|---|---|---|
| 0.4 | 0.0 | 0.0 | 0% of paths | 0% |
| 0.7 | 0.2 | 0.0 | 100% | 92% |
| 1.0 | 0.5 | 0.2 | 99.5% | 88% |

<figure>
<img class="dark:sl-hidden" src="/figures/notebooks/nb09-controls-light.svg" alt="Left: median ADF statistic against d for simulated series of integration order 0.4, 0.7 and 1.0. Right: the distribution of the chosen d* for each order, with the theoretical value dotted; for orders 0.7 and 1.0 the chosen d* sits to the left of theory." />
<img class="light:sl-hidden" src="/figures/notebooks/nb09-controls-dark.svg" alt="Left: median ADF statistic against d for simulated series of integration order 0.4, 0.7 and 1.0. Right: the distribution of the chosen d* for each order, with the theoretical value dotted; for orders 0.7 and 1.0 the chosen d* sits to the left of theory." />
<figcaption>On random walks the sweep stops near d = 0.2, not at 0.5.</figcaption>
</figure>

<figure>
<img class="dark:sl-hidden" src="/figures/notebooks/nb09-series-at-dstar-light.svg" alt="Three stacked panels for SYN_A: the log price, the FFD series at d* = 0.25, which still follows the level's slow swings, and the log returns." />
<img class="light:sl-hidden" src="/figures/notebooks/nb09-series-at-dstar-dark.svg" alt="Three stacked panels for SYN_A: the log price, the FFD series at d* = 0.25, which still follows the level's slow swings, and the log returns." />
<figcaption>SYN_A at d = 0, d*, and 1. The FFD series still carries the level's slow swings.</figcaption>
</figure>

## What it shows

- **H1 passes as written, on 5 of 5 symbols.** `SYN_D` passes at $d=0$, which for a random walk
  is a 5% false rejection.
- **Passing ADF does not mean the series is stationary.** ADF's null hypothesis is a unit root. A
  series integrated of order between 0.5 and 1 has no unit root, so ADF rejects, but it is still
  not stationary. On random walks the rule therefore picks $d \approx 0.2$-$0.3$, where theory
  needs more than 0.5, on 199 of 200 paths.
- **Part of the retained memory is the level itself.** At `thresh = 1e-3` the truncated weights at
  $d^*$ sum to 0.22-0.37 instead of 0, so a fifth to a third of the log price passes through the
  filter unchanged.
- **$d^*$ is noisy.** Compared with the full sample, a sweep on the second half alone moves $d^*$
  for three of the five symbols, by up to 0.25.
- **No look-ahead.** No FFD output changes when later bars are replaced.

## Promotion decision

**Do not promote.** Later runbooks should not use "per-symbol $d^*$ = the smallest $d$ passing ADF"
as their default feature transform. Keep `frac_diff_ffd` as a candidate feature, with $d$ chosen
on the training span and counted as a trial. If $d$ is chosen by a test, pair ADF with a test whose
null is stationarity (KPSS or similar). Report the FFD weight sum next to any FFD feature. Issue
#49's other question, whether FFD features beat returns in a purged-CV classifier net of costs,
needs real data or a planted signal. This runbook does not test it.

## Run it on your own data

The notebook reads three environment variables. Write the executed copy and its figures outside
the repository:

```bash
OPENQUANT_RUNBOOK_SOURCE=/path/to/daily_ohlcv.parquet \
OPENQUANT_RUNBOOK_SYMBOLS=ES,NQ,CL \
OPENQUANT_RUNBOOK_RANGE=2015-01-01:2024-12-31 \
OPENQUANT_FIGURE_DIR=/tmp/fracdiff-figures \
  uv run --python .venv/bin/python python notebooks/python/scripts/execute_notebook_cells.py \
    notebooks/python/09_fracdiff_stationarity_memory.ipynb --out /tmp/09_on_my_data.ipynb
```

For a vendor API, set `SOURCE` in the notebook's parameters cell to a `CallableSource` around
your own fetch function. Read your $d^*$ table next to the controls table: the controls do not
depend on your data, and they show how far below the stationarity boundary the ADF rule tends to
land.

## Reproducibility

The notebook's last cell prints the package version, the `dataset_hash` of the sample
(`sha256:b311c219…`), the hash of the simulated controls, the seed (49) and the trial counts: 105
(symbol, $d$) evaluations on the data, 210 more for the half-sample re-sweeps, and 25,200 on the
simulations, which describe the method and select nothing. It also prints the git commit, to
stderr. The trial registry (`openquant.evaluation`) is not on this branch yet, so the counts are
recorded in the notebook's run manifest.

---
title: "Runbook: HRP vs IVP and CLA out of sample"
description: "AFML §16.6's Monte Carlo, re-run with the openquant bindings: out-of-sample variance of HRP, inverse-variance and CLA minimum-variance portfolios on SYNTHETIC data."
status: authored
last_authored: '2026-09-25'
audience:
  - quant-dev
afml_chapter:
  - "16"
citation:
  - "López de Prado, M. (2018). Advances in Financial Machine Learning. Wiley. Chapter 16: §16.6 Out-of-Sample Monte Carlo Simulations; Snippets 16.4 and 16.5."
  - "López de Prado, M. (2016). Building diversified portfolios that outperform out of sample. Journal of Portfolio Management 42(4), 59–69."
examples:
  - "notebooks/python/10_hrp_vs_ivp_cla_oos.ipynb"
sidebar:
  order: 2
---

:::caution[SYNTHETIC data]
Every number and figure on this page comes from simulated returns (AFML Snippet 16.4) or from the
committed SYNTHETIC sample `SYN_A`..`SYN_E`. None of it describes a real market.
:::

**Notebook:** [`notebooks/python/10_hrp_vs_ivp_cla_oos.ipynb`](https://github.com/Open-Quant/openquant/blob/main/notebooks/python/10_hrp_vs_ivp_cla_oos.ipynb),
committed with its outputs and executed in CI. Run it with
`just notebooks-run --only 10`; set `OPENQUANT_RUNBOOK_RUNS=10000` for the book's run count (CI runs
2,000).

## Hypothesis

Under the Snippet 16.4 process (10 assets: 5 independent series plus 5 noisy copies, with common and
specific shocks that fall out of sample), HRP's out-of-sample portfolio variance is:

- **H1:** lower than CLA's minimum-variance portfolio, and
- **H2:** comparable to or lower than inverse-variance weighting (IVP).

Each is tested on the per-run paired log variance ratio, one-sided at 5%.

## Method

As in Snippet 16.5: at each rebalance, every 22 observations from row 260, estimate the covariance on
the previous 260 rows only, compute weights with
[`hrp.allocate_hrp`](/modules/hrp/), [`cla.allocate_cla`](/modules/cla/)
(`solution="min_volatility"`), IVP ($w_i\propto 1/\sigma_i^2$) and equal weight, and hold them for the
next 22 rows. Each run yields a 260-row out-of-sample series per method. A test cell checks that
replacing every row from the rebalance date on leaves the weights bit-identical. Costs: 10 bps per unit
of one-way turnover.

## Result (2,000 runs, SYNTHETIC)

| Method | Mean OOS variance ×10⁴ | Ratio to HRP | Runs where HRP is lower | Effective N (of 10) | Turnover per rebalance |
| --- | ---: | ---: | ---: | ---: | ---: |
| HRP | 3.90 | 1.00 | — | 7.25 | 0.22 |
| IVP | 5.01 | 1.28 | 76% | 8.26 | 0.07 |
| CLA (min. variance) | 5.16 | 1.32 | 62% | 5.36 | 0.19 |
| Equal weight | 8.42 | 2.16 | 87% | 10.00 | 0.00 |

Both hypotheses are supported ($t$ = 10.6 against CLA, 31.5 against IVP). A 10,000-run local run gives
ratios of 1.34 (CLA) and 1.30 (IVP).

<figure>
<img class="dark:sl-hidden" src="/figures/notebooks/nb10-oos-variance-light.svg" alt="Left: histograms of per-run out-of-sample variance for HRP, IVP and CLA on a log scale, with equal weight's median as a vertical line; HRP's distribution sits furthest left. Right: histograms of the per-run log2 ratio of IVP and CLA variance to HRP variance, both centred above zero." />
<img class="light:sl-hidden" src="/figures/notebooks/nb10-oos-variance-dark.svg" alt="Left: histograms of per-run out-of-sample variance for HRP, IVP and CLA on a log scale, with equal weight's median as a vertical line; HRP's distribution sits furthest left. Right: histograms of the per-run log2 ratio of IVP and CLA variance to HRP variance, both centred above zero." />
<figcaption>Per-run out-of-sample variance, and the paired ratio to HRP. SYNTHETIC Monte Carlo.</figcaption>
</figure>

<figure>
<img class="dark:sl-hidden" src="/figures/notebooks/nb10-weight-concentration-light.svg" alt="Box plots across runs of the effective number of assets and the largest weight. CLA is the most concentrated, IVP the least apart from equal weight, and HRP lies between them." />
<img class="light:sl-hidden" src="/figures/notebooks/nb10-weight-concentration-dark.svg" alt="Box plots across runs of the effective number of assets and the largest weight. CLA is the most concentrated, IVP the least apart from equal weight, and HRP lies between them." />
<figcaption>Weight concentration, averaged over the 12 rebalances of each run.</figcaption>
</figure>

## Against the book

Quoting AFML §16.6 from memory (check the book before citing): CLA's out-of-sample variance is about
72% higher than HRP's and IVP's about 38% higher. The direction replicates; the size does not. Here the
ratios are 1.32 and 1.28 on mean per-run variance. On the statistic Snippet 16.5 appears to print (the
variance across runs of each run's standard deviation), CLA is 1.36× HRP but IVP is 0.90×. The notebook
lists untested candidate reasons, chiefly that Snippet 16.1 clusters on distances between rows of the
distance matrix while `openquant.hrp` clusters on the pairwise distances.

Two findings beyond the book's claim: HRP turns over more than CLA here (0.22 vs 0.19 per rebalance),
because single linkage on near-duplicate pairs can reorder the leaves between windows; and HCAA with
`"minimum_variance"` is left out because it returns HRP's weights exactly
([#108](https://github.com/Open-Quant/openquant/issues/108)).

## Decision

**Replicated on synthetic data; not promoted as a market claim.** Whether HRP lowers variance on a real
universe is still open and needs this procedure run on real data.

## Run it on your own data

The notebook's last study reads bars through [`openquant.data.fetch`](/modules/data/). Set these in its
setup cell:

```python doc-check=skip
from openquant.data import LocalFileSource

SOURCE = LocalFileSource("my_daily_bars.parquet")  # ts, symbol, close, ... one row per symbol and day
SYMBOLS = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]
START, END = "2015-01-02", "2024-12-31"
```

A vendor client can be wrapped in `openquant.data.CallableSource` instead (see `DATA_SOURCES.md`). One
path over a handful of assets is an illustration, not evidence: a real study needs tens of assets, many
years, and its trial count recorded.

<figure>
<img class="dark:sl-hidden" src="/figures/notebooks/nb10-fetch-illustration-light.svg" alt="Out-of-sample growth of one unit in 2023 for HRP, IVP, CLA and equal weight on the five synthetic sample symbols." />
<img class="light:sl-hidden" src="/figures/notebooks/nb10-fetch-illustration-dark.svg" alt="Out-of-sample growth of one unit in 2023 for HRP, IVP, CLA and equal weight on the five synthetic sample symbols." />
<figcaption>The same rolling procedure on the SYNTHETIC <code>SYN_A</code>..<code>SYN_E</code> sample: one path, no inference.</figcaption>
</figure>

---
title: "evaluation"
description: "Python research evaluation: probabilistic and deflated Sharpe ratios and minimum track record from a returns series, a trial registry that persists across runs, meta-labeling overlay metrics and the probability of strategy failure."
status: authored
last_authored: '2026-09-25'
audience:
  - quant-dev
  - platform-engineering
module: "evaluation"
api_surface: "python-only"
afml_chapter:
  - "3"
  - "14"
  - "15"
citation:
  - "López de Prado, M. (2018). Advances in Financial Machine Learning. Wiley. Chapter 14: §14.7.1 The Sharpe Ratio; §14.7.2 The Probabilistic Sharpe Ratio; §14.7.3 The Deflated Sharpe Ratio. Chapter 3: §3.6 Meta-Labeling; §3.7 How to Use Meta-Labeling. Chapter 15: §15.4 The Probability of Strategy Failure (Snippets 15.4–15.5)."
  - "Bailey, D. H. and López de Prado, M. (2012). The Sharpe ratio efficient frontier. Journal of Risk 15(2), 3–44."
  - "Bailey, D. H. and López de Prado, M. (2014). The deflated Sharpe ratio: correcting for selection bias, backtest overfitting, and non-normality. Journal of Portfolio Management 40(5), 94–107."
python_api:
  - "evaluation.return_moments"
  - "evaluation.probabilistic_sharpe_ratio"
  - "evaluation.deflated_sharpe_ratio"
  - "evaluation.expected_max_sharpe"
  - "evaluation.minimum_track_record_length"
  - "evaluation.TrialRegistry"
  - "evaluation.config_hash"
  - "evaluation.meta_label_metrics"
  - "evaluation.strategy_failure_probability"
sidebar:
  badge: Module
---

The question at the end of a backtest is not "what was the Sharpe ratio" but "is this Sharpe
ratio evidence of anything". `openquant.evaluation` answers it from the returns series you
already have. It is a thin Python layer: the statistics are the Rust implementations in
[`backtest_statistics`](/modules/backtest-statistics/) and
[`strategy_risk`](/modules/strategy-risk/), which take pre-computed moments; this module
computes the moments, validates the input, and adds the one thing those functions cannot
know — how many configurations you tried before this one.

## From a returns series

`return_moments(returns)` gives the four numbers everything else is built from: the number
of observations $T$, the Sharpe ratio $\widehat{SR}$, skewness $\hat\gamma_3$ and kurtosis
$\hat\gamma_4$. Its conventions are those of Bailey and López de Prado, and the ones the Rust
functions silently assume:

- $\widehat{SR}$ is **per period** — mean over the sample standard deviation, not annualised;
- $\hat\gamma_3 = m_3/m_2^{3/2}$ and $\hat\gamma_4 = m_4/m_2^2$ from population central moments,
  so $\hat\gamma_4$ is **raw** kurtosis, 3 for a normal distribution.

From these, `probabilistic_sharpe_ratio(returns, benchmark_sr)` is the confidence that the
true Sharpe ratio exceeds a benchmark, and `minimum_track_record_length(returns,
benchmark_sr, alpha)` the number of observations needed for that confidence to reach
$1-\alpha$:

$$
\widehat{PSR}(SR^*) = Z\!\left[\frac{(\widehat{SR}-SR^*)\sqrt{T-1}}
{\sqrt{1-\hat\gamma_3\widehat{SR}+\frac{\hat\gamma_4-1}{4}\widehat{SR}^2}}\right],
\qquad
\mathrm{MinTRL} = 1+\Bigl[1-\hat\gamma_3\widehat{SR}+\tfrac{\hat\gamma_4-1}{4}\widehat{SR}^2\Bigr]
\left(\frac{Z^{-1}(1-\alpha)}{\widehat{SR}-SR^*}\right)^{2}
$$

with $Z$ the standard normal CDF. Unlike the Rust function, `minimum_track_record_length`
returns `math.inf` when $\widehat{SR} \le SR^*$: no track record of that quality is long
enough.

## Deflating by the trials you actually ran

The deflated Sharpe ratio is the PSR measured against $SR_0$, the Sharpe ratio the luckiest of
$N$ skill-less trials would be expected to show (Bailey and López de Prado, 2014):

$$
SR_0 = \sigma_{SR}\Bigl[(1-\gamma)\,Z^{-1}\!\bigl(1-\tfrac1N\bigr)+\gamma\,Z^{-1}\!\bigl(1-\tfrac{1}{Ne}\bigr)\Bigr],
\qquad \mathrm{DSR} = \widehat{PSR}(SR_0)
$$

where $\sigma_{SR}$ is the standard deviation of the trials' Sharpe ratios and $\gamma \approx
0.5772$ the Euler–Mascheroni constant. The formula is easy; $N$ is the hard part. Nobody
remembers how many variants they tried last week, and every forgotten trial makes the DSR
too kind.

`TrialRegistry(path)` keeps the count. Each `record(config, returns)` stores the
configuration, the SHA-256 of its canonical JSON, a UTC timestamp and the trial's Sharpe
ratio, $T$, skewness and kurtosis in a JSON file at `path`. The file is re-read before every
write and replaced atomically, so a registry opened tomorrow, in another notebook or another
process, continues the same count. `registry.deflated_sharpe_ratio(returns)` then deflates
with $N$ = the number of registered trials and $\sigma_{SR}$ = the population standard
deviation of their Sharpe ratios.

A configuration whose returns are constant — typically all zeros, because its filter took no
bet — has no Sharpe ratio, but it was still tried. `record` stores it with Sharpe 0 (what a
strategy with no excess return earns), skewness 0 and kurtosis 3, instead of raising. It
counts towards $N$ and its 0 enters $\sigma_{SR}$, like any other trial. A caller that caught
an error and skipped such configurations would undercount $N$ and deflate too weakly.

```python
import random
import tempfile
from pathlib import Path

from openquant import evaluation as ev

# Three years of daily returns on an asset that is a pure random walk: no rule can have skill.
rng = random.Random(24)
market = [rng.gauss(0.0, 0.01) for _ in range(756)]


def backtest(lookback):
    """Long when the trailing `lookback`-day return is positive, short otherwise."""
    return [
        (1.0 if sum(market[t - lookback:t]) > 0 else -1.0) * market[t]
        for t in range(100, len(market))
    ]


registry_path = Path(tempfile.mkdtemp()) / "trials.json"

# Session 1: try fifty lookbacks, recording every one.
registry = ev.TrialRegistry(registry_path)
for lookback in range(2, 101, 2):
    registry.record({"rule": "momentum", "lookback": lookback}, backtest(lookback))

# Session 2, another day: the registry file still knows about all fifty.
registry = ev.TrialRegistry(registry_path)
best = max(registry.trials, key=lambda t: t.sharpe)
returns = backtest(best.config["lookback"])

print(f"trials recorded: {registry.n_trials}   best lookback: {best.config['lookback']}")
print(f"best Sharpe: {best.sharpe:.4f} per day, {best.sharpe * 252 ** 0.5:.2f} annualised")
print(f"PSR(0): {ev.probabilistic_sharpe_ratio(returns):.3f}")
print(f"minimum track record at 95%: {ev.minimum_track_record_length(returns):.0f} days")
print(f"trial Sharpe dispersion: {registry.sharpe_std():.4f}   luckiest of {registry.n_trials} expected at: "
      f"{registry.expected_max_sharpe():.4f} per day")
print(f"DSR: {registry.deflated_sharpe_ratio(returns):.3f}")
```

```text
trials recorded: 50   best lookback: 8
best Sharpe: 0.0748 per day, 1.19 annualised
PSR(0): 0.972
minimum track record at 95%: 488 days
trial Sharpe dispersion: 0.0352   luckiest of 50 expected at: 0.0801 per day
DSR: 0.446
```

Taken alone, the eight-day rule passes every single-strategy test: an annualised Sharpe ratio
of 1.19, 97% confidence that it beats zero, and a minimum track record of 488 days against
the 656 it has. Only the registry knows it was the best of fifty. The luckiest of fifty
skill-less rules with this spread of results would be expected to reach 0.080 per day, more
than the 0.075 this one shows, and the deflated Sharpe ratio falls to 0.45 — a coin flip, which
is the right answer for a random walk.

The same deflation is available without a registry:
`deflated_sharpe_ratio(returns, trial_sharpes)` with every trial's per-period Sharpe ratio,
or `deflated_sharpe_ratio(returns, n_trials=N, sharpe_std=σ)`; `expected_max_sharpe(N, σ)`
returns $SR_0$ alone.

## Meta-labeling and the probability of failure

Meta-labeling (AFML §3.6) puts a secondary model on top of a primary one: the primary model
picks the side, the secondary decides whether to take the bet. The meta-label of a bet is 1
if the primary model's bet made money. `meta_label_metrics(meta_labels, meta_predictions,
threshold)` scores the overlay — bets taken when the prediction is at least `threshold` —
against the primary model alone, which takes every bet and so has recall 1 and precision
equal to its hit rate. `strategy_failure_probability` then asks the Chapter 15 question of
either set of bets: the probability that, over the investor's horizon, the precision falls
below what the target Sharpe ratio requires (see [`strategy_risk`](/modules/strategy-risk/)
for the method; `failure_probability` is its `empirical_failure_probability`).

```python
import random

from openquant import evaluation as ev

# Two years of a primary model's bets: 500 of them, 55% profitable, +1% or -1% each.
# A secondary model scores each bet; its score is informative but noisy.
rng = random.Random(8)
won = [1 if rng.random() < 0.55 else 0 for _ in range(500)]
score = [min(1.0, max(0.0, rng.gauss(0.58 if w else 0.45, 0.15))) for w in won]
outcome = [0.01 if w else -0.01 for w in won]

m = ev.meta_label_metrics(won, score, threshold=0.5)
print(f"primary alone: precision {m['primary_precision']:.3f}  recall {m['primary_recall']:.3f}  "
      f"F1 {m['primary_f1']:.3f}")
print(f"with overlay:  precision {m['precision']:.3f}  recall {m['recall']:.3f}  F1 {m['f1']:.3f}  "
      f"({m['true_positives'] + m['false_positives']} of 500 bets taken)")

taken = [r for r, s in zip(outcome, score) if s >= 0.5]
for name, bets in (("primary alone", outcome), ("with overlay", taken)):
    risk = ev.strategy_failure_probability(bets, years_elapsed=2.0, target_sharpe=1.0,
                                           investor_horizon_years=1.0, seed=1)
    print(f"{name}: P(annualised Sharpe < 1 over the next year) = {risk['failure_probability']:.2f}")
```

```text
primary alone: precision 0.526  recall 1.000  F1 0.689
with overlay:  precision 0.693  recall 0.688  F1 0.691  (261 of 500 bets taken)
primary alone: P(annualised Sharpe < 1 over the next year) = 0.56
with overlay: P(annualised Sharpe < 1 over the next year) = 0.00
```

The F1 score barely moves, and that is the lesson of the example rather than a flaw in it:
the overlay trades recall for precision, and F1 weighs the two equally. What the overlay buys
is visible in the last two lines. With symmetric payouts the Sharpe ratio is driven by
precision, so filtering out the weakest half of the bets turns a strategy that misses its
target more often than not into one that almost never does, despite taking half as many bets.

## API

| Function | Returns |
| --- | --- |
| `return_moments(returns)` | `ReturnMoments(n_obs, mean, std, sharpe, skewness, kurtosis)` |
| `probabilistic_sharpe_ratio(returns, benchmark_sr=0.0)` | $\widehat{PSR}(SR^*)$ |
| `minimum_track_record_length(returns, benchmark_sr=0.0, alpha=0.05)` | MinTRL in observations; `inf` at or below the benchmark |
| `expected_max_sharpe(n_trials, sharpe_std)` | $SR_0$ |
| `deflated_sharpe_ratio(returns, trial_sharpes)` or `(returns, n_trials=, sharpe_std=)` | DSR |
| `TrialRegistry(path)` | registry with `record(config, returns)`, `trials`, `n_trials`, `reload()`, `sharpe_std()`, `expected_max_sharpe()`, `deflated_sharpe_ratio(returns)` |
| `config_hash(config)` | SHA-256 of the configuration's sorted-key JSON |
| `meta_label_metrics(meta_labels, meta_predictions, threshold=0.5)` | precision, recall, F1, accuracy, confusion counts, and `primary_*` for the primary model alone |
| `strategy_failure_probability(bet_outcomes, years_elapsed, target_sharpe, investor_horizon_years, ...)` | the `strategy_risk` report plus `failure_probability` |

Returns must hold at least three finite values and must not be constant, except in
`TrialRegistry.record`, which records constant returns as Sharpe 0 (above). `n_trials` must
be an integer of at least 2, `alpha` must lie in $(0, 1)$, and a registry must hold two
trials before it can deflate. Violations raise `ValueError` (or `TypeError` for non-numeric input) before anything
is computed.

## What to watch for

- **Record every trial, including the one you evaluate**, and record it when you run it, not
  when it looks promising. A registry that only holds the survivors is the selection bias DSR
  exists to remove.
- **Re-recording a configuration replaces its entry.** Identity is the hash of the config's
  JSON, so re-running a notebook does not inflate $N$ — but two genuinely different
  experiments with the same config (a different data window, say) must say so in the config,
  or the second overwrites the first.
- **Correlated trials overstate $N$ and understate $\sigma_{SR}$.** Fifty lookbacks of one
  momentum rule are not fifty independent ideas. The formula assumes independent trials;
  AFML suggests clustering them ([`onc`](/modules/onc/)) and counting clusters. In the example
  the correction still bites because the trials disagree enough, but with near-identical
  variants $\sigma_{SR}$ collapses and so does $SR_0$.
- **Every Sharpe ratio in one registry must be per period at one frequency.** Mixing daily and
  weekly trials makes $\sigma_{SR}$ meaningless. Use one registry per frequency and dataset.
- **The registry is a file, not a database.** Writes are atomic and each re-reads the file
  first, so sequential runs and separate sessions share one count; two processes recording at
  the same instant can still lose one write.
- **`meta_label_metrics` reports 0.0 for a ratio with nothing in its denominator** — precision
  when the overlay takes no bets, for example — as scikit-learn does by default.

## Related modules

- [`backtest-statistics`](/modules/backtest-statistics/) — the Rust PSR, DSR and MinTRL this
  module calls, and a longer treatment of what they assume.
- [`strategy-risk`](/modules/strategy-risk/) — the Chapter 15 failure probability.
- [`labeling`](/modules/labeling/) — triple-barrier and meta-labels.
- [`backtesting-engine`](/modules/backtesting-engine/) — CPCV paths, whose Sharpe ratios are
  trials too.

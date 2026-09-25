---
title: "ef3m"
description: "EF3M: fit a mixture of two Gaussians by matching its first four or five moments exactly, from many random starts, and take the mode of the fits."
status: authored
last_authored: '2026-09-25'
audience:
  - quant-dev
  - platform-engineering
module: "ef3m"
api_surface: "both"
afml_chapter:
  - "10"
  - "15"
citation:
  - "López de Prado, M. and Foreman, M. (2014). A mixture of Gaussians approach to mathematical portfolio oversight: the EF3M algorithm. Quantitative Finance 14(5), 913–930."
  - "López de Prado, M. (2018). Advances in Financial Machine Learning. Wiley. §10.2 Strategy-Independent Bet Sizing Approaches; §15.4.1 Algorithm (strategy risk)."
rust_api:
  - "M2N"
  - "FitResultRow"
  - "centered_moment"
  - "raw_moment"
  - "most_likely_parameters"
python_api:
  - "ef3m.fit_m2n"
  - "ef3m.most_likely_parameters"
  - "ef3m.centered_moment"
  - "ef3m.raw_moment"
sidebar:
  badge: Module
---

Bet outcomes are rarely one bell curve. A strategy that cuts losers and lets winners run
produces many small losses and a few larger gains; a series of concurrent bets clusters
around a few typical levels. A mixture of two Gaussians,

$$
f(x) = p_1\,\phi(x;\mu_1,\sigma_1) + (1-p_1)\,\phi(x;\mu_2,\sigma_2),
$$

describes both with five numbers. **EF3M**, the *Exact Fit of the first 3 Moments*
(López de Prado and Foreman, 2014), finds them from the sample's moments rather than from
the data points, so it needs only five sums over the sample, however long it is.

AFML uses the fitted mixture twice. In §10.2 its CDF turns the number of concurrent long and
short bets into a bet size; in §15.4.1 its two means stand in for the average loss and the
average gain when estimating the probability that a strategy fails.

## How the fit works

A two-component mixture has five parameters, so five moments pin it down. Each raw moment
$m_k=\mathrm E[x^k]$ of the mixture is a polynomial in them; for example
$m_1 = p_1\mu_1+(1-p_1)\mu_2$ and $m_2 = p_1(\sigma_1^2+\mu_1^2)+(1-p_1)(\sigma_2^2+\mu_2^2)$.
The equations have no closed-form solution, so EF3M iterates. Given a guess for $\mu_2$ and
$p_1$, it solves in turn:

1. $\mu_1$ from $m_1$;
2. $\sigma_2$ from $m_3$, then $\sigma_1$ from $m_2$;
3. a new $p_1$ from $m_4$ (**variant 1**, four moments), or a new $\mu_2$ from $m_4$ and a new
   $p_1$ from $m_5$ (**variant 2**, five moments).

It repeats until $p_1$ stops moving by more than `epsilon`, keeping whichever iterate
reproduces the five moments with the smallest squared error. A step whose variance comes out
negative, or whose $p_1$ leaves $[0,1]$, ends that attempt.

`fit_m2n` runs that iteration from a grid of starting values
$\mu_2 = m_1 + i\,\varepsilon\,\cdot\,\texttt{factor}\cdot\sigma$ for $i=1,\dots,1/\varepsilon$,
each with a random starting $p_1$, and returns the best fit found. The random start means two
calls give slightly different answers. The paper's remedy is to repeat the whole search
(`n_runs`) and take, for each parameter separately, the peak of a kernel density over the
runs: `most_likely_parameters`.

## Recovering a known mixture

```python
import random
from openquant import ef3m

# 20,000 bet outcomes: 70% are losses around -1, 30% are wins around +2.
random.seed(7)
outcomes = [
    random.gauss(-1.0, 1.0) if random.random() < 0.7 else random.gauss(2.0, 0.5)
    for _ in range(20_000)
]

# EF3M works from raw moments E[x], E[x^2], ..., E[x^5].
moments = [sum(x**k for x in outcomes) / len(outcomes) for k in range(1, 6)]
print("raw moments:", "  ".join(f"{m:.3f}" for m in moments))
print("variance:", round(ef3m.centered_moment(moments, 2), 3))

# 25 independent fits of the five-moment variant, summarised by the mode of each parameter.
rows = ef3m.fit_m2n(moments, epsilon=1e-4, variant=2, n_runs=25)
fit = ef3m.most_likely_parameters(rows, 10_000)
print("fits:", len(rows))
for name in ("mu_1", "sigma_1", "mu_2", "sigma_2", "p_1"):
    print(f"  {name:8s} {fit[name]:5.1f}")

# The simple alternative in AFML 15.4.1: the average loss and the average win.
losses = [x for x in outcomes if x <= 0]
wins = [x for x in outcomes if x > 0]
print("average loss", round(sum(losses) / len(losses), 2), " average win", round(sum(wins) / len(wins), 2))
print("share of wins", round(len(wins) / len(outcomes), 2))
```

```text
raw moments: -0.097  2.672  0.027  13.751  -2.775
variance: 2.663
fits: 25
  mu_1      -1.0
  sigma_1    1.0
  mu_2       2.0
  sigma_2    0.5
  p_1        0.7
average loss -1.28  average win 1.6
share of wins 0.41
```

The fit recovers the mixture that generated the data. The last two lines show why that
matters. Splitting the outcomes at zero gives an average win of 1.6 and a 41% hit rate, because
the right tail of the loss component falls above zero and is counted as winning. The mixture
separates the two populations: wins average 2.0 and are 30% of bets. Fed into
[`strategy_risk`](/modules/strategy-risk/), the two descriptions imply quite different
probabilities of failure.

To size bets from a fitted mixture, pass it in the order `[mu_1, mu_2, sigma_1, sigma_2, p_1]`
as the `fit` argument of [`bet_sizing`](/modules/bet-sizing/)'s `bet_size_reserve` or
`single_bet_size_mixed`.

## Which variant

On the exact moments of the mixture above, 100 runs of each variant gave:

| `variant` | `epsilon` | time per run | runs within 0.05 of the truth | median error |
| --- | --- | --- | --- | --- |
| 1 (four moments) | 1e-3 | < 0.1 ms | 6% | 0.96 |
| 1 | 1e-4 | 0.4 ms | 17% | 0.14 |
| 1 | 1e-5 (default) | 3.5 ms | 69% | 0.015 |
| 2 (five moments) | 1e-3 | < 0.1 ms | 41% | 0.006 |
| 2 | 1e-4 | 0.3 ms | 100% | 0.0001 |
| 2 | 1e-5 | 3.2 ms | 100% | < 1e-6 |

Variant 2 is more accurate at every setting and no slower. At coarse `epsilon`, two-thirds of
variant 1's runs ended on a degenerate fit with $p_1\approx1$: one Gaussian for everything
and a second, arbitrarily wide one with no weight. Variant 1's error is measured on all five
moments, including the fifth it does not fit, which accounts for part of its higher error but
not the degenerate fits. Prefer variant 2 unless the fifth moment is too noisy to trust.

## From Rust

```rust
use openquant::ef3m::{centered_moment, most_likely_parameters, raw_moment, M2N};

// The exact raw moments of 0.7 N(-1, 1) + 0.3 N(2, 0.5^2). Parameters are ordered
// [mu_1, mu_2, sigma_1, sigma_2, p_1].
let truth = [-1.0, 2.0, 1.0, 0.5, 0.7];
let moments = M2N::with_defaults(vec![]).get_moments(&truth, true).unwrap();
assert!((moments[0] + 0.1).abs() < 1e-12); // 0.7 * -1 + 0.3 * 2

// Raw and centred moments convert both ways. The first centred moment is 0.
let central = (1..=5).map(|k| centered_moment(&moments, k)).collect::<Result<Vec<_>, _>>()?;
assert!(central[0].abs() < 1e-12);
assert!((central[1] - 2.665).abs() < 1e-12); // the variance
let round_trip = raw_moment(&central, moments[0]);
assert!(round_trip.iter().zip(&moments).all(|(a, b)| (a - b).abs() < 1e-9));

// 25 runs of the five-moment fit (variant 2), then the mode of each parameter.
let m2n = M2N::new(moments.clone(), 1e-4, 5.0, 25, 2, 100_000, 1);
let rows = m2n.mp_fit()?;
assert_eq!(rows.len(), 25);
let fit = most_likely_parameters(&rows, None, 10_000);
for (name, want) in
    [("mu_1", -1.0), ("mu_2", 2.0), ("sigma_1", 1.0), ("sigma_2", 0.5), ("p_1", 0.7)]
{
    assert!((fit[name] - want).abs() < 0.02, "{name} = {}", fit[name]);
}

// Each row's error is the squared moment error of that row's own parameters.
let mut check = M2N::with_defaults(moments.clone());
for row in &rows {
    let implied = check
        .get_moments(&[row.mu_1, row.mu_2, row.sigma_1, row.sigma_2, row.p_1], true)
        .unwrap();
    let error: f64 = moments.iter().zip(&implied).map(|(a, b)| (a - b).powi(2)).sum();
    assert!((error - row.error).abs() < 1e-9);
}
```

In Rust, `M2N::single_fit_loop` is one run and `mp_fit` is `n_runs` of them; Python's
`fit_m2n` calls `mp_fit`.

## What to watch for

- **Results are random and there is no seed.** The starting $p_1$ of every attempt comes from
  the thread's random number generator. Use `n_runs` and `most_likely_parameters`, and round to
  the precision you actually need; the example is stable to one decimal with 25 runs.
- **Variant 2 fits about the mean.** Its fourth-moment step determines only $\mu_2^2$ and
  takes the positive root, so on raw moments it could not return a negative $\mu_2$.
  `single_fit_loop`, `mp_fit` and `fit_m2n` therefore run variant 2 on the moments of
  $X-\mathrm E[X]$ and add the mean back to $\mu_1$ and $\mu_2$; each row's `error` is still
  measured against the raw moments you passed. Either sign of $\mu_2$ works, as for a mixture
  of $-3$ and $-1$, and you no longer need to centre the sample yourself. Calling `M2N::fit` or
  `iter_5` directly on raw moments still takes the positive root
  ([#115](https://github.com/Open-Quant/openquant/issues/115)).
- **The components may come back in either order.** The search starts with $\mu_2$ above the
  mean, but the iteration can end with the labels swapped. A mode taken over runs that disagree
  on the labelling is meaningless, so check that the runs agree before calling
  `most_likely_parameters`.
- **The mode of each parameter is not a fit.** `most_likely_parameters` takes the peak of each
  column separately, so the five numbers it returns may come from different runs and need not
  reproduce the moments. Its kernel bandwidth is $\hat\sigma\,n^{-1/5}$, it evaluates the density
  on `res` points (at least 10) between the column's minimum and maximum, and it rounds to five
  decimals. The `error` column is ignored by default.
- **The sample moments carry the risk.** The fifth moment of 20,000 draws is still noisy, and
  heavy tails make it worse. A fit that matches noisy moments exactly is exactly wrong. Check
  a fit against a histogram of the data, or against
  [`bet_sizing`](/modules/bet-sizing/)'s EM fit (`bet_size_reserve_full`), which uses every
  observation.
- **The moments are raw, not centred.** `fit_m2n` expects $\mathrm E[x^k]$ for $k=1..5$. Use
  `raw_moment(central, mean)` to convert centred moments, passing the first centred moment,
  which is 0, as `central[0]`.
- **`mp_fit` is serial.** Despite the name and the `num_workers` field, the runs execute one
  after another. At the default `epsilon` a run takes a few milliseconds on the example, so
  this rarely matters.
- **Until this release, a row's `error` could belong to other parameters.** A run that
  converged returned its last iterate alongside the error of an earlier, better one; on the
  example's moments with variant 1 at `epsilon=1e-3` this happened in about one run in seven,
  once reporting an error of 1.3 for parameters whose error was 8.1. A run now returns its best
  iterate. Python's `fit_m2n` also used to ignore `n_runs` and return at most one row.

## Related modules

- [`bet-sizing`](/modules/bet-sizing/) — sizes bets from a two-Gaussian mixture of concurrent
  bets (AFML §10.2), with its own EM fit.
- [`strategy-risk`](/modules/strategy-risk/) — the probability that a strategy fails, from the
  average win and loss that a mixture can supply (AFML §15.4.1).
- [`backtest-statistics`](/modules/backtest-statistics/) — the higher moments of returns in
  the probabilistic and deflated Sharpe ratios.

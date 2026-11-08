# Self-financing fees and rebalancing controls

Status: optional synthetic research example, not a supported portfolio optimizer.

This lab asks what an explicit post-fee target means and how costs change a fixed
rebalancing comparison. It does not change `risk_model_lab.py`, which already
labels its full-L1 proportional wealth debit as a toy convention. No market fee,
forecast, execution-quality or investment-performance estimate is supplied.

## Model and independent references

Pretrade risky weights `w` are nonnegative and sum to at most one; the remainder
is cash. Target risky weights `v` are fully invested and long-only. A symmetric
fee `c` in `[0,1)` applies to each dollar bought or sold. If post-fee wealth is
`alpha` times pretrade wealth, solve

    alpha + c * sum(abs(alpha * v - w)) = 1.

The trades are `alpha * v - w`, expressed as fractions of pretrade wealth. This
is full-L1 traded notional, not half-L1 turnover. The fee changes the dollar
amount needed to achieve a post-fee target. The function is continuous and
piecewise affine, with slopes at least `1-c`; its unique root lies in `(0,1]`.

Independent elementary fixtures are:

- Entry from cash: `alpha = 1/(1+c)`.
- Complete switch between assets: `alpha = (1-c)/(1+c)`.
- Already at target: `alpha = 1`.
- At stress fee 10%, changing `(1,0)` to `(.9,.1)` gives `45/46`, while reversing
  that change gives `54/55`. The toy multiplier is `.98` in both cases. Its
  approximation error therefore has no universal direction.

For a two-period path `(a,0)` followed by `(-a/(1+a),0)`, compare buy-and-hold
with one midpoint 50/50 rebalance, paying the same initial entry cost. The latter
beats hold exactly when `c < a/(a+2)`. At `a=.002`, this is about 9.99001 basis
points. This is a deterministic counterexample, not an estimated trading rule.

## Numerical contract

The ordinary path solves the affine intervals separated by `w_i/v_i`, with a
rational boundary fallback. Target sums must be within 16 ulps of one and are
explicitly divided by their rounded sum. Pretrade sums slightly above one use
the same roundoff allowance; meaningful remaining cash is preserved. Malformed
sums, Boolean weights, negative/nonfinite weights and invalid rates reject.

For fee above .25, exact rational simplex normalization precedes the rational
root and trade calculation. This is an engineering stress path, not a fee
recommendation. Without exact normalization, even a half-ulp target-sum error
can substantially change a root near `c=1`. Returned exact coefficient/root
strings identify that model; ordinary float fields are rounded representations.
The full trajectory remains binary64 arithmetic, not exact arithmetic.

`self_financing_residual` is a floating ledger gap. In the stress path it uses
rounded alpha and rounded exact fee, not the exact function evaluated at rounded
alpha. `residual_condition_indicator` is diagnostic, **not a certified error
bound**. Positive-price-relative growth and independently carried cash avoid
inventing cash from cancellation or small weight-sum errors after drift.

## Fixed policies and experiment

Every risky policy begins with the same fee-paying equal-weight purchase. The
policies are cash, buy-and-hold, daily equal weights, equal weights every 21
observations, and max-coordinate drift thresholds .02 and .10. Decisions use
only pretrade holdings; current/future returns cannot influence the current
trade. Cash earns zero. Holdings drift between trades. There is no final
liquidation, leverage, financing, price impact, spread variation, taxes or lots.

Each policy/accounting combination has its own ledger over the same exogenous
paths. `rebalance_attempt_count` includes scheduled zero-trade resets;
`nonzero_trade_count` does not. Fee-model gaps are fractions of contemporaneous
pretrade wealth, not dollar amounts. Total fees are in units of initial wealth.
The toy comparator can exhaust wealth at extreme fees; it fails explicitly
instead of clipping. The fixed grid uses rates safely below that region.

The plan was recorded before outcomes: seed 20261108, two assets, 252
observations, fees 0/1/10/100 bps, two deterministic paths (alternating reciprocal
price relatives at amplitude .02 and a fixed trend), and 128 paths each of IID
independent and perfectly common zero-mean Rademacher returns of magnitude .02.
There are 258 paths and 12,384 policy/fee/accounting records. No method is selected
using the results. Raw paths and their hashes are retained.

At the final source snapshot, daily minus buy-and-hold terminal wealth was:

| Family | No fee | 10 bps | 100 bps |
| --- | ---: | ---: | ---: |
| Deterministic alternating | .050648 | .045394 | -.000293 |
| Deterministic trend | -.081575 | -.081920 | -.084986 |
| IID zero-mean paths | -.001552 | -.004073 | -.026257 |
| Common zero-mean paths | 0 | 0 | 0 |

IID paired Monte Carlo standard errors are approximately .003129, .003125 and
.003121, respectively. They are pointwise uncertainties from 128 independent
whole paths, not simultaneous intervals, time-series standard errors or market
uncertainty. Deterministic paths have no Monte Carlo error. Common shocks preserve
equal weights, so scheduled resets do not produce nonzero trades after entry.

Under the declared conditional-zero-mean model, predictable self-financing
policies with nonnegative fees cannot increase expected wealth above the cash
benchmark. This follows by conditional expectation; finite simulation means can
be above one. It is not a claim that market returns satisfy those assumptions.

The first run is retained separately. Independent boundary review found and
fixed near-total-loss cancellation, invented cash from drift roundoff, and
extreme-fee normalization. The unchanged fixed grid was then requalified. Paths
and config matched exactly; maximum terminal-wealth change was 7.78e-15. The
final grid took 29.43 seconds locally. These timings are observations, not a
performance benchmark against a production implementation.

## Run and verify

Only NumPy is needed by this lab; pytest runs the tests. It imports no native
OpenQuant extension. Use the project environment, or the offline research
requirements where appropriate:

    python experiments/self_financing_lab.py --out /tmp/self-financing.json
    python experiments/replay_offline_report.py /tmp/self-financing.json --lab self_financing
    pytest python/tests/test_self_financing_lab.py -q

The CLI refuses to overwrite an existing output. Replay checks exact scientific
results with matching source and NumPy identities, not authenticity or financial
applicability. Reports preserve every failed path rather than dropping it.

Independent qualification used 6,448 all-sign-pattern rational oracle cases,
36 rational-reference trajectories and 25 separate adversarial tests. Maximum
single-rebalance root error in that grid was .667 ulp; maximum trajectory wealth
relative error was about 1.12e-15. These finite tests do not prove correctness for
all binary64 inputs. The source suite adds 44 tests, including chronology,
invalid inputs, analytic cost thresholds and report replay/tampering.

## Sources and attribution limits

[Boyd et al., Multi-Period Trading via Convex Optimization](https://web.stanford.edu/~boyd/papers/pdf/cvx_portfolio.pdf)
provides a public discussion of dollar holdings, cash and self-financing costs.
Its models and turnover conventions are not silently substituted here. The
single-rebalance and reversal formulas above are independent elementary
derivations, not copied implementations.

[Paleologo's official author page](https://linktr.ee/paleologo) and the
[Wiley-VCH book record](https://www.wiley-vch.de/en/areas-interest/finance-economics-law/finance-investments-13fi/investments-securities-13fi3/the-elements-of-quantitative-investing-978-1-394-26545-9)
establish that transaction costs and capacity are relevant topics in *The Elements
of Quantitative Investing* (2025). The chapter/sample text was inaccessible in
this session. No theorem or implementation is attributed to an unread chapter,
and no access restriction was bypassed.

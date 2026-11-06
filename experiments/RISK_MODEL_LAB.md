# Rolling risk-model utility lab

Status: experimental research scaffold, not a supported portfolio product.
Owner: OpenQuant maintainers. Audience: researchers comparing risk estimates and allocations.

## Purpose and scope

The [selection-validation lab](VALIDATION_LAB.md) tests candidate search and diagnostic
scores. This next experiment asks a different question: how do existing allocations
behave when covariance is estimated from a rolling historical window and evaluated
on subsequent observations? It measures realized risk, known synthetic conditional
risk, covariance conditioning, weight changes, concentration and traded fractions.
Every method is fixed before simulation. There is no selected winner or promotion gate.

The implementation calls existing OpenQuant allocators. It changes no production API,
adds no dependency, and makes no claim of new covariance-estimator theory. Alternatives
were another diagnostic score or a new shrinkage estimator; comparing existing methods
first gives a reproducible baseline without duplicating those capabilities. Real market
data, new estimators, forecasts of expected returns, cost calibration and execution are
outside this slice. No book text or third-party implementation is copied.

## Run offline

Build the extension as described in [CONTRIBUTING.md](../CONTRIBUTING.md). Then, from
the repository root:

```bash
.venv/bin/python experiments/risk_model_lab.py --out /tmp/openquant-risk.json
.venv/bin/python -m pytest python/tests/test_risk_model_lab.py -q
```

Defaults: 50 independent paths per family, 8 assets, 126 training observations,
252 evaluated observations, rebalancing every 21 observations, seed 7 and a toy
10-basis-point cost per full traded fraction. The three families and six methods
produce 900 method-path evaluations. A default full report is about 31 MB because
it retains daily returns and every rebalance. Results are not annualized.

For a shorter-window sensitivity check, keep the rest of the protocol unchanged:

```bash
.venv/bin/python experiments/risk_model_lab.py --train-size 42 \
  --repetitions 200 --out /tmp/openquant-risk-short-window.json
```

The `--assets`, `--holdout-size`, `--rebalance-every`, `--cost-bps`, `--seed` and
`--families` options define a new experiment, not an optimization search. Repeatedly
choosing these settings after inspecting results is itself researcher selection.

## Fixed allocation methods

All methods are long-only and fully invested, with bounds `[0, 1]`. Assets always
remain in their original column order. Inputs use simple returns, not prices.

| Method | Covariance and allocator |
| --- | --- |
| `equal_weight` | Rebalance each asset to `1 / n_assets`; sample covariance is retained only for diagnostics. |
| `inverse_variance` | Existing `portfolio.allocate_from_inputs(..., "inverse_variance")`; sample diagonal only. |
| `min_vol_sample` | Existing `portfolio.allocate_from_inputs(..., "min_volatility")`; full sample covariance. |
| `min_vol_shrunk` | Same minimum-variance solver, with the fixed diagonal shrink below. |
| `hrp_sample` | Existing `hrp.allocate_hrp`; sample covariance and explicit `distance_of_distances`. |
| `hrp_shrunk` | Same HRP allocator, with the fixed diagonal shrink below. |

Sample covariance uses the centered cross-product divided by `train_size - 1`.
The fixed shrink is `0.9 * S + 0.1 * diag(diag(S))`: marginal variances stay unchanged
and off-diagonal entries are multiplied by 0.9. This reproduces the existing HRP
`use_shrinkage=True` convention, tested against that API. It is **not Ledoit-Wolf**,
an estimated shrinkage intensity, or a claim of optimal shrinkage. Supplying covariance
explicitly makes the same model available to both HRP and minimum variance.

Expected returns are zeros because none of these allocations requires a mean forecast.
No hidden jitter, pseudoinverse or fallback allocator is added when a method fails.
Solver weight residuals below the documented numerical tolerance are clipped to zero
and renormalized, not used to permit leverage.

## Chronology and drift

At rebalance index `t`, fit only rows `[t - train_size, t)`, then hold the resulting
portfolio over `[t, min(t + rebalance_every, n_observations))`. The final short block
is retained. There is no overlap or gap in the evaluated daily returns. Later fits
can include earlier evaluated observations because they have become history by then.
They never use the current or future row. Tests change the future suffix at several
rebalance boundaries and verify that the decision at that boundary is unchanged.

Within a block there is no daily rebalance. Given current weights `w` and asset returns
`r`, gross portfolio return is `g = sum(w * r)`, and the next observation's weights are
`w * (1 + r) / (1 + g)`. Weight drift matters even for equal weight: target-weight
changes can be zero while traded fractions at subsequent rebalances are positive.

At each rebalance:

- `traded_fraction = sum(abs(target_weights - previous_ending_weights))` is **full L1**,
  not half L1. A complete move between assets is 2. Initial weights are zero (cash),
  so the initial purchase has traded fraction 1.
- `target_weight_change_l1` compares consecutive target weights, separately from
  actual drifted holdings. It is unavailable at the first rebalance.
- `cost_fraction = cost_bps / 10000 * traded_fraction` is a declared **toy debit**.
  The first net return in the block is `(1 - cost_fraction) * (1 + g) - 1`.
  Other net returns equal gross returns. The debit reduces wealth proportionally,
  so it does not change asset weights.

This is not an exact self-financing transaction-fee solution based on post-fee target
dollar holdings. It is a consistent turnover proxy applied identically to all methods.
It excludes impact, spreads varying by asset, financing, taxes and a final liquidation.
Terminal gross/net returns are bookkeeping outputs, not expected-return estimates.

## Synthetic controls

All observations are zero-mean Gaussian draws, independent across time conditional
on the specified deterministic covariance schedule. Asset standard deviations are
linearly spaced from 0.005 to 0.02. These are mathematical controls, not market estimates.

| Family | Known covariance |
| --- | --- |
| `diagonal` | Independent assets with heterogeneous variances. |
| `correlated` | The same variances and pairwise correlation 0.75. |
| `regime_shift` | Starts like `correlated`; at the midpoint of the **entire generated sample**, standard deviations reverse asset order and pairwise correlation becomes 0.2. |

The shift index is `floor((train_size + holdout_size) / 2)`. At defaults it is 189,
63 observations after evaluation starts. Changing the training/holdout lengths can
move the shift into training, so record that position when interpreting comparisons.
The true covariance is used only after allocations to measure diagnostics. Tests
change the truth matrices without changing any decisions or realized returns.

Gaussian draws are unbounded. A draw at or below -1 fails instead of being clipped
or resampled, because the buy-and-hold simple-return accounting would be invalid.
The lab does not simulate bankruptcy, heavy tails, serial dependence, missing assets
or transaction latency. Those are future controls, not properties of this run.

## Measurements and uncertainty

Each complete method-path result retains all gross/net returns and rebalance records:

- `realized_variance`: sample variance of the full evaluated gross-return path,
  with denominator `holdout_size - 1`.
- `mean_conditional_variance`: mean of `w_t' * true_covariance_t * w_t`, using the
  actual start-of-observation drifted weights. It is available only when truth is supplied.
- `forecast_variance`: one-observation variance at each rebalance from fitted
  covariance and target weights. It is not an entire-block risk forecast.
- Covariance eigenvalues, numerical rank, condition number and Frobenius error
  against the true covariance at the next observation. Numerical rank uses
  `largest_eigenvalue * n_assets * float64_epsilon`; singular condition numbers are
  `null`, not an arbitrary large replacement.
- Concentration `sum(target_weights ** 2)`, traded fraction and target-weight changes.
  Mean concentration weights rebalances equally, not time or block length.
- Compounded gross and net terminal returns. No annualization or Sharpe estimate.

Summaries average **independent complete paths**, never individual overlapping training
windows. The Monte Carlo standard error is sample standard deviation across paths
divided by the square root of their count. Paired summaries subtract equal weight
within each path before calculating the mean and standard error. This preserves common
random-number pairing. It is not a real-world confidence bound or a multiple-comparison
adjustment, and small replication counts give uncertain simulation precision.

Failure is explicit. Nonfinite/complex returns, invalid dimensions, impossible simple
returns or invalid configuration are rejected. Training covariance must have finite
entries and positive diagonal for this shared experiment domain; a constant-asset
window makes even the equal-weight control unavailable. A method failure makes its
entire path unavailable, with reason and completed-block count. No partial path or
zero result enters the mean. Summaries count unavailable paths. If failures occur,
means and paired comparisons describe successful paths only and may be selection-biased.

## Reproducibility and retained evidence

Every family/replication has a deterministic seed stream independent of requested
family order. The scenario SHA-256 includes shapes plus little-endian float64 bytes
of all generated returns **and known covariance matrices** in replication order.
The report retains the full configuration and generator parameters. CLI provenance
records git revision, experiment source hash, compiled extension hash, Python/NumPy
versions and platform. Cross-version random-distribution or numerical-solver bitwise
identity is not promised. Source/extension hashes distinguish uncommitted code from
the recorded git base.

Timing covers generation, fitting, all six method evaluations, hashing and summaries.
It excludes provenance collection and JSON writing. It is a machine-specific runtime,
not an execution-speed claim about a production strategy or a comparison with another library.

Acceptance is checked by `python/tests/test_risk_model_lab.py`: independent diagonal
allocation/covariance fixtures, native shrinkage equivalence, hand-computed drift and
costs, later-window no-lookahead invariance, diagnostic-only oracle data, regime behavior,
failure accounting, paired precision, deterministic hashes and strict-JSON CLI provenance.
Run the full Python suite, lint and type gates as well as focused tests before merging.

## Sources and limits

- Markowitz, [*Portfolio Selection*, The Journal of Finance 7(1), 1952](https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1540-6261.1952.tb01525.x).
  Publisher bibliographic record for the mean-variance method used by the existing allocator.
- López de Prado, [*Building Diversified Portfolios that Outperform Out-of-Sample*, 2016](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2708678).
  Author-posted paper and publisher DOI [10.3905/jpm.2016.42.4.059](https://doi.org/10.3905/jpm.2016.42.4.059)
  identify the HRP method. This original synthetic protocol is not a reproduction of
  that paper's experiment and does not inherit its empirical performance claims.
- Existing implementation contracts: [portfolio optimization](../crates/openquant/src/portfolio_optimization.rs),
  [HRP](../crates/openquant/src/hrp.rs), and [Python binding tests](../python/tests/test_risk_model_lab.py).
  The fixed shrink and distance convention come from the inspected repository implementation,
  not an unverified textbook formula. No inaccessible full text was used or bypassed.

Next useful extensions are prespecified dependence/tail controls and licensed real-data
replication with explicit market-cost assumptions. Adding a new estimator or choosing
a live allocation requires separate evidence; this lab does neither.

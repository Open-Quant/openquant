# Repeated validation feedback and an independent audit

Status: private experimental scaffold; not a production metric or investment strategy.
Owner: OpenQuant maintainers. Audience: researchers evaluating a search procedure.

## Question and scope

The [existing validation lab](VALIDATION_LAB.md) freezes a training-selected candidate
before one holdout evaluation. It explicitly leaves repeated human reuse of the holdout
and changing candidate families untested. This additional lab tests that gap: a search
uses validation feedback to construct new linear blends, while a final independently
simulated audit evaluates the frozen results.

The hypotheses are descriptive: validation scores improve mechanically when the
incumbent remains a choice, but audit scores need not improve. The experiment reports
both sides and does not tune the search using audit outcomes. In particular, it does
not turn the largest audit score across rounds into a new recommended method.

This is original synthetic code. It does not implement a privacy-preserving reusable
holdout, a Rademacher bound, PBO, PSR, DSR, or an investment promotion decision.

## Run and reproduce

Only NumPy is required; the native OpenQuant extension is not imported:

```bash
python experiments/adaptive_holdout_lab.py --out /tmp/adaptive-holdout.json
python -m pytest python/tests/test_adaptive_holdout_lab.py -q
```

Defaults are 200 independent replications, 252 observations in each of three segments,
32 candidate columns, 16 search rounds, seed 7, and all six existing synthetic families.
Use `--repetitions`, `--sample-size`, `--candidates`, `--rounds`, `--seed`, and
`--families` to specify a protocol before viewing results. Increasing the replication
count estimates this fixed protocol more precisely; it does not supply market evidence.

## Experimental specification

**Data:** Each replication generates three separate matrices: training, validation,
and audit. Candidate columns have the same meanings across matrices. Every segment
has its own deterministic random stream. They are independent simulated samples from
the same family, **not three adjacent slices of one observed time series**. The AR(1)
family retains serial dependence within each segment and resets stationary state
between segments. Neither a temporal gap nor a chronological holdout is claimed to
produce this independence in practice. Known positive drift stays in column zero.

**Score:** Per-observation mean divided by sample standard deviation, with no
annualization. Each proposed blend is scaled by one common positive scale across
its active columns before its moments are evaluated; per-column normalization would
change the blend. Unused columns cannot change a blend's scaling or rounding.
There is no rebalancing, transaction-cost model, portfolio-budget constraint, learned
predictive model, or market calibration. Signed weights are mathematical research
combinations. Their absolute weights sum to at most one.

**Training:** Rank all original candidates by training Sharpe, breaking exact ties
by original column order. Freeze that order. Start with its first candidate.

**Validation feedback:** At round `t`, use the next training-ranked column `e_t` and
the incumbent weights `w`. Evaluate three proposals, in order:

1. Retain `w`.
2. Use `(w + e_t) / 2`.
3. Use `(w - e_t) / 2`.

Choose the largest validation Sharpe; ties retain the first proposal. The chosen
weights become the next round's incumbent, so later candidates genuinely depend on
earlier validation feedback. There is no threshold-based early stop. The maximum
number of rounds is one less than the number of original candidates. Report every
round, including round zero. A shorter search is an exact prefix of a longer search.

**Controls:** The frozen training winner never adapts. A second comparator chooses
the best individual candidate on validation among the same training-ranked prefix
queried so far. The adaptive search has a richer hypothesis class and more score
queries, so these are diagnostic controls, **not a matched-compute comparison**.
All are scored on the same audit matrix for paired comparisons.
The report counts `1 + 2*t` adaptive new-proposal slots versus `1 + t` single-candidate
scores used after round `t`; these are logical search counts, not wall-clock calls or
unique hypothesis counts (some proposals may duplicate earlier ones or be undefined).
Validation scores are evaluated lazily only when their columns are first queried.

**Audit isolation:** The search function takes no audit argument. It returns the
complete frozen trajectory before the evaluator reads audit data. Altering audit
values cannot change any training order, proposal choice, or selected weights. The
audit is read once for reporting all prespecified rounds. Choosing a round or changing
the method after those results would reuse the audit as validation, requiring a new
evaluation design. Process isolation and access controls are not implemented.

**Undefined results:** Constant training candidates or queried validation candidates, nonfinite inputs, shape
mismatches and invalid configurations fail closed. A constant blend proposed on
validation cannot win and contributes to the explicit rejected-proposal count.
A selected blend that becomes constant on audit is reported as unavailable (`null`),
never zero. Replication records remain present. Summaries show available/unavailable
counts; if outcomes are missing their means describe only available outcomes.

**Precision:** Each round's summaries use independent replication-level observations.
Differences in audit scores are computed within each replication before taking their
mean and Monte Carlo standard error. Rounds within one replication are not independent
observations, and their repeated inspection is not a multiple-testing correction.
The heavy-tail and short-sample controls can have unstable moment estimates. Monte
Carlo precision is not a probability of profitable investment performance.

**Provenance:** Save all weights, proposal choices, queried-column counts, raw scores,
rejected-proposal counts and paired summaries. Shape and canonical little-endian
float64 bytes for all input matrices feed a scenario hash in replication/segment
order. Record config, Git base, Python and NumPy versions, and source hashes for both
this script and its base-lab dependency. Scenario streams are independent of requested
family ordering. Reproducibility across NumPy versions is not promised.

## Acceptance and limitations

### Recorded synthetic findings, November 7, 2026

The retained private run used 400 replications per family, 252 observations per
segment, seed 7, and candidate/round settings `(1,0)`, `(8,7)`, `(32,16)` and `(64,32)`.
All six families were retained at every setting, including every intermediate round.
For 32 candidates and 16 rounds, selected illustrative means were:

| Family | Adaptive validation Sharpe | Adaptive audit Sharpe | Single-candidate audit Sharpe | Paired adaptive minus single audit, mean ± MCSE |
| --- | ---: | ---: | ---: | ---: |
| IID zero-mean | 0.17890 | 0.00134 | 0.00454 | -0.00320 ± 0.00319 |
| Serial zero-mean | 0.36259 | -0.00203 | -0.01106 | 0.00904 ± 0.00710 |
| Known positive drift | 0.19125 | 0.02942 | 0.04431 | -0.01489 ± 0.00378 |

These are simulation summaries, not significance tests or calibrated skill measures.
The richer adaptive search did not establish an audit advantage. In the known-drift
family its audit mean was lower than the best queried individual-candidate control:
paired differences were -0.01772 ± 0.00385 at 8 candidates/7 rounds and
-0.01674 ± 0.00377 at 64 candidates/32 rounds. These negative findings remain in the
report. Different candidate counts produce different random matrices; do not interpret
differences between those configurations as paired comparisons.

At 64 candidates/32 rounds, IID and serial validation means reached 0.21571 and
0.42997 while their audit means were -0.00396 and 0.00684. The one-candidate/zero-round
control made all three methods identical. No audit mixtures were unavailable in
these runs, although the explicit unavailable-case regression test is retained.

Full source identities, input hashes, per-replication records, and all six-family
summaries belong to the accompanying private result files. These examples do not
replace those records, select a winning budget, or justify use on actual portfolios.

Tests include hand-computed winners and scores, constant proposals, an unavailable
audit blend, zero-round identity, audit perturbation invariance, budget-prefix identity,
stable exact ties, monotone validation score, independent segment stream identities,
paired standard errors, deterministic family-order invariance, all six families, strict
JSON, direct CLI use without the native extension, and malformed input rejection.

The experiment cannot reconstruct how an actual researcher used holdout feedback.
An audit log is useful evidence but cannot enforce independence after someone sees
the data. None of these synthetic results demonstrates that the blends are tradeable,
that a method dominates in real markets, or that a real backtest is free of leakage.
Do not change production APIs until formal requirements, maintained work records,
repository checks, and independent review justify promotion.

## Primary sources and relationship to prior work

- Cawley and Talbot (2010), [On Over-fitting in Model Selection and Subsequent Selection
  Bias in Performance Evaluation](https://www.jmlr.org/beta/papers/v11/cawley10a.html).
  Finite-sample model-selection criteria can themselves be overfit. This motivates
  separating the score optimized by a search from its later evaluation.
- Dwork, Feldman, Hardt, Pitassi, Reingold and Roth (2015), [Generalization in Adaptive
  Data Analysis and Holdout Reuse](https://arxiv.org/abs/1506.02629).
  The paper studies adaptive reuse and provides protected reuse algorithms. This lab
  illustrates unprotected reuse; it does not claim or implement their guarantees.

No book text or third-party implementation was copied. Paleologo's full text remains
unverified; no method or guarantee is attributed to that book by this experiment.

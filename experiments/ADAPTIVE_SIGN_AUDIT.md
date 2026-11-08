# Sign-test calibration after adaptive selection

Status: private experimental scaffold. Owner: OpenQuant maintainers.
This is an additive synthetic study; no production API or investment rule changes.

The [adaptive holdout lab](ADAPTIVE_HOLDOUT_LAB.md) shows score optimism. This study
asks a narrower question: how often does a fixed one-sided sign test reject after
that same selection? It separates contaminated validation scores from an independently
simulated audit. Sign-test p-values never choose weights or stop the search.

## Statistical target

The null is `Pr(return > 0 | return != 0) <= 1/2`. It concerns sign balance, **not**
expected return, Sharpe, PSR, trading profitability, or a general measure of skill.
An iid symmetric distribution centered at zero supplies fair independent signs.
A zero mean alone does not: an asymmetric distribution can have a different median.
Conditioning matters when there are ties: 60% zeros and 40% positive outcomes have
40% positives overall but 100% positives conditional on a nonzero outcome.

Given `n` nonzero observations and `k` positive observations, the one-sided tail is
`sum(comb(n, j), j=k..n) / 2**n`. Zero observations, including signed zero, are dropped;
all-zero data are unavailable rather than a rejection or a non-rejection. Conditional
on the retained count, iid fair nonzero signs justify this calculation. Dependence
among observations invalidates that binomial model.

Integer combinatorial counts compute the tail. Rejection uses an inclusive `<=`
comparison with the exact binary rational represented by the configured float alpha.
The displayed p-value is rounded to a float; the decimal string numerator and the
power-of-two denominator are also saved so tiny probabilities remain reconstructable.
There is no stochastic approximation inside the test.

Discrete tails make the attained null rejection rate conservative. The report records
the minimum attainable p-value, first rejecting positive-sign count, and exact null
rejection probability for the configured sample length with no ties. For four signs,
alpha 0.05 cannot reject at all because the smallest tail is 1/16. For ten signs,
only nine or ten positives reject at alpha 0.05, attaining 11/1024 rather than 0.05.
Actual nonzero counts and ties remain in every replication record.

## Frozen experimental protocol

Use the adaptive lab's six generators and its stream convention
`[seed, family_id, replication, segment, 8401]`. Training, validation and audit are
separately simulated; this is **not** a promise that adjacent market samples become
independent after a gap. The AR(1) generator resets its stationary state across
segments but retains serial dependence within each segment.

Training fixes the candidate order. Validation adaptively chooses the final linear
blend at the configured round budget. Freeze it before reading the audit. Also retain
the training-only winner and the validation-selected best original candidate.

For each of the three choices, report sign-test results on validation and audit.
Adaptive and best-single validation p-values are post-selection and **not calibrated**.
The frozen training winner's validation test uses a choice independent of that segment.
Audit tests use the fixed choices and an independent segment. No choice is revised
after observing audit outcomes, and no best round is selected from the audit.

| Generator | Interpretation of audit rejection rate |
|---|---|
| iid Gaussian null | Fair independent signs; null size diagnostic |
| Correlated Gaussian null | Candidates correlated across columns; rows independent and symmetric |
| Duplicated Gaussian null | Exact duplicate candidates; nondegenerate selected mixtures have fair independent signs |
| Student-t(3) null | Heavy tails but independent symmetric signs; fourth moments are unnecessary for this test |
| AR(1) null | Intentional violation of within-segment independence; no binomial calibration claim |
| Positive drift | Sensitivity/power scenario; not a null false-positive rate |

Mixtures use one positive scale across active columns before summation; signs are
unchanged mathematically. Inactive columns cannot cause underflow of a selected tiny
column. These are floating-point computations, not exact symbolic cancellation tests;
the numerical regime and all ties are retained in the evidence.

## Run and provenance

Only NumPy is needed. No compiled extension, market data, credentials, or book text:

```bash
python experiments/adaptive_sign_audit.py --seed 71 --repetitions 1000 \
  --out /tmp/sign-audit-preliminary.json
python experiments/adaptive_sign_audit.py --seed 107 --repetitions 10000 \
  --families iid_null serial_null --out /tmp/sign-audit-confirmation.json
python -m pytest python/tests/test_adaptive_sign_audit.py -q
```

Both protocols were specified before examining results. They use 252 observations
per segment, 32 candidate columns, 16 adaptive rounds and alpha 0.05. Every replication
retains final weights, selected candidate indices, sign counts, exact tails, rejections
and unavailable states. Shape and canonical little-endian float64 bytes hash all input
segments. Source hashes cover this script plus both imported local lab modules. Config,
Git base, Python/NumPy versions and runtime are saved. No result is dropped for being
unfavorable; summaries report unavailable counts.

Rejection rates use the available outcomes and Wilson 95% intervals across independent
replications. These intervals quantify simulation precision; they are not simultaneous
intervals across families or methods and not confidence in a trading strategy.
Differences between methods are paired within each replication before computing the
mean and Monte Carlo standard error. A path with an unavailable endpoint contributes
to the missing-pair count, not a fabricated zero difference. Results conditional on
availability must not be interpreted as unconditional rates when outcomes are missing.

## Acceptance checks

Tests include hand-derived four- and ten-sign tails, inclusive exact thresholds,
exhaustive enumeration of all 1,024 fair ten-sign outcomes, the no-rejection four-sign
control, dropped ties/all ties, tiny displayed p-value underflow with retained exact
tail, audit-mutation invariance, zero-round frozen-control equality, extreme inactive
columns, Wilson endpoints, paired rejection differences, deterministic family-order
invariance, CLI provenance and malformed inputs.

## Sources

- [NIST Sign Test](https://www.itl.nist.gov/div898/software/dataplot/refman1/auxillar/signtest.htm):
  sign counts, omitted ties, binomial model and discreteness. This implementation states
  its target directly in terms of conditional sign probability to avoid conflating a
  median/sign hypothesis with a mean hypothesis.
- [SciPy binomtest reference](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binomtest.html):
  exact binomial test and one-sided alternative convention. SciPy is not a dependency.
- [NIST confidence intervals](https://www.itl.nist.gov/div898/handbook/prc/section2/prc241.htm):
  Wilson proportion interval formula, used for replication-level uncertainty only.

No adaptive-reuse correction is implemented. No significance claim is transferred from
this synthetic audit to a financial strategy. Reusing the final audit to choose this
protocol would invalidate its role as an untouched confirmation sample.

## Retained results

The seed-71 preliminary run used 1,000 independent replications for each family.
The selected adaptive blend's validation/audit rejection rates were 0.773/0.040
(iid), 0.432/0.051 (correlated), 0.566/0.042 (duplicated), 0.707/0.041
(heavy-tailed), 0.997/0.165 (serial), and 0.813/0.076 (positive drift).
All outcomes were available. The positive-drift adaptive audit rate was below the
frozen training winner's 0.145: this search did not improve this sensitivity control.
That negative result is retained rather than treating higher validation rejection
as evidence of greater power.

The separately seeded confirmation used 10,000 replications each for iid and serial
nulls. The exact no-ties iid binomial size at 252 observations is 0.0443864, with
rejection beginning at 140 positive signs. Alpha 0.05 is the upper bound, not a
promise that finite-sample rejection equals exactly 0.05.

| Family | Adaptive validation rejection | Adaptive audit rejection | Wilson 95% audit interval | Frozen audit rejection |
|---|---:|---:|---|---:|
| iid null | 0.7701 | 0.0453 | [0.04140, 0.04955] | 0.0449 |
| serial null | 0.9967 | 0.1598 | [0.15275, 0.16711] | 0.1567 |

For iid, adaptive-minus-frozen audit rejection was 0.0004 (paired MCSE 0.00294).
For serial, it was 0.0031 (0.00505). The serial distortion affects even the frozen
training winner: independent train/audit segments do not repair dependence within
the audit itself. This is an intentional failed-assumption control, not evidence
that a binomial sign test handles dependent returns.

The iid audit result is compatible with its exact attainable null size in this
simulation. It does not validate a sign test under different data-generating laws,
prove profitable skill, or correct the deliberately invalid post-selection
validation p-values. The confirmation reports retain every selected weight and
sign count, not just these summaries.

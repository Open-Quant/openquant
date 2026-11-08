# Independent experimental units for sign tests

Status: private synthetic diagnostic. Owner: OpenQuant maintainers.
Audience: researchers checking independence assumptions behind rejection rates.
Production APIs and the existing adaptive-sign experiment are unchanged.

The [adaptive sign audit](ADAPTIVE_SIGN_AUDIT.md) separates model selection from an
independent audit sample. That separation does not ensure independence **within**
the audit. This lab studies that second boundary without any adaptive selection.

## What is being tested

For the chosen experimental unit `U`, the one-sided test targets
`Pr(U > 0 | U != 0) > 1/2`. The unit is either a raw observation or an entire
block's sum. Zero units are omitted; all-zero samples are unavailable. The test
uses the existing exact integer binomial tail and compares it with the exact
binary value of alpha. Floating p-values are presentation values.

The NIST sign-test reference describes dropping ties and the discreteness of the
binomial reference distribution. Its binomial definition explicitly requires
independent Bernoulli trials. Those assumptions, not the presence of a column
called “block,” determine calibration.
[Sign-test reference](https://www.itl.nist.gov/div898/software/dataplot/refman1/auxillar/signtest.htm),
[independent Bernoulli definition](https://csrc.nist.gov/glossary/term/binomial_distribution).

This is **not generally a test of mean return or Sharpe**. In this particular
continuous Gaussian location model, symmetry and finite moments make positive
mean equivalent to sign probability above one half. That equivalence does not
extend to arbitrary return distributions. Changing from individual signs to signs
of sums also changes the experimental unit and generally the estimand.

## Three explicit data-generating controls

All observations have marginal variance one and fixed mean `drift`. Within each
length-`m` block the zero-mean process is stationary Gaussian AR(1):

`Z_0 ~ N(0,1)` and `Z_t = phi * Z_(t-1) + sqrt(1-phi²) * epsilon_t`.

Innovations are independent standard Gaussians. This construction gives
`Cov(Z_t,Z_s) = phi^abs(t-s)` by recursively substituting the recurrence. The
experiment restricts `0 <= phi < 1`; it does not estimate phi from a dataset.

| Family | Construction | Independent block signs? |
|---|---|---|
| `independent_blocks` | A new stationary state and innovations for every block | Yes |
| `continuous_path` | One continuous stationary path, reshaped into adjacent blocks | No for positive phi and multiple blocks |
| `shared_factor_blocks` | `sqrt(.5)*common_AR_path + sqrt(.5)*independent_AR_block` | No for multiple blocks |

In the shared-factor control, the same length-`m` common AR path enters every
block. Each block still has exactly the same within-block AR covariance; its sum
has correlation 0.5 with every other block sum. Merely restarting idiosyncratic
states does not remove that common information. Different simulation replications
use new common paths and are independent.

Degenerate controls are handled explicitly. At phi zero, separate blocks and one
continuous path both have iid observations. A single block has no cross-block
independence violation. A one-observation independent block has independent raw
observations regardless of phi.

## Original model reference derivation

Summing the `m` diagonal variances and each off-diagonal covariance twice gives

`v_m = m + 2 * sum((m-lag) * phi**lag for lag in 1..m-1)`.

Thus a block sum is `N(m*drift, v_m)`, and

`q = Phi(m*drift / sqrt(v_m))`.

For `B` independently generated blocks, the positive count is `Binomial(B,q)`.
Let `c` be the smallest attainable positive count whose fair-sign tail is at most
alpha. The model rejection probability is

`sum(comb(B,k) * q**k * (1-q)**(B-k) for k in c..B)`.

If no such `c` exists, rejection probability is zero at every drift. At zero drift,
`q=1/2`, recovering the exact attainable null size. The binomial mass function is
also documented by [NIST](https://itl.nist.gov/div898/handbook/eda/section3/eda366i.htm).
The covariance and Gaussian-sum derivation above is original algebra for this
specified simulator. Reference powers use floating-point evaluation of that model
formula; they are not empirical threshold calibration.

The reference is valid only for independent blocks (and declared degenerate iid
cases). Dependent controls retain the same marginal `q`, but their counts are not
binomial; their report's exact-reference field is `null`. At B=16 and alpha=.05,
the attainable null size is approximately .038406, not .05. At B=4, even four
positive sums have p=.0625, so rejection is impossible. Zero rejection in that
case is not evidence that a useful test was calibrated.

## Run and reproduce

NumPy is the only third-party runtime dependency. The script imports the existing
sign-test helpers and records their full local dependency closure.

```bash
python experiments/cluster_sign_lab.py --out /tmp/cluster-sign.json
python -m pytest python/tests/test_cluster_sign_lab.py -q
```

Defaults: 1,000 replications, 16 blocks of 16 observations, phi=.6, zero drift,
alpha=.05, seed113. Config arguments map directly to `run_lab(**config)`.
Each random stream is keyed by seed, family, block count, block size, exact binary
phi/drift representation, and replication index. Family or grid ordering cannot
change its data. Cross-version or cross-machine bitwise equivalence is not promised.

Every report retains per-replication raw and block sign counts, exact tail integer,
nonzero and tie counts, rejection, and all block sums. Input arrays contribute
shape and canonical little-endian float64 bytes to a SHA-256 digest. Source and
local dependency hashes, NumPy/Python versions, git revision, config and timing
are saved by the CLI. Raw signs are not serialized individually; hashes identify
the replayed arrays. Exact sign-test results are cached by sufficient counts to
avoid repeating identical large-integer tail calculations.

Rates condition on available tests and retain unavailable counts. Wilson 95%
intervals describe independent replication-level rejection events; they are not
simultaneous guarantees over the grid. MCSE uses the sample standard deviation of
those event indicators divided by `sqrt(available replications)`. Paired raw-minus-block
differences preserve dependence between tests on the same path. These are
simulation uncertainties, not uncertainty about a real strategy's profitability.

## Frozen first grid and retained failures

Before running, the first grid fixed seed113 and 1,000 replications for all three
families in 17 cases:

- B=16, m=16: phi in {0,.6,.9} crossed with drift in {0,.1,.2}.
- phi=.9, drift in {0,.1}: (B,m) in {(4,16),(64,16),(16,4),(16,64)}.

This is a selective grid, not the complete Cartesian product. Sample length changes
when B or m changes, so those sweeps do not isolate a pure block-length treatment.
The complete records and frozen grid are retained in the private checkpoint.

At phi=.9, B=16, m=16, zero-drift rejection rates were:

| Family | Raw observations | Block sums |
|---|---:|---:|
| Independent blocks | .270 | .039 |
| Continuous path | .324 | .084 |
| Shared AR factor | .415 | .281 |

Independent block-sum null behavior was close to its .038406 attainable reference.
Grouping a continuous path reduced the failure but did not establish independence;
the shared-factor failure remained substantial. Under independent blocks with
fixed drifts .1 and .2, observed block-sum rejection was .086 and .171 against model
references .086246 and .167945. The larger raw rates cannot be interpreted as a
power advantage without their much larger false-positive rate.

Negative controls remain in the evidence. Four blocks rejected nothing, including
the positive-drift cases. With phi=.6, B=m=16, the continuous-path block rate was
.036: a favorable finite simulation does not prove its independence assumption.
One independent-block null case (phi=.9, B=16, m=4) returned .024 with Wilson
interval [.01618,.03546], excluding the known .038406 reference. This outlier is
preserved. Marginal intervals across many cases are not simultaneous coverage.
Follow-up simulation is explicitly a post-grid diagnostic, not a replacement of
the original sample or a reason to change its threshold.

## Acceptance and limits

Tests compare sum variance with a directly constructed covariance matrix and
hand-derived values; verify exact power `q**5` for the five-all-positive critical
case; exercise the four-block unattainability; check phi-zero generator equality;
and preserve ties, unavailable results, strict config and replay invariance.
A seeded covariance diagnostic independently confirms approximate cross-block
sum correlations for each generator. No model settings are fitted to that test.

There is no chosen block size from data, market model, HAC correction, trading
cost, heavy-tailed block model, adaptive search, or claim that real-data blocks
are independent. Long blocks in a stationary mixing process may reduce dependence,
but finite grouping alone supplies no exact binomial guarantee. The practical
lesson is to identify independently generated experimental units and verify the
assumptions separately from the mechanics of a sign test.

The post-grid diagnostic used two new independent seeds, 271 and 811, with 20,000
replications each of the outlying independent-block configuration. Block rejection
was .03705 (Wilson [.03452,.03976]) and .03670 ([.03418,.03940]); both intervals
included .038406. These checks are consistent with the model reference. They do
not explain away or remove the original .024 observation, and no threshold,
family definition, or test statistic was changed.

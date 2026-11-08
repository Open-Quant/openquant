# Candidate multiplicity and repeated monitoring

This experiment addresses both candidate selection and optional stopping. It
reuses the frozen [sequential sign utility](SEQUENTIAL_SIGN_LAB.md), with the
same predeclared equal mixture q=(.55,.65,.8). Candidates are fixed before data.
Each candidate is monitored throughout the horizon. Three procedures are
compared without choosing between them after observing results:

- Uncorrected individual capitals cross 1/alpha; making any candidate claim this
  way ignores candidate multiplicity.
- Bonferroni candidate capitals cross 1/(alpha/K), retaining candidate attribution.
- Global average capital K^-1 sum_j M_j crosses 1/alpha, making only a global-null
  discovery. Its largest contributor is descriptive and is not a certified winner.

## Assumptions and derivations

For null candidate j, the sufficient condition is
`P(X_tj=1 | the full joint past)<=.5`. The frozen utility derives the conditional
expected betting factor. Under the global null, all candidate capitals are
nonnegative supermartingales with respect to this common filtration, and their
fixed average is also one. [Ville's inequality, Howard et al., Lemma 1 equation
2.11](https://arxiv.org/pdf/1808.03204) therefore bounds its ever-crossing probability
by alpha. Separate own-history supermartingale assumptions do not automatically
establish the common-filtration premise for the average. Same-time dependence in
IID row vectors is permitted; arbitrary cross-temporal dependence is not claimed.

For attribution, unioning the individual anytime guarantees bounds the chance
of ever making any false candidate claim by K*(alpha/K). The union argument
requires no independence across candidates, and it remains valid when other
candidates have a signal if the individual null guarantees still hold. See
[Stark §3.1](https://statistics.berkeley.edu/sites/default/files/tech-reports/417.pdf)
for the general confidence-event union argument. The implemented budget is rounded
down using exact rational comparisons. If allocation underflows to zero, that
method makes no claims and records the underflow explicitly.

For independent, identically distributed candidate streams with single-stream
crossing probability c, the chance of at least one crossing is
`1-(1-c)^K`. This is a separate independence calculation, not needed for either
validity argument. It is used only for IID independent controls. Fully duplicated
candidates have exactly one unique stream and use that reference. Correlated
streams are deliberately not compared with the independence oracle.

The global-average leader is recorded at first discovery with index tie-breaking.
Its probability of being a true-null candidate, both unconditional and conditional
on discovery, is descriptive. Neither is asserted to have the Bonferroni candidate
error guarantee. In particular, a global rejection says some null premise failed;
it does not establish the leader's individual positive-sign probability.

## Exact decisions and frozen protocol

For each time and success count, an integer table stores the equal-mixture
capital numerator over a shared dyadic denominator. The global crossing compares
the exact sum of candidate numerators with an integer threshold calculated from
the exact binary-float alpha. No logarithmic approximation determines the outcome.
The table and the exact individual count thresholds use the same binary q inputs.
An exact maximum-capital gate skips checks only when even the maximum candidate
capital is below the global threshold. Horizons are limited to 256 for this
higher-cost research implementation; cached tables are immutable. Monte Carlo
estimates and the single-stream forward probability references remain floating
calculations, separately from exact crossing decisions.

The predeclared grid is horizons {64,256}, K={1,8,32}, alpha=.05, 1,000 paths
per family and seed19. Families: independent fair streams; all columns copies
of one fair stream; IID correlated rows made by replacing each independent sign
with a common fair row sign with probability .75 (pairwise correlation .5625);
and independent streams with candidate zero p=.65 and all others p=.5. The first
three satisfy the global null, while the signal family retains the conditional
premise for its null candidates. No serial dependence is introduced here.

Retain every individual and global first-crossing time, global descriptive leader,
familywise false and true candidate claim rates, global discovery rates, sample
hashes, exact allocated budget, source/dependency hashes and runtime. Tiny joint
binary-path enumeration checks the global integer comparisons independently with
Fractions. K=1 makes all methods coincide, and fully duplicated global capital
matches the single-stream capital. These controls distinguish dependence effects
from implementation mistakes. Bonferroni attribution and global-average discovery have separate alpha
guarantees under their respective premises; uncorrected monitoring has only
per-candidate guarantees. Selecting any method's success has no asserted joint
guarantee.

```sh
OPENBLAS_NUM_THREADS=1 python experiments/multiple_sequential_lab.py \
  --horizon 256 --n-candidates 32 --out joint.json
```

## Retained findings

The six planned settings completed with 1,000 paths per family (24,000 paths).
All 24 independent/duplicated-reference comparisons were retained; the largest
absolute discrepancy was 2.021 reference Monte Carlo standard errors. Source
SHA256 is `b163db29cdd23fcb73b2cb8b1ee14574ab9ac5710e97135aefd641d6a09559e7`.

At H=256,K=32 under independent null streams, uncorrected any-candidate claims
occurred in .633 of paths, versus .030 Bonferroni and .030 global-average discovery.
Fully duplicated streams produced .034 uncorrected/global discovery and .001
Bonferroni claims. Correlated IID rows produced .309,.010,.022 respectively.
Dependence changes conservatism and power without making either correction's
validity depend on cross-candidate independence under the stated premises.

For one p=.65 signal at H=64,K=32, Bonferroni attributed the signal in .112 of
paths, while global discovery occurred in .119. Eight of the 119 global discoveries
had a true-null descriptive leader: unconditional frequency .008, conditional
fraction .0672. This small-sample conditional fraction has no asserted alpha bound
and is not evidence of violating the global guarantee. The two procedures answer
different questions; reporting .119 as certified winner power would be incorrect.
At H=256,K=32, signal attribution was .872 and global discovery .884, with .026
familywise false Bonferroni claims and .023 global discoveries led by a null.

Private retained evidence is `multiple-sequential-final/h*-k*.json`, `summary.json`,
reproducible `summarize.py`, and PNG/PDF null-control comparisons. Global-leader
outputs remain explicitly descriptive. No candidate or method was chosen from
the grid to create a new uncalibrated procedure.

# Optional stopping for sign evidence

A fixed-sample test does not automatically remain calibrated when repeatedly
checked and stopped at its first favorable result. This synthetic lab compares
that practice with fixed, predeclared betting rules. Binary observations X are
positive indicators with no zero-return ties in this model. It is not a live
market monitor or a claim of investment skill.

## Primary source and conditional assumption

[Howard, Ramdas, McAuliffe and Sekhon, Lemma 1, equation 2.11](https://arxiv.org/pdf/1808.03204)
states Ville's inequality: a nonnegative supermartingale starting at one crosses
1/alpha at any time with probability at most alpha. A related primary discussion
is [Shafer et al., Test Martingales, Bayes Factors and p-Values](https://arxiv.org/pdf/0912.4269).

For a fixed q>.5, define capital

`L_t(q) = (2q)^S_t * [2(1-q)]^(t-S_t)`.

If `pi_t=P(X_t=1 | the entire past)<=.5`, then the conditional expected next
factor is `2(1-q)+2(2q-1)*pi_t <= 1`. This elementary calculation supplies the
supermartingale premise; a fixed average of such capitals preserves it. Thus
fixed q=.65 and the equal mixture q=(.55,.65,.8) have an anytime crossing bound.
The conditional premise is stronger than marginally fair observations and does
not follow from uncorrelated returns. Independence is sufficient but not required.
No same-path tuning of q, weights, method or stopping threshold is performed.
Mixture capital is calculated with log-sum-exp; it is not an average of log
capitals and is not the maximum capital after inspecting outcomes.

The comparator tests the one-sided Bin(t,.5) upper tail at level alpha separately
at each t. Integer tail numerators and the exact binary-float alpha implement
these fixed-time rejection thresholds. Repeatedly checking them has no anytime
guarantee. The deliberately illegal anticipatory bettor chooses q=.8 after
seeing a positive observation and q=.5 after seeing a negative one. Its capital
is 1.6^S_t: the current outcome informed the bet, so the premise fails. It is
retained only as a negative control, never recommended as a valid procedure.

## Frozen protocol

Before broad runs: horizons 64,256,1024; alpha=.05; 2,000 simulated paths per
family; seed11; fixed mixture above. Retain every first-crossing time (0 means
no crossing), deterministic numerical forward probabilities and boundaries.

The six families are IID probabilities .5, .45, .55, .65; a dependent conditional
null with P(next1|last0)=.3, P(next1|last1)=.5, initial probability .375; and a
stationary marginally fair persistent chain with transition probabilities .1
and .9, initial .5. The last family deliberately violates the conditional null.
Calling it marginally fair does not make anytime sign evidence valid for it.
The dependent conditional null shows that non-IID behavior alone is not the
relevant distinction. Initial distributions are explicitly recorded.

For every method a forward dynamic program retains probability mass of paths
that have not previously crossed, indexed by current positive count and last
sign. At each time it removes crossing mass. It handles both IID and Markov
models. This is evaluation of a finite probability law with floating arithmetic,
not a Monte Carlo estimate or certified exact arithmetic. Tiny-path exhaustive
enumeration and independent integer-binomial tests validate its transitions and
boundaries. Monte Carlo rates have MCSE, and the finite-law reference is retained
so apparent sampling discrepancies can be assessed without tuning the protocol.

All four methods use their stated same alpha but are not jointly covered by a
single alpha guarantee. Choosing whichever method crosses would be another
procedure. The valid methods' infinite-time mathematical bound does not depend
on the simulation horizon; numerical routines intentionally limit DP horizons
to 4096, and standalone log-capital counts to the exact floating integer range.
Log calculations propose a boundary, then exact integer comparisons using the
binary-float q and alpha ratios verify the candidate and adjacent count. Since
capital increases with the success count, this yields the exact minimal crossing
count for those inputs. Equal mixture weights are exact rational weights. Fixed
binomial thresholds also use integer arithmetic; DP probability masses and
standalone log-capital values remain floating approximations. Any future zero-tie extension requires
an explicit conditional nonzero-sign premise, not blind row deletion.

Reports include immutable configuration, source hash, NumPy version, canonical
uint8 path/shape hashes, all first-crossing times and per-time crossing masses.
No local-module dependency or optional SciPy installation is required.

```sh
OPENBLAS_NUM_THREADS=1 python experiments/sequential_sign_lab.py \
  --horizon 256 --out sequential.json
```

## Retained protocol findings

All three planned horizons completed, with 36,000 simulated paths overall and
72 method/family/horizon comparisons against finite-law calculations. The largest
absolute Monte Carlo discrepancy was 1.845 reference standard errors. Source
SHA256 is `0ae963b6ff5a8a6c4a03dc900200dc78b71700945fcccc93f9947c8e3421a116`.
The presentation script preserves raw results and clips only tiny probability
roundoff outside [0,1] when forming Monte Carlo variances.

For IID fair signs, the fixed-final-time exact binomial rejection sizes at
H=64,256,1024 were .02997,.04566,.04881. Repeated fixed-time checking crossed with
probabilities .17419,.25996,.34013. Fixed betting crossed with .02749,.04389,.04445,
and the predeclared mixture with .02121,.03097,.03994. These are separate method
results; picking any crossing among methods has not been calibrated at alpha.

The persistent marginally fair chain invalidated the conditional-null premise:
at H=1024, fixed betting crossed with probability .61547 and the mixture .67531.
The valid dependent conditional-null chain instead yielded .00326 and .00555.
Marginal fairness alone does not supply the conditional premise; dependent
observations can satisfy it, as the valid dependent control demonstrates.

There is also a power cost and no universally best fixed alternative. For IID
p=.55, at H=256 the fixed bet crossed with .26803 and mixture .21225, but at
H=1024 the mixture reached .75936 versus .33515 for the fixed bet. The invalid
anticipatory bettor crossed almost surely in ordinary null controls and is never
a valid competitor. The deliberately naive repeated test's higher detection
rate cannot be compared as if it had controlled the same false-alarm budget.

Private evidence includes `sequential-sign-final/h*.json`, `summary.json`, the
reproducible `analyze_and_plot.py`, and PNG/PDF plots of crossing probabilities
and cumulative first-crossing distributions. The plots separate the invalid
persistent-chain control and identify the three planned horizons.

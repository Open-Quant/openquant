# Selection-aware sign probability intervals

This optional research utility estimates the probability of a positive return
conditional on a nonzero return. It does not estimate expected returns, Sharpe
ratios, profitability, or investment skill. Install the separately pinned
`requirements-selection-intervals.txt`; production manifests remain unchanged.

## Construction and assumptions

For a fixed candidate j, let N be the number of nonzero observations and X the
positive count. Under IID row vectors, conditional on N, X is binomial with
parameter p=P(R>0 | R!=0). Across-candidate dependence is unrestricted. A target
requires positive population probability of a nonzero observation. Sample N=0
returns [0,1] and an undefined estimate; an entirely zero population has an
undefined target, not a covered probability estimate.

The lower endpoint solves P_p(Bin(N,p)>=X)=t and the upper endpoint solves
P_p(Bin(N,p)<=X)=t, with t=alpha/(2K). They use inverse beta tails; X=0 and X=N
use closed forms. The construction and endpoint formulas are verified in
[Thulin, section 2.1](https://arxiv.org/pdf/1303.1288). The original
[Clopper–Pearson 1934 publisher record](https://doi.org/10.1093/biomet/26.4.404)
is accessible but its full text was not read. “Exact” means at least nominal
statistical coverage, not exactly nominal coverage or exact floating arithmetic.

For fixed K, the probability that any interval misses is at most the sum of its
marginal failure probabilities, at most alpha. This yields a valid interval for
an arbitrarily selected member of that fixed family. No independence between
candidate estimates is needed: see [Stark, section 3.1](https://statistics.berkeley.edu/sites/default/files/tech-reports/417.pdf).
Conditioning on a random subset having nonempty observations need not retain
nominal coverage. Reports therefore retain unconditional selected-miscoverage
frequency across all repetitions (no selection makes no assertion), and label
coverage conditional on selection as descriptive. A separately generated audit
is independent of the frozen chosen candidate; its interval has the ordinary
single-candidate guarantee under IID audit rows.

## Numerical contract

`clopper_pearson(positive,n,alpha=...,family_size=...)` accepts failure probability
directly, avoiding cancellation from `1-alpha`. Exact rational comparison rounds
the per-tail binary-float budget downward so 2Kt<=alpha. Tail underflow returns
[0,1] with `tail_underflow`; failed, nonfinite, reversed or out-of-support inverse
beta endpoints return [0,1] with `numerical_fallback`. All positive subnormal
tail budgets also fail closed: SciPy can return finite but inaccurate inverse
beta values there, so finiteness is insufficient. Bounds are rounded outward
one floating step; this is not certified interval arithmetic. SciPy 1.17 can
return NaN for `betainccinv(2,9,smallest_positive_float)`, explicitly tested.
Counts and family size must be integers, not booleans. Numerical qualification is
limited to N<=1,000,000; larger integer counts return [0,1] with
`unsupported_count` before calling floating beta routines. This avoids overflow
and misleading precision at huge counts. It is a deliberate supported range,
not a statistical restriction of CP. No automatic observed-path
deduplication or exclusion of empty candidates reduces the correction size.

## Frozen protocol

Before broad simulations, the protocol fixes 2,000 repetitions per setting,
alpha=.05, seed=7, and (sample size, reported family size) pairs (32,1), (64,8),
(64,32), (256,32). All eight families run in each setting:

- IID fair signs, with independent candidates.
- Fixed pairwise duplicated candidates, retaining the full correction size.
- A common row sign replaces independent signs with probability .75 per column;
  the resulting pairwise sign correlation is .75 squared, or .5625.
- IID fair signs with candidate tie probabilities equally spaced from 0 to .95.
- One candidate with positive-sign probability .65, the remainder .5.
- A single random row repeated across time: deliberate IID violation in both
  discovery and audit. Independence of two samples does not repair either sample.
- A hidden pool of H=16K candidates searched on discovery data, reporting only
  the top K. Applying correction K violates the fixed-reported-family premise.
  The fourth full-pool correction H is valid because all H functions are fixed
  before seeing discovery data. Merely logging the number of realized adaptive
  queries does not in general make a data-dependent function family fixed.
- All observations tied: no population conditional target or selected candidate.

The selector maximizes X/N among nonempty candidates using exact integer
cross-products, breaking score ties by original index. No minimum-N filter is
tuned after results. Reported hidden-pool IDs are retained. Discovery selection
is frozen before generating audit rows. Each sample has equal length; this is a
comparison of inference procedures, not an equal-total-data efficiency claim.
Methods are naive selected CP, reported-family Bonferroni CP, independent-holdout
CP, and full-pool CP. Their nominal guarantees are separate, not a joint guarantee
across methods. Repeated-row and hidden-family failures are preserved.

`iid_max_coverage` evaluates the finite exact distribution of the maximum of H
independent Bin(N,p) counts via F(k)^H-F(k-1)^H. It is a numerical finite-sum
oracle, distinct from Monte Carlo and from the empirical conditional selection
problem with ties. Independent integer enumeration tests validate it at p=.5.

Every report records raw intervals/counts, selected original IDs, canonical int8 data/shape hashes, config,
NumPy/SciPy versions, source hash, wall time, coverage MCSE and Wilson intervals,
widths, fallback counts, no-selection counts and selected-N histograms. Monte
Carlo evidence neither proves guarantees nor validates real-world assumptions.

Run, for example:

```sh
OPENBLAS_NUM_THREADS=1 python experiments/selection_interval_lab.py \
  --repetitions 2000 --sample-size 64 --n-candidates 32 --out report.json
```

## Frozen-grid findings

Four settings completed with 2,000 repetitions per family (64,000 total).
Retained private reports are in `selection-interval-final/`, all generated from
source SHA256 `b0565e219e3aaf55a8562fa93a61a34695f8c6060b6882d19c8456011e4e1f94`.
No ordinary-protocol numerical fallback occurred. Representative IID null
coverage for N=64,K=32 was .586 naive, .991 Bonferroni and .9695 holdout;
for N=256,K=32 it was .5475, .9775 and .960 respectively. The finite-sum oracle
predicts naive coverage .58943 and .53292 for those two configurations.

At N=256,K=32 the hidden-family reported-K correction covered .713 versus .982
for full-pool correction and .962 for holdout. Every repeated-row configuration
had zero coverage for every method. Independent discovery and audit samples do
not make observations within either sample independent. All-tie cases made no
selection in all repetitions; their conditional coverage remains undefined.

Tie heterogeneity produced a selection cost: at N=64,K=32, selected nonzero N
had median 8; at N=256,K=32, median 34. The maximum empirical fraction favors
small effective samples; nominal procedures can consequently yield very wide
intervals. Duplicates and cross-candidate correlation reduced naive selection
bias in these controls but did not restore nominal coverage. No findings justify
assuming independence or sign predictability in market data.

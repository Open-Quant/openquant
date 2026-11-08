# Selection and audit under a fixed observation budget

This follow-up asks what a separate audit costs in selection information. It
compares a full-sample simultaneous CP interval with discovery on half the rows
followed by a single-candidate CP interval on independent audit rows. The row
budget is equal; computation is not. The full method uses NK candidate-row
evaluations, while the split method uses (N/2)K+N/2, evaluating only the frozen
winner on audit rows. No financial transaction or acquisition costs are modeled.

The model is fixed and narrow: independent candidates, IID Bernoulli positive
signs, no ties in returns. Candidate zero has probability p; the other K−1 have
probability .5. The maximum positive count wins, and candidate zero wins score
ties. These are conditional sign probabilities, not expected returns. This is
a numerical evaluation of a finite probability model, not Monte Carlo and not
an assertion that either procedure universally wins.

## Finite sums

Let b_p(s) and F_p(s) be Binomial(n,p) mass and CDF, with F_p(−1)=0.
The joint mass that candidate zero wins with count s is

`b_p(s) * F_0.5(s) ** (K-1)`.

The joint mass that another candidate wins with count m is

`[F_0.5(m) ** (K-1) - F_0.5(m-1) ** (K-1)] * F_p(m-1)`.

The strict second CDF implements the score tie rule. For K=1, the second mass
is zero explicitly, avoiding zero-to-zero ambiguity. These formulas follow
from independence and enumeration; they are not imported selection theorems.
Small joint-count exhaustive enumeration tests provide an independent check.

Full-sample metrics weight the CP interval at each winning count by those joint
masses. Split-sample metrics first compute the selection probability using N/2
rows, then mix independent Bin(N/2,p) and Bin(N/2,.5) audit distributions with
those selection probabilities. The inherited [interval construction](SELECTION_INTERVAL_LAB.md)
uses two-sided CP intervals with family correction K for the full sample and
correction 1 for audit. Source equations are documented in
[Thulin §2.1](https://arxiv.org/pdf/1303.1288), and fixed-family coverage uses
[Stark §3.1](https://statistics.berkeley.edu/sites/default/files/tech-reports/417.pdf).

A positive claim means strictly `lower > .5`, using these two-sided intervals;
this is not a separately optimized one-sided test. Reports distinguish total
claim probability, claims about a candidate with p>.5, and false claims about
a candidate with p<=.5. Miscoverage concerns the selected candidate's actual
probability, not merely whether .5 was excluded. Separate method guarantees do
not imply joint coverage across both methods. No observed grid extremum is a
newly calibrated procedure.

## Predeclared grid and validation

The grid is N in {32,64,128,256,512,1024}, K in {1,8,32,128}, and designated
p in {.5,.55,.65,.8}, with alpha=.05: 96 settings including an all-null control.
Each records both methods, interval width, selection probability, expected
selected sign probability, claims, miscoverage and candidate evaluations.
The implementation accepts probabilities in [0,1], even positive row budgets,
and at most one million rows or candidates; these are numerical scope limits.
Its CP numerical contract is inherited from the qualified interval utility.
Mass conservation and nonnegativity are checked before metrics are computed.
Cached interval tables are immutable; cache contents do not affect reports.

Before interpreting grid results, validate exhaustive small-n candidate counts,
then independently simulate 50,000 repetitions at (N,K,p)=(32,8,.5),
(64,32,.65), and (256,8,.55), seed 1907. The simulation independently generates
discovery count arrays and chosen audit counts, and compares selection frequency,
positive claims, interval widths and miscoverage with the finite sums. Save
Monte Carlo uncertainty and every comparison, including discrepancies. These
validation runs are not an opportunity to tune the grid.

Reports include direct and dependency source hashes, NumPy/SciPy versions,
configuration and runtime. Use the optional pinned dependency file
`requirements-selection-intervals.txt`; production manifests are unchanged.

```sh
OPENBLAS_NUM_THREADS=1 python experiments/selection_budget_lab.py --out budget.json
```

## Retained findings

The 96-setting grid completed. Across both methods its largest selected-target
miscoverage was .049289, below the .05 budget in this particular model/grid.
Independent simulation at the three declared settings produced 36 comparisons;
the largest absolute standardized difference was 2.133 MC standard errors.
All comparisons and raw compressed count arrays were retained privately.

At N=64,K=32,p=.65, full-sample selection chose the designated signal with
probability .67766 versus .44814 for the split. Mean interval widths were .37040
and .35040; positive-claim probabilities were .16124 and .12584. Narrower audit
intervals did not compensate for weaker discovery in this setting.

The opposite direction also occurred: at N=32,K=128,p=.65, split true-positive
claim probability was .02052 versus .01347 for full-sample Bonferroni. Across the
72 non-null settings, split true-positive claims exceeded full-sample claims in
6; split intervals were narrower in 48 of all 96 settings. K=1 illustrates the
pure cost of splitting when selection multiplicity is absent. These descriptive
grid comparisons do not define a data-adaptive choice between procedures.

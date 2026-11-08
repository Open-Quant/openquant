# Structural missing-data bounds and finite-sample uncertainty

Known outcome support is Y in {-1,+1}; every row is counted, including missing records. Propensities are unknown and no restriction is imposed on the missing outcomes. Write a=P(R=1,Y=+1), b=P(R=1,Y=-1), c=1-a-b. The missing positive mass z can be any value in[0,c], so E[Y]=2(a+z)-1 ranges over the sharp identified set [2a-1,1-2b]. Every endpoint and interior value is attainable by assigning that missing mass; the compatible-model helper constructs the associated population and propensities when their conditioning events have positive probability.

This elementary binary specialization follows the missing-distribution identification-region principle in [Manski's original article](https://doi.org/10.1016/j.ijar.2004.10.006). The publisher abstract was accessible; specific formulas above are directly derived here rather than attributed to unread full text. Identification assumes population observable probabilities are known. It differs from uncertainty due to a finite sample.

## Outer confidence set

For N IID rows, observed positive and negative counts each have a marginal binomial law, although they are dependent. Let La and Lb be the lower endpoints of ordinary two-sided Clopper–Pearson intervals at failure alpha. Each lower endpoint exceeds its probability with probability at most alpha/2. By the union bound, both lower bounds hold with probability at least1-alpha, without independence between counts. Then [2La-1,1-2Lb] contains the entire sharp identified set. The code rounds transformed endpoints outward one float and clips them to known support; this is not certified interval arithmetic and inherits the [CP numerical contract](SELECTION_INTERVAL_LAB.md).

Plug-in bounds [2Kplus/N-1,1-2Kminus/N] are estimates of the identified set, not confidence intervals. Their coverage of one particular mean may be high even when coverage of the whole identified set is low. Both quantities are recorded. The observed-only mean is a separate, generally biased point estimator and is undefined when no outcome is observed.

All-missing data give outer bounds[-1,1]. When all outcomes are observed the population identified set collapses to a point, but its finite-sample confidence set still has sampling uncertainty. More rows shrink sampling uncertainty; they do not remove structural width2c when propensities remain unknown.

## Information-set distinction

The earlier missingness study marks HT unavailable at zero propensity. That is a limitation of weighting, not a universal statement of nonidentification. If one knows splus=1,sminus=0 and exact binary support, R itself reveals Y. Here those propensity restrictions are deliberately not known. Model A with(p=.5,splus=1,sminus=.25) and model B with(p=.625,splus=.8,sminus=1/3) share observable probabilities(.5,.125,.375), while their means are0 and.25; both lie in the same identified set[0,.75]. External knowledge of propensities supplies extra information.

## Frozen protocol

N32,128,512,4096; 2,000 replicates; alpha=.05; seed113. Models are the equivalent pair, MCAR with observable(.25,.25,.5), fully observed(.5,.5,0), and all missing(0,0,1). Equivalent models intentionally share the exact same observed count draws, so their bounds and observed-only estimates are identical while their latent targets differ. Other settings use independent declared streams. Every count and interval is retained; scientific timing excludes serialization. This is not an investment or causal-effect claim.

```bash
python experiments/partial_identification_lab.py --out partial-id.json
```

## Retained findings

All20 model/sample-size settings completed. For equivalent A/B the outer mean width decreases from1.2815 at N32 to.8007 at N4096 while structural width remains.75. Plug-in identified-set coverage falls from.3115 to.191; plug-in endpoints are not calibrated confidence limits. Outer identified-set coverage at N4096 was.949 with Monte Carlo SE.00492: this observed value is retained rather than rounded upward or treated as a theorem violation. The finite-sample guarantee is mathematical; the simulation is a noisy check. The equivalent pair shares all observed counts and bounds, yet one mean lies on the identified-set boundary and the other in its interior, so pointwise mean coverage can differ. There are40,000 model-replicate records but only32,000 distinct generated observable samples because the pair is deliberately coupled. Full-observation uncertainty shrinks, and all-missing bounds stay[-1,1].

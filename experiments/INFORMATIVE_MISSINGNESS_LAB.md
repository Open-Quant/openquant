# Informative missingness and identification

Y takes values +1 or -1 with positive probability p. Its observation indicator R depends on Y through splus and sminus. The population target is E[Y]=2p-1. All N rows, including missing records, remain in the denominator of the Horvitz–Thompson mean. The full-data mean is an inaccessible diagnostic control in an actual missing-data problem.

The observed-only mean targets [p splus-(1-p)sminus]/[p splus+(1-p)sminus], conditional on at least one observation. With positive, externally known propensities the HT estimator is sum(RY/s(Y))/N. Direct conditioning gives expectation 2p-1 and variance [p/splus+(1-p)/sminus-(2p-1)^2]/N in this IID superpopulation model. This includes outcome variation as well as observation variation; it is not the conditional finite-population survey variance. The normalized Hájek ratio divides by sum(R/s(Y)); it is not generally finite-sample unbiased. [PSU STAT506 §3.3](https://online.stat.psu.edu/stat506/Lesson03) provides the original teaching derivation of HT weighting and unbiasedness. The superpopulation variance here follows the stated direct calculation.

Propensities are oracle information, not inferred from unobserved outcomes. At a zero propensity, weighting is marked unavailable rather than claiming recovery. The implementation conservatively excludes any zero propensity even if the corresponding outcome happens to have zero population mass. With positive propensities and no observed rows HT is defined as zero; observed-only and Hájek remain undefined. Conditional summaries of available estimates are labeled. No confidence interval or investment claim is made.

## Observable equivalence

Model A: p=.5,splus=1,sminus=.25. Model B: p=.625,splus=.8,sminus=1/3. Both give observable probabilities (+1 observed,.5),(-1 observed,.125),(missing,.375). Their population means are0 and.25. Thus identical observable distributions alone cannot distinguish the target. Knowing the correct external propensities changes the information available; it does not make the means identifiable from observed outcomes alone. Separate simulations need not give identical finite sample counts despite equality of their laws.

## Frozen protocol

Before retained simulations: N32,128,512; 5,000 independent replicates per setting; seed79. Six models: MCAR(.5,.5,.5), the equivalent pair, near-zero negative propensity(.5,1,.01), zero negative propensity(.5,1,0), and all missing(.5,0,0). Every count and estimate is retained. Methods are observed-only, oracle HT, oracle Hájek and full-data control. Quantities are abstract observations, not market returns or a trading rule. No propensity estimation, outcome model or weight clipping is fitted.

HT can leave the outcome range [-1,1] and incur large variance near zero propensity. Hájek normalizes into that range but can be biased. Means, variance, mean Monte Carlo standard errors and availability are reported; these refer to this bounded fixed-propensity grid, not arbitrary vanishing-propensity regimes. Numerical scope is N<=1e6 and each positive propensity>=1e-12. Integer counts are normalized before arithmetic to prevent unsigned wraparound.

```bash
python experiments/informative_missingness_lab.py --out missingness.json
```

## Retained findings

All18 settings completed. At N32, the equivalent observable models produced observed-only means .6003 and .5972, while their true means are0 and.25. Oracle HT means were .0006 and .2467; Hájek means .0484 and .2716 retain finite-sample normalization bias. Under near-zero negative propensity, observed-only mean was .9804 and Hájek .7439 despite true mean0; HT mean .0052 had much larger variance. Zero-propensity weighting remained unavailable, and all-missing observed means remained undefined. These are finite simulation summaries with Monte Carlo uncertainty, not guarantees that realized weighted estimates are close to truth. All observations, failures and negative controls remain retained.

### Positivity is specific to the weighting construction

HT unavailability does not imply that every possible identification argument fails. In the zero-propensity model here, splus=1 and sminus=0 are treated as known and Y has exactly binary support. Then R=1 identifies Y=+1 and R=0 identifies Y=-1: observation status itself reveals the outcome. HT division by a zero propensity is unavailable, but an estimator using that additional structural knowledge can identify the mean. The earlier equivalent-model example instead concerns unknown propensities and shows nonidentification from the common observed distribution alone. These are different information sets.

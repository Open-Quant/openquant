# Finite-variance mean robustness

This standalone NumPy experiment estimates the population arithmetic mean, not sign probability or investment performance. IID observations and a known variance upper bound V are assumptions, not estimates learned for free from the sample.

For n rows, Chebyshev gives radius sqrt(V/(n alpha)) around the sample mean. For median-of-means, split the first B floor(n/B) rows into B disjoint equally sized blocks, with B odd and chosen before data. Each block mean lies outside radius 2 sqrt(V/m) with probability at most 1/4. Independence bounds median failure by P(Bin(B,1/4)>B/2). The smallest qualifying odd B is selected by exact integer comparison with the binary-float alpha. Unused tail rows are recorded and ignored only by the median. If n<B the median interval is unavailable, not silently replaced.

This follows the argument in [Lugosi and Mendelson, section 2.1](https://arxiv.org/abs/1906.04280); the exact binomial tail replaces the looser exponential relaxation. It does not assert optimal constants or dominance over Chebyshev. Gaussian known-variance intervals are a reference whose exact calibration requires Gaussian data. They are not certified under arbitrary finite variance.

Clipping at [-2,2] supplies a bounded-variable Hoeffding interval for E[clip(X)], using range length 4. It need not cover E[X]. The skew Bernoulli control has original mean zero but clipped expectation .02-.99*.01/sqrt(.01*.99). Both targets are retained. Dependence breaks the IID premise for clipping as well.

## Frozen protocol

Before retained runs: n=64,256,1024; alpha=.05,.001; 5,000 independent replicates; seed41. Families are standard Gaussian; t3/sqrt(3) (variance1, infinite fourth moment); standardized Bernoulli(.01); one Gaussian draw repeated within each sample; and Gaussian variance4 while the assumed bound remains1. Last two are deliberate violations. Clipping retains its bounded IID guarantee under the variance-misspecification control, but not repeated rows.

All intervals have separate marginal claims, not simultaneous coverage across procedures/settings. Fixed widths can be conservative. Empirical non-failure is not proof. Reports retain every interval, input hashes, configuration, block sizes/remainders, availability, both coverage targets and Monte Carlo uncertainty. Numerical qualification is alpha in [1e-12,1) and V in [1e-200,1e200]; nonfinite arithmetic fails explicitly. Arithmetic is not certified interval arithmetic.

```bash
python experiments/mean_robustness_lab.py --sample-size 256 --alpha .05 --out mean.json
```

## Retained findings and focused followup

Six declared settings completed. At n=1024, median-of-means width .376 exceeds Chebyshev .280 for alpha=.05; at alpha=.001 the ordering reverses (.718 versus1.976). Repeated rows strongly violate coverage. The false variance bound can remain hidden by conservative procedures: coverage above nominal under that control does not validate its premise.

All initial clipped intervals covered both targets despite their difference; conservatism concealed the bias. A separately declared post-grid diagnostic used n=32768,500 replicates,seed73 and only rare_skew. Clipped intervals covered their own target in all500 runs and the original mean in none. Their width .06002 is smaller than the target separation .07950. Original Chebyshev and MOM intervals covered in all runs; Gaussian-reference coverage was .952. This followup illustrates distinct estimands and is retained alongside, not substituted for, the original grid. Zero empirical failures do not imply zero failure probability.

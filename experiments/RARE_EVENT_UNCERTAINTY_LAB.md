# Rare-event blindness in empirical uncertainty

This finite-law study takes IID X=A Bernoulli(p), with A=1/p fixed by the simulation designer so E[X]=1. The amplitude is treated as known by the support-based interval; this is additional information, not something inferred from an all-zero sample. Known-variance Chebyshev similarly receives the true variance A²p(1-p). These are assumption-rich controls, not equally informed competitors or investment procedures.

Given K positive observations, the sample mean is AK/n, unbiased sample variance is A²K(n-K)/(n(n-1)), and estimated mean standard error is A sqrt(K(n-K)/(n²(n-1))). Substituting this variance into Chebyshev does not inherit its guarantee. Gaussian and Student plugin-SE intervals are references whose exact premises do not hold for these Bernoulli data.

Nonparametric resampling draws from the empirical distribution; thus bootstrap positive count is exactly Bin(n,K/n) conditional on K. At the default alpha=.05 the percentile interval uses its .025 and .975 inverse-CDF quantiles (smallest count with CDF at least the requested probability), scaled by A/n. There is no Monte Carlo bootstrap error here. The empirical approximation can still fail. See [CMU's original bootstrap lecture](https://www.stat.cmu.edu/~cshalizi/dm/20/lectures/21/lecture-21.html) for the resampling construction and distinction between simulation and approximation errors. This study does not condemn all bootstrap methods or asymptotic settings.

The known-two-point-support interval applies the [existing CP construction](SELECTION_INTERVAL_LAB.md) to p then scales endpoints by known A. Its guarantee concerns this fixed IID model and fixed confidence level. No zeros are dropped: zeros are observed failures, unlike nonzero-sign conditioning in other labs.

Coverage, width and zero-width probability are summed over K=0..n with Bin(n,p) masses. These are finite probability calculations in floating arithmetic, not simulated estimates or certified exact arithmetic. All counts, masses, intervals and CP statuses are retained. At K=0, empirical variance and every bootstrap resample are zero, so plugin and percentile intervals collapse to [0,0] although the true mean is1. That event has probability (1-p)^n. CP retains uncertainty using the known support; a finite observed support alone would not justify it.

## Predeclared protocol

n=32,128,512,2048; p=.001,.01,.05,.5; alpha=.05; all16 settings retained. A changes with p solely to hold the mean constant. The p=.5 control tests a nonrare regime. Supported numerical scope n2..10000, p in[1e-8,1-1e-8], alpha in[1e-8,.5]. NumPy and SciPy versions and the frozen CP dependency hash are recorded. No data-adaptive selection among procedures is calibrated.

```bash
python experiments/rare_event_uncertainty_lab.py --out rare-events.json
```

Very small binomial masses can underflow in floating arithmetic. The no-event log probability and explicit underflow flag distinguish numerical zero from mathematical impossibility.

## Retained finite-grid findings

At p=.001,n128, no-event probability is .879797. Plugin Chebyshev covers .120203 and percentile bootstrap .120193, compared with .992524 for known-support CP. At n2048 the no-event probability remains .128861; bootstrap coverage is .869877 versus CP .981698. The p=.5 control is retained: bootstrap coverage ranges approximately .9499 to .9584 across this grid. This is a finite-sample failure example under explicit rare-event regimes, not evidence that all bootstrap intervals fail. Known-variance Chebyshev and known-support CP benefit from information unavailable to a data-only method.

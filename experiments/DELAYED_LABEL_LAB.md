# Label maturity at model-fit time

A historical archive can contain labels that had not matured when a forecast was made. This standalone example isolates label timing using an intercept-only forecast: the fitted mean of the n most recent training labels. It is not a trading strategy or a test of OpenQuant's native model pipeline.

All clocks are integer ticks. Innovations are latent generating/scoring variables; the learner receives only released labels, not a raw innovation feed from which it could reconstruct unreleased labels. The cutoff follows the time-T innovation, but this fixed learner uses the released-label interface and does not reconstruct delayed labels from another raw feed. This does not assert that unreleased labels are physically unknowable to every possible observer. Event occurrence and provider availability are distinct contracts. Innovations epsilon_t are independent N(0,1); label Y_t=sum_{j=1}^H epsilon_(t+j) is released at t+H+lag. At cutoff T, the target is Y_T, supported strictly after T. A safe training decision must satisfy t<T and release<=T (inclusive). `eligible_training` sorts distinct decision IDs and returns the last n eligible IDs, or None when history is insufficient. Duplicate decisions, mismatched arrays and release before decision are rejected. The helper trusts supplied release clocks; it does not infer whether a real publication was actually available.

The naive complete-archive comparator takes the last n decisions t<T even when labels had not been released. The zero predictor is the optimal predictor under the stipulated IID future model. Safe fitting adds estimation variance; this matters even though safe chronology is correct. Overlapping training labels are correlated, and the reference explicitly includes that covariance.

Every forecast is a linear combination of innovations. If w is its weight vector and a is the future target vector, prediction variance is w'w, target covariance w'a, and expected squared error is (w-a)'(w-a). These identities follow directly from independent standard Gaussian innovations. For either fitted mean,

V=[nH+2 sum_{d=1}^{min(n-1,H-1)} (n-d)(H-d)]/n².

The safe covariance with its future target is zero. Naive covariance is sum_{r=1}^{H-1} min(n,r)/n. Thus safe MSE=H+V, naive MSE=H+V-2C, and zero MSE=H. The naive MSE can improve falsely through shared future innovations, but need not beat zero in every configuration. Release delay violations and future-content leakage are different: with H=1,lag=2 the naive labels violate release timing but contain no innovations after cutoff. Both audit counts are retained.

Gaussian squared-error standard deviation is sqrt(2)*MSE. Paired differences against zero use the joint Gaussian fourth-moment identity Var(e1²-e0²)=2v1²+2v0²-4Cov(e1,e0)². These are model calculations, not assumptions about market data. Analytical values are checked independently through overlap counting and covariance matrices.

## Frozen protocol

H1,4,16; n=H or4H; lag0 or2; cutoff0; 10,000 independent paths per setting; seed101. There is one forecast per path. Monte Carlo standard errors use independent paths, not overlapping decisions as if they were independent observations. Every method shares the same innovations within a setting. Reports retain complete training IDs/release clocks, innovation weights, target and prediction arrays, squared errors, input hashes and exact model references. Sufficient history is provided for this grid; insufficient-history behavior is tested separately without fallback. All methods and settings are fixed before results.

Numerical design limits H,n,lag<=4096; integer inputs are converted to Python integers before products to avoid unsigned overflow. No clock is interpreted as a calendar timestamp, and no model is fitted with future truth by the safe method.

```bash
python experiments/delayed_label_lab.py --out delayed-labels.json
```

The construction includes one extra unused historical innovation with zero prediction/target weight. It changes neither forecast nor reference moments.

## Retained findings

All12 settings completed. At H16,n16, expected MSE is16 for zero,26.6875 for the mature-only fitted mean and11.6875 for the complete-archive mean. The latter advantage comes from positive covariance with future target innovations. At H4,n16, the archive mean still has MSE4.171875, worse than zero's4 despite the leakage advantage over safe fitting. H1,lag0 arms coincide. Release lag changes eligible training IDs but not their IID estimation variance; release violations need not imply future-content overlap. Across36 marginal MSE comparisons the largest deviation was2.039 analytical Monte Carlo standard errors, retained without tuning. Paired errors and reference standard errors are included separately.

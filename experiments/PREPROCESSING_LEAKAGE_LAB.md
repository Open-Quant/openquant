# PCA preprocessing information-boundary lab

Status: private experimental research scaffold. Owner: OpenQuant maintainers.
Audience: researchers checking what a validation pipeline is allowed to observe.

## Question and scope

Can a pipeline use future information despite having no overlapping training and
holdout labels? Yes: fitting PCA to future feature rows changes the retained
subspace. This lab isolates that mechanism. Both ridge regressions use exactly the
same training labels, fixed penalty, and chronological split. Only the rows used
to learn the PCA transform differ.

This is an additive experiment, not a change to OpenQuant's production PCA,
cross-validation, or evaluation APIs. Formal production behavior is unchanged;
this guide records the experimental protocol. It complements the existing
[selection-validation lab](VALIDATION_LAB.md), which addresses candidate selection.

The scikit-learn maintainers recommend learning transformations on training data
and applying the frozen transformation to evaluation data; their guidance
explicitly includes PCA. This experiment follows that information contract.
[Source: scikit-learn common pitfalls](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage).

Global fitting here uses **future covariates, never future labels**. If access to
unlabeled evaluation covariates is part of a declared transductive task, it is a
different task. It is not equivalent to freezing a model before those rows exist.
Better scores do not by themselves establish leakage, and worse scores do not
establish absence of leakage. The observable information boundary is the criterion.

## Run

Only NumPy is needed; no compiled OpenQuant extension, network, market data, or
book excerpts are used.

```bash
python experiments/preprocessing_leakage_lab.py \
  --repetitions 200 --out /tmp/openquant-preprocessing.json
python -m pytest python/tests/test_preprocessing_leakage_lab.py -q
```

Defaults: 128 training rows, 128 holdout rows, six features, ridge penalty 1,
200 independent replications, seed 19. Each family's stream uses its fixed family
index and replication number; changing the requested family order does not change
its generated datasets. Random draws are reproducible in the recorded NumPy
version; cross-version or cross-BLAS bitwise equality is not promised.

## Frozen protocol

- Rows are independent Gaussian point observations. Labels resolve on the same
  row, so no training label overlaps any holdout label. There is no actual
  purging call: the example establishes that zero overlap does not constrain a
  preprocessing fit, not that any particular purge implementation failed.
- Features have mean zero. The target is feature 0 plus independent Gaussian noise
  with standard deviation 0.5, except in the stationary null where it is noise only.
- Fit centered PCA without whitening or feature standardization. The main
  comparison retains one component; the control retains all six.
- **Training-only:** PCA mean and component directions are fitted on the training
  prefix and frozen before transforming the suffix.
- **Future-features comparator:** PCA is fitted on the concatenated training and
  holdout features. Future target values are never given to the fitter.
- Fit ridge on training transformed features with an unpenalized intercept. The
  objective is sum of squared errors plus `ridge * ||coefficient||²`; it is not a
  mean-error objective with the same numeric penalty.
- Score both methods on the same holdout using mean squared prediction error.
  Subtract training-only MSE from future-features MSE within each replication.
  Negative differences favor future access. No hyperparameters or families are
  chosen after observing a report.
- The all-component control has isotropic ridge and an intercept, so an
  orthogonal PCA basis change must leave predictions unchanged up to numerical
  precision. This is an algebraic control, not another method to optimize.

The first two feature standard deviations are:

| Family | Training | Holdout | Target |
|---|---|---|---|
| `stationary_signal` | 2, 1 | 2, 1 | Feature 0 + noise |
| `stationary_null` | 2, 1 | 2, 1 | Independent noise |
| `signal_variance_rises` | 1, 2 | 4, 2 | Feature 0 + noise |
| `nuisance_variance_rises` | 2, 1 | 2, 4 | Feature 0 + noise |

Other features have standard deviation 0.5 throughout. These are intentionally
strong mechanism controls, not estimated market regimes. The two shift controls
are deliberately asymmetric: future access can uncover the predictive direction
or displace it with nuisance variance.

## Evidence and interpretation

The JSON retains every replication, original-feature regression coefficients and
intercepts, both MSEs, paired differences, and the full-component maximum prediction
difference. It hashes each generated feature and target array including its shape
and canonical little-endian float64 bytes. CLI output also records the source hash,
Git revision, NumPy/Python/platform versions, config, and elapsed time.

For each metric, the Monte Carlo standard error is sample standard deviation across
independent replications divided by the square root of their count. The difference
uses **paired** errors from the same generated path. This measures simulation
precision; it is neither a strategy confidence interval nor a guarantee about
financial returns. The better-fraction statistic is descriptive; values near zero
for the full-component control can reflect roundoff, not evidence of superiority.

A first default run (NumPy 2.3.5, seed 19, 200 replications) gave these paired mean
MSE differences and Monte Carlo standard errors:

| Family | Future minus training MSE | MCSE |
|---|---:|---:|
| Stationary signal | -0.00342 | 0.00064 |
| Stationary null | +0.000040 | 0.000013 |
| Signal variance rises | -15.69066 | 0.14003 |
| Nuisance variance rises | +5.69061 | 0.16353 |

The all-component maximum prediction difference was below `1.8e-14`. The stationary
null did **not** show a useful predictive benefit; the nuisance-shift comparator
was much worse. These results are retained, not tuned away. Only the synthetic
mechanism and declared parameters support these observations. They do not show
that a global PCA fit invariably raises a backtest score.

## Acceptance checks and limitations

Tests use a hand-derived one-dimensional ridge solution, a diagonal covariance
example where future features remove the predictive direction, and a direct
original-feature ridge solve for the all-component control. Mutating only future
features leaves training-only fitted predictions on fixed training rows identical,
while it changes the global-fit model. Report replay, stable family streams,
finite output, invalid dimensions and nonrepresentable arithmetic are covered.

There is no serial dependence, delayed publication, feature imputation, event
purging, transaction cost, trading policy or adaptive model search in this lab.
Adding those would require a separately specified experiment. A fixed number of
PCA components also need not be economically useful; it is fixed here to expose
the boundary rather than improve a strategy. The external-source claim above is
limited to preprocessing practice; all simulation code and controls are original.

A separate confirmation used seed 83 and 2,000 replications per family without
changing any family or model setting. Paired differences were -0.00232 (MCSE
0.00016), +0.000023 (0.000004), -15.79422 (0.04554), and +5.76001 (0.05435), in
family-table order. The direction of both covariance-shift controls was preserved;
full-component differences remained below `2.5e-14`. This is additional synthetic
precision, not external validation. The seed-19 results above remain recorded.

---
title: Select a model and evaluate a chronological holdout
description: Purged inner selection, training-only preprocessing, explicit refitting and one untouched classification holdout.
status: authored
last_authored: '2026-11-07'
audience:
  - quant-dev
  - platform-engineering
sidebar:
  order: 4
---

This walkthrough connects public Python APIs into a complete selection/refit/evaluation
procedure. It needs NumPy and the installed OpenQuant extension, without scikit-learn.
The [research journey](/workflows/research-journey/) traces a fixed baseline through
walk-forward folds; this example adds a small parameter search before a final chronological
holdout. Both report classification evidence, not trading P&L.

## Fix the boundary before searching

Reserve the last 20 rows before inspecting candidate scores. Each feature is a function of
its own observation time; the synthetic label uses the next observation. Closed label spans
overlap adjacent rows, so the final development label crossing the holdout boundary is
excluded. The retained 79 development rows, their labels, weights and spans stay aligned.
The fixture constructs the complete synthetic series for convenience; no holdout label or
feature enters the search, preprocessing fits or final refit.

The tiny classifier estimates its standardization and class centers inside each `fit` call.
The row-ID column is only for audit assertions and is never a predictive feature. Unit
weights are fixed by the protocol; they are not estimated using future labels. Real feature
construction, label availability and weighting need their own causal review.

## Search, refit, then evaluate once

```python
import numpy as np
from openquant import cross_validation as cv, hyperparameter_tuning as ht

# Fixed synthetic protocol, chosen before the final 20 observations are scored.
n = 100
clock = np.arange(n + 1)
signal = np.sin(clock / 4.0) + 0.15 * np.cos(clock * 1.7)
X = np.column_stack([signal[:-1], np.arange(n)])  # second column audits row identity only
y = (signal[1:] > 0).astype(float)
t0 = np.arange(n) * 2
t1 = t0 + 3  # closed label spans overlap adjacent rows
weights = np.ones(n)  # fixed, not estimated from future outcomes
holdout = np.arange(80, n)
development = np.flatnonzero((np.arange(n) < 80) & (t1 < t0[holdout[0]]))
assert len(development) == 79  # row79 touches/crosses the holdout boundary
assert t1[development].max() < t0[holdout[0]]
fit_rows = []


class StandardizedCentroids:
    def __init__(self, params):
        self.temperature = params["temperature"]

    def fit(self, x, labels, sample_weight=None):
        ids = x[:, 1].astype(int)
        assert set(ids).isdisjoint(holdout)
        fit_rows.append(ids.copy())
        # Fit preprocessing on this call's training rows, never all development rows.
        self.mean = float(x[:, 0].mean())
        self.scale = max(float(x[:, 0].std()), 1e-12)
        z = (x[:, 0] - self.mean) / self.scale
        self.centers = []
        for label in (0, 1):
            mask = labels == label
            assert mask.any()
            self.centers.append(float(np.average(z[mask], weights=sample_weight[mask])))
        return self

    def predict_proba(self, x):
        z = (x[:, 0] - self.mean) / self.scale
        evidence = ((z - self.centers[0]) ** 2 - (z - self.centers[1]) ** 2)
        p = 1 / (1 + np.exp(-np.clip(evidence / self.temperature, -30, 30)))
        return np.column_stack([1 - p, p])


splits = cv.purged_kfold_splits(t0[development], t1[development], 4, 0.02)
for train, test in splits:
    assert len(train) and len(test)
    for i in development[train]:
        for j in development[test]:
            assert t1[i] < t0[j] or t1[j] < t0[i]

candidates = [{"temperature": value} for value in (0.5, 1.0, 2.0)]
search = ht.purged_search(
    StandardizedCentroids, candidates, X[development], y[development],
    t0[development], t1[development], n_splits=4, pct_embargo=0.02,
    scoring="neg_log_loss", sample_weight=weights[development],
)
assert all(np.isfinite(trial["fold_scores"]).all() for trial in search["trials"])
assert len(fit_rows) == len(candidates) * len(splits)
for actual, (train, _) in zip(fit_rows, splits * len(candidates)):
    assert np.array_equal(actual, development[train])

# Selection returns parameters, not a fitted estimator. Refit once, then score once.
model = StandardizedCentroids(search["best_params"]).fit(
    X[development], y[development], weights[development],
)
assert np.array_equal(fit_rows[-1], development)
probabilities = model.predict_proba(X[holdout])[:, 1]
score = ht.classification_score(y[holdout], probabilities, weights[holdout], "neg_log_loss")
assert np.isfinite(score)
print("development/holdout:", len(development), len(holdout))
print("candidate fits/final refit:", len(fit_rows) - 1, 1)
print("selected temperature:", search["best_params"]["temperature"])
print("holdout negative log loss:", round(score, 4))
```

```text
development/holdout: 79 20
candidate fits/final refit: 12 1
selected temperature: 0.5
holdout negative log loss: -0.245
```

The assertions verify every actual fitting row against the expected inner splits and exclude
the holdout from all fits. They also verify that no retained training label overlaps an inner
test label. The public splitter applies the configured embargo in addition to purging.
These checks cannot discover future information hidden inside externally supplied features.

## Interpret and preserve the result

`purged_search` fits a fresh estimator for each candidate/fold and averages fold scores with
equal fold weight; it does not pool all test predictions or return a fitted winner. Scores
here are **negative** log loss, so larger is better. Finite exact ties select the later
candidate. Inspect all fold scores before accepting the winner; keep the full trial list,
candidate order, split protocol and any abandoned searches with the result.

The chosen parameters are refitted on all eligible development rows, including newly fitted
preprocessing. Only then is the reserved holdout scored. The inner CV score selected the
winner and is not an independent estimate of its performance. Inner purged K-fold can train
on rows later than an inner test fold; it is not walk-forward validation. Use an inner
chronological procedure if that restriction is part of the study, and keep the same final
holdout boundary.

This synthetic score is a demonstration, not a performance threshold or calibrated
probability claim. The 20 adjacent holdout labels overlap and are not independent trials.
Do not turn this one score into a confidence interval or a Sharpe ratio. Once inspected,
the holdout has been used: changing features, candidates, thresholds or preprocessing in
response requires new evaluation data or a separately specified outer evaluation design.
The example asserts structural separation; a dated protocol and retained execution evidence
are needed to establish how a real study was conducted.

For details, see [hyperparameter tuning](/modules/hyperparameter-tuning/),
[cross-validation](/modules/cross-validation/) and the
[trial identity contract](/workflows/research-notebook-contract/#trial-identity-is-part-of-the-method).

# Synthetic selection-validation lab

Status: experimental research scaffold, not a production metric or investment strategy.
Owner: OpenQuant maintainers. Audience: researchers evaluating a search procedure.

## Why this exists

OpenQuant already exposes probabilistic/deflated Sharpe ratios, minimum track-record
length, a trial registry, purged k-fold, CPCV, walk-forward splits, and null-score
helpers. Adding another score before checking how selection affects it would repeat
existing functionality. `run_flywheel_grid` ranks configurations evaluated on one
dataset. Its leaderboard is useful for exploration; it is not an untouched holdout.

This lab makes selection bias visible before any trading model is fitted. It selects
one candidate using a chronological training prefix, freezes that candidate, and
reports its score on the suffix. A separate comparator deliberately picks the best
candidate on that suffix and is labeled **contaminated**. It is never a valid estimate
of deployable performance.

This experiment addresses candidate selection only. It does not measure leakage from
feature engineering, label overlap, fitting, transaction costs, repeated human reuse of
a holdout, or a changing candidate family. Those remain separate validation obligations.

## Run offline

NumPy is the only runtime dependency; the OpenQuant compiled extension is not imported.
Use the project's environment when available, or any environment with NumPy:

```bash
python experiments/validation_lab.py --out /tmp/openquant-validation.json
python -m pytest python/tests/test_validation_lab.py -q
```

The default is 200 independent replications per family, 252 training observations,
252 holdout observations, 64 candidates and seed 7. A candidate-count stress test:

```bash
for n in 1 8 64 256; do
  python experiments/validation_lab.py --families iid_null --candidates "$n" \
    --repetitions 1000 --out "/tmp/openquant-validation-$n.json"
done
```

`--gap 20` excludes 20 rows between training and holdout. It is a fixed separation,
not label-span purging and not proof of independence. AR(1) generation continues across
that gap. The experiment requires at least four observations in each sample: with
three, even an iid Gaussian sample Sharpe has infinite variance, making an ordinary
finite-variance Monte Carlo standard error inappropriate. Use materially larger
samples and more replications for interpretable precision.

## Experimental contract

- Arrays are `(chronological observations, candidates)`.
- Score is mean divided by **sample** standard deviation. It is **per observation**,
  never annualized; multiplying by square-root-of-time would require assumptions,
  particularly for the serially dependent scenario.
- Selection uses only rows `[0, train_size)`. Holdout rows start at `train_size + gap`.
- Exact score ties select the first column. No holdout-dependent tiebreaker is allowed.
- Constants, nonfinite values, complex inputs, ragged arrays and undersized samples
  fail closed. Scores are scale-normalized before moments to avoid simple overflow.
- Every scenario and replication has its own deterministic random stream. Reordering
  requested scenarios cannot change their samples. NumPy version is recorded because
  cross-version random-distribution reproducibility is not promised.
- Every generated matrix contributes its shape and canonical little-endian float64
  bytes to a scenario SHA-256 hash. Config, source hash, git base, Python/NumPy version,
  and platform are saved. The source hash identifies local code beyond the git base.
- All replication records are retained, including both chosen candidate indices,
  explicit split ranges, and training, honest-holdout and contaminated-holdout scores.

## Six controlled families

All nominal marginal standard deviations are 0.01. They are synthetic mathematical
returns, not calibrated or economically bounded market observations.

| Family | Construction | Diagnostic purpose |
|---|---|---|
| `iid_null` | Independent zero-mean Gaussian columns | Selection alone can inflate the winning training score. |
| `correlated_null` | Common Gaussian factor; pairwise correlation 0.75 | Nominal trial count differs from independent opportunity count. |
| `duplicated_null` | Every independent Gaussian column copied twice | Exact copies add no distinct opportunity; odd counts retain one final unpaired column. |
| `heavy_tailed_null` | Student t with 3 degrees of freedom, scaled to variance 0.0001 | Sample moments become unstable; the theoretical fourth moment is infinite. |
| `serial_null` | Gaussian AR(1), coefficient 0.6, stationary initial state | Zero unconditional mean does not imply temporally independent observations. |
| `positive_drift` | Column 0 has mean 0.001; other Gaussian columns have mean 0 | Check ability to select a known nonzero mean among distractors. |

The last family injects a known mean; it is **not** a learned predictive signal. The
AR family is unconditional-null, not a claim that no strategy can predict AR returns.
All scenarios are generated by original code, not copied from a book or community repo.

## Read the report

Each metric summary provides its replication mean and empirical Monte Carlo standard
error (`sample std / sqrt(repetitions)`). Replications use independent random streams.
This measures simulation precision, not confidence that a real strategy is profitable.
It can be unstable with short samples or few replications. No calibrated p-value or
claim about a strategy's true Sharpe is made.

`selection_optimism` is training score minus honest-holdout score, not the contaminated
score gap. `holdout_nonpositive_fraction` is descriptive and is not a probability of
backtest overfitting (PBO). The known-signal selection fraction appears only in the
positive-drift scenario. No result is called a promotion decision.

The wall clock covers generation, selection, scoring, hashing, and summary construction.
It excludes JSON writing and provenance collection. Throughput is candidate observations
processed per second, not trades per second or an improvement over an existing library.
The benchmark is machine-specific and may contend with other work.

## Compare the existing PSR/DSR diagnostics

The optional adapter uses the built OpenQuant extension and the project's Python
environment. Build it as described in [CONTRIBUTING.md](../CONTRIBUTING.md), then run:

```bash
.venv/bin/python experiments/compare_validation_diagnostics.py \
  --repetitions 200 --candidates 64 --out /tmp/openquant-diagnostics.json
.venv/bin/python -m pytest python/tests/test_validation_diagnostics.py -q
```

The adapter replays the base lab's exact matrices and checks their hashes. For the
candidate chosen using training data, it reports training PSR, training DSR and
honest-holdout PSR. It also reports PSR for the deliberately contaminated holdout
winner. The PSR scores use a zero reference Sharpe; DSR instead uses the expected
maximum null Sharpe estimated from the trial count and trial-Sharpe dispersion.
Sharpe uses sample standard deviation; skewness and raw kurtosis use population
central-moment ratios. No statistic is annualized. Positive per-column scaling protects the moment
calculations from avoidable overflow without changing the scores.

Training DSR uses all candidate training Sharpes and the production API's population
standard-deviation convention. A second DSR collapses numerically exact duplicate
training columns as a sensitivity comparison. This is not an estimate of an effective
independent trial count, and it does not correct general cross-candidate correlation.
DSR is unavailable (`null`) with fewer than two applicable trials, rather than being
reported as zero.

By default the first replication of each family also exercises `TrialRegistry` in a
new temporary file. Every candidate ID is a different configuration, including exact
return duplicates. The audit verifies count persistence after reopening, stable counts
after recording the same configuration again, and agreement with direct DSR. Only
training returns enter the registry. `--registry-repetitions 0` disables this audit;
larger values audit that many replications per family. Existing registries are never
modified.

`--threshold` defaults to 0.95. Each metric's summary retains available/unavailable
counts, the mean score, Monte Carlo standard error and threshold-exceedance fraction.
The Wilson 95% interval describes uncertainty across independent replication events;
it is not a simultaneous interval across metrics or a profitability confidence bound.
The report records caveats for cross-correlated candidates, duplicate configurations,
infinite fourth moments and serial dependence. Training and contaminated-holdout
scores are post-selection. None of these outputs establishes calibrated skill
probabilities or a trading promotion decision.

The CLI retains hashes of the adapter, Python evaluation source and compiled native
extension. Its timing includes the base simulation, replay, diagnostics and temporary
registry audits, but excludes JSON serialization and provenance collection. This is
more work than the NumPy-only lab, so their timings are not like-for-like benchmarks.

## Research sources and limits

- Giuseppe A. Paleologo, [*The Elements of Quantitative Investing*, Wiley, 2025](https://www.wiley.com/en-us/the-elements-of-quantitative-investing-p-9781394265459).
  Publisher contents identify backtesting protocol and Rademacher Anti-Serum (RAS) in
  chapter 8. The published full text and exact theorem constants have **not** been
  reviewed here. This lab is an original engineering proposal motivated by validation
  needs, not a reproduction or implementation of that book's RAS guarantee.
- Bailey, Borwein, López de Prado and Zhu, [*Pseudo-Mathematics and Financial Charlatanism*, 2014](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2308659).
  Primary research on backtest overfitting and out-of-sample degradation supports
  separating the search result from independent evaluation.
- Marcos López de Prado, [*Causal Factor Investing*, Cambridge, 2023](https://www.cambridge.org/core/elements/causal-factor-investing/9AFE270D7099B787B8FD4F4CBADE0C6E).
  The publisher's summary distinguishes associational evidence from causal claims.
  Passing a synthetic benchmark does not establish a causal investment mechanism.
- Coqueret and Guida, [*Machine Learning for Factor Investing: Python Version*, CRC Press, copyright 2024](https://www.routledge.com/Machine-Learning-for-Factor-Investing-Python-Version/Coqueret-Guida/p/book/9780367639747).
  A relevant newer Python-oriented resource for subsequent factor-research examples;
  no method from its full text was implemented in this change.

Publisher summaries and legitimate public research were used. No full-book purchase,
paywall bypass, copied book text, or unlicensed third-party implementation is included.

## Continuation and acceptance gates

The practical sequence is an engineering proposal, not verified evidence of user demand:

1. **Now: reproducible selection benchmark.** This scaffold, its deterministic tests,
   and retained timing/results establish controls without changing production APIs.
2. **Now: existing-diagnostic comparison.** The optional adapter applies OpenQuant's
   PSR/DSR and trial registry to the same fixed candidate matrices. Inspect its explicit
   convention/assumption caveats, duplicate/correlated-family behavior and threshold
   frequencies with simulation precision. Do not call those frequencies calibrated
   without verifying assumptions.
3. **Then: risk-model utility benchmark.** Compare held-out realized portfolio risk,
   covariance conditioning, weight stability and turnover under identical rebalancing
   and cost conventions, using existing estimators before adding another one.
4. **Later: experimental RAS estimation.** Only after an authorized full text or complete
   public primary source verifies formulas and assumptions. Keep empirical complexity
   estimation separate from theorem-based confidence bounds; preserve full candidate
   membership, normalization, Monte Carlo precision and seed. Continue time-aware
   holdouts; no formula makes an adaptively reused holdout fresh again.

Before promoting the experiment to a supported production API, complete the repository's
maintainer-reviewed work record and any required formal specification, full
locked-environment gates and independent review. Publishing this research scaffold does
not establish a new production metric, package release or trading recommendation.

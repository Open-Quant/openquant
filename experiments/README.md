# Experiments Scaffold

This directory provides config-driven, reproducible research runs for OpenQuant.

## Run

```bash
uv run --python .venv/bin/python python experiments/run_pipeline.py \
  --config experiments/configs/futures_oil_baseline.toml \
  --out experiments/artifacts
```

Grid run (algo-wheel style sweep):

```bash
uv run --python .venv/bin/python python experiments/run_pipeline.py \
  --config experiments/configs/futures_oil_baseline.toml \
  --grid-config experiments/configs/futures_oil_grid.toml \
  --out experiments/artifacts
```

## Artifacts

Each run writes a deterministic folder. Existing run folders are never overwritten;
use a fresh `--out` directory to repeat a run. An incomplete run folder is also retained
for inspection rather than replaced on a later attempt.

`meta.name` and grid entry names must be non-empty single path components (no directory
separators, absolute paths, or `.`/`..`). Grid entries that would share the same name and
configuration-derived directory are rejected before the dataset or pipeline runs. The
same label with different configurations is supported, as are unnamed entries with their
default numbered names.

Each folder contains:

- `run_manifest.json` (config hash, git sha, versions, seed, `dataset_hash`: the content hash of the data the run used, see `openquant.data.dataset_hash`)
- `metrics.parquet` (summary table)
- `events.parquet` (event stage output)
- `signals.parquet` (signal timeline)
- `weights.parquet` (portfolio weights)
- `backtest.parquet` (equity/returns/position)
- `equity_curve.svg` (equity line chart from `backtest.parquet`)
- `drawdown.svg` (drawdown from the running peak, `equity / peak - 1`)
- `decision.md` (promotion verdict and rationale)

The two SVGs are fixed-size and written without a plotting library, so the same config and
seed give byte-identical files.

Grid runs additionally include:
- `leaderboard.parquet` (ranked cross-run summary)
- one artifact subdirectory per grid entry with the same single-run files

## Notes

- Default config is futures-centric and oil-inclusive.
- Runner uses `openquant.research.run_flywheel_iteration` for notebook/script parity.

For a map of the inference targets, assumptions and failure controls, read the
[offline inference research guide](INFERENCE_RESEARCH_GUIDE.md).

## Offline selection-validation lab

[The synthetic validation lab](VALIDATION_LAB.md) studies candidate-selection bias
using six controlled return families. It requires only NumPy and keeps training-only
selection separate from a deliberately contaminated holdout comparator:

```bash
python experiments/validation_lab.py --out /tmp/openquant-validation.json
python -m pytest python/tests/test_validation_lab.py -q
```

The JSON records all replications, input hashes, source/version provenance and timing.
These are synthetic diagnostics, not a trading strategy, calibrated confidence bound,
or promotion decision.

With the built OpenQuant extension, compare the existing PSR/DSR and trial-registry
diagnostics on the same controls:

```bash
.venv/bin/python experiments/compare_validation_diagnostics.py \
  --out /tmp/openquant-diagnostics.json
```

The [lab guide](VALIDATION_LAB.md#compare-the-existing-psrdsr-diagnostics) explains
score conventions, unavailable DSR values, registry audits and simulation intervals.

## Offline rolling risk-model comparison

[The risk-model utility lab](RISK_MODEL_LAB.md) compares existing minimum-variance,
HRP and baseline allocations on three controlled covariance families. Rolling fits
use only historical rows; the report separates realized risk, concentration, target
changes, drift-aware traded fractions and a declared toy cost model.

```bash
.venv/bin/python experiments/risk_model_lab.py --out /tmp/openquant-risk.json
```

All six methods are fixed before simulation. The lab reports independent-path and
paired Monte Carlo precision, unavailable paths, input hashes and native/source
provenance. It selects no winner and provides no investment guarantee.

## Reusing validation feedback

The [adaptive holdout lab](ADAPTIVE_HOLDOUT_LAB.md) extends the one-shot selection
experiment with feedback-dependent candidate blends. It retains the complete search
path and compares validation scores with a separately simulated audit sample. Frozen
training selection and fixed-family validation search provide distinct controls.
The audit sample is independent by construction; this does not establish independence
for adjacent real-world time periods.

```bash
python experiments/adaptive_holdout_lab.py --out /tmp/adaptive-holdout.json
```

## Future-feature preprocessing audit

The [PCA preprocessing lab](PREPROCESSING_LEAKAGE_LAB.md) compares a training-only
frozen transform with a deliberately contaminated transform fitted to future feature
rows. Both response models use training labels only. Stationary and covariance-shift
controls retain cases where future-feature access helps, hurts, or has little effect;
an all-component ridge control checks rotation invariance. These are original synthetic
experiments, not measurements of OpenQuant's production PCA implementation.

```bash
python experiments/preprocessing_leakage_lab.py --out /tmp/preprocessing.json
```

## Exact sign-test audit after search

The [sign audit](ADAPTIVE_SIGN_AUDIT.md) compares apparent validation rejection with
independent-audit rejection after the same adaptive search. It uses an exact one-sided
binomial tail for `Pr(return > 0 | return != 0)`, not
a test of mean return or Sharpe. It reports the attainable discrete null size and keeps
the serial-dependence case as an intentional violation of the test's assumptions.

```bash
python experiments/adaptive_sign_audit.py --out /tmp/adaptive-sign.json
```

## Bounded finite-family mean bounds

The [bounded finite-family lab](BOUNDED_FINITE_FAMILY_LAB.md) checks a published
empirical Rademacher bound against a simple finite-family Hoeffding bound under the
same failure budget. Payoffs have a known [0, 1] range and the hypothesis family is
fixed before iid evaluation rows are drawn. The Monte Carlo approximation gets its
own failure budget; its empirical standard error is not substituted for that bound.
The report retains vacuous bounds, tighter simple controls, and repeated-row failures.
This is not Paleologo RAS and does not apply directly to unbounded or dependent returns.

```bash
python experiments/bounded_finite_family_lab.py --out /tmp/bounded-family.json
```

## Independent units and clustered observations

The [cluster sign lab](CLUSTER_SIGN_LAB.md) compares raw-observation and block-sum
sign tests in Gaussian AR controls. Independent blocks have an explicit Gaussian and
binomial power reference. A continuous path divided into blocks and a shared-factor
control show why grouping observations does not establish independent statistical
units. Small-block-count cases retain the test's inability to reject at the configured
threshold. These constructions do not prescribe a block size for market data.

```bash
python experiments/cluster_sign_lab.py --out /tmp/cluster-sign.json
```

## Replay saved offline evidence

Replay runs the recorded configuration and compares all scientific fields, including
input hashes, individual replications, and summaries. It rejects changed source hashes
and NumPy versions; timings and host provenance are excluded from the comparison.
Use a fresh Python process so imports cannot reuse stale notebook dependencies:

```bash
python experiments/replay_offline_report.py /tmp/openquant-validation.json \
  --lab selection_validation
python experiments/replay_offline_report.py /tmp/adaptive-holdout.json \
  --lab adaptive_holdout --out /tmp/adaptive-replay.json
python experiments/replay_offline_report.py /tmp/preprocessing.json \
  --lab preprocessing_leakage
python experiments/replay_offline_report.py /tmp/adaptive-sign.json \
  --lab adaptive_sign_audit
python experiments/replay_offline_report.py /tmp/bounded-family.json \
  --lab bounded_finite_family
python experiments/replay_offline_report.py /tmp/cluster-sign.json \
  --lab cluster_sign
```

To check source/dependency hashes and numerical-runtime identity before spending
time on a large replay, use:

```bash
python experiments/replay_offline_report.py /tmp/selection-budget.json \
  --lab selection_budget --check-identity
```

This mode prints the saved configuration for review and explicitly reports
`scientific_replay_performed: false`. It does not validate configuration semantics
or scientific results, and it never imports or runs a lab module. It still reads
the full JSON report and imports any required numerical runtime to check versions.

The exit status is 0 for an exact replay match (or a successful identity-only
check), 1 for changed scientific results during replay, and 2 for
invalid input or mismatched source/version identity. Only the named local labs
can run; a report cannot supply code or import paths. Review the recorded configuration
before replaying an unfamiliar file because its sample sizes determine CPU/memory use.
The optional output cannot overwrite the input, including through a hardlink.

This checks reproducibility, not authenticity: someone can rewrite both data and hashes.
Exact comparisons may fail on another numerical platform even with the same NumPy
version. A matching synthetic report is not evidence of profitable real-market behavior.
The in-process Python helper checks source files on disk but does not reset previously
imported dependencies; use the fresh-process CLI for the recorded replay evidence.


## Selection-aware intervals and observation budgets

The [selection interval lab](SELECTION_INTERVAL_LAB.md) compares naive selected
Clopper–Pearson intervals, fixed-family Bonferroni correction, and an independent
audit. It preserves hidden-family, dependent-row, and all-tie failures. Its target
is the conditional probability of a positive nonzero observation, not mean return.
The [selection budget lab](SELECTION_BUDGET_LAB.md) evaluates full-sample correction
and split-sample audits under equal observation-row budgets using a finite IID
Bernoulli model. Candidate evaluation costs are recorded separately.

These two optional research labs need SciPy; install their pinned requirements in
a separate environment rather than changing the application's dependencies:

```bash
python -m pip install -r experiments/requirements-selection-intervals.txt
python experiments/selection_interval_lab.py --out /tmp/selection-interval.json
python experiments/selection_budget_lab.py --out /tmp/selection-budget.json
python experiments/replay_offline_report.py /tmp/selection-interval.json --lab selection_interval
python experiments/replay_offline_report.py /tmp/selection-budget.json --lab selection_budget
```

Replay checks both NumPy and SciPy versions for these labs, plus the recorded local
source and dependency hashes. Earlier NumPy-only labs do not require SciPy.


## Repeated looks and optional stopping

The [sequential sign lab](SEQUENTIAL_SIGN_LAB.md) compares repeated fixed-time
binomial tests with fixed betting rules under an explicit conditional sign null.
A finite-state calculation checks first-crossing probabilities, including a valid
dependent process and invalid marginally fair persistent and anticipatory controls.
This NumPy-only experiment does not implement a live market monitor.

```bash
python experiments/sequential_sign_lab.py --horizon 256 --out /tmp/sequential-sign.json
python experiments/replay_offline_report.py /tmp/sequential-sign.json --lab sequential_sign
```


## Candidate multiplicity and covariance estimation

The [multiple sequential lab](MULTIPLE_SEQUENTIAL_LAB.md) combines a fixed
candidate family with repeated observation times. It distinguishes a global
discovery from a candidate-specific claim, with dependent-stream controls and
exact integer crossing decisions.

The [covariance estimation lab](COVARIANCE_ESTIMATION_LAB.md) studies unrestricted
sample minimum-variance weights under known Gaussian covariance. Exact-law
references expose in-sample optimism, estimation risk, and infinite-moment
regimes. It uses no native allocator and models no trading costs or leverage limits.
SciPy adds reference quantiles; without it, those fields are explicitly unavailable.
Replay requires matching its recorded presence or absence as well as its version.

```bash
python experiments/multiple_sequential_lab.py --out /tmp/multiple-sequential.json
python experiments/replay_offline_report.py /tmp/multiple-sequential.json --lab multiple_sequential
python experiments/covariance_estimation_lab.py --out /tmp/covariance-estimation.json
python experiments/replay_offline_report.py /tmp/covariance-estimation.json --lab covariance_estimation
```


## Sign evidence and economic payoff

The [payoff target lab](PAYOFF_TARGET_LAB.md) applies different fixed win/loss
magnitudes and costs to identical sign paths. Bounded-stopping calculations
retain both crossing and non-crossing paths: convincing gross-sign evidence
can coexist with negative expected net arithmetic payoff. This models fixed
payoff units per observation, without compounding, capital constraints or a
deployable trading strategy.

```bash
python experiments/payoff_target_lab.py --out /tmp/payoff-target.json
python experiments/replay_offline_report.py /tmp/payoff-target.json --lab payoff_target
```


For the complete optional numerical runtime used by these offline studies, install
`experiments/requirements-offline-research.txt` in a separate environment. This does
not change production dependencies. Plot scripts additionally use Matplotlib.

The normal locked development environment does not include SciPy, so its
SciPy-dependent research tests are skipped there. CI separately installs these
pinned research requirements and runs every `test_*_lab.py` module plus
`test_replay_offline_report.py`: on Python 3.13 in normal CI, and on both
Python 3.11 and 3.13 in nightly validation. An explicit NumPy/SciPy import check
precedes that phase so missing numerical dependencies fail rather than silently
skip its coverage. The standard tests and runtime smokes run before that install.

## Lossless storage and mean robustness

[Lossless report storage](OFFLINE_REPORT_STORAGE.md) keeps exact JSON bytes in gzip,
verifies a byte-for-byte round trip and preserves the original. Replay accepts both
plain and gzip reports with the same strict JSON checks.

The [mean robustness lab](MEAN_ROBUSTNESS_LAB.md) compares known-variance bounds,
median-of-means and naive controls under heavy tails and dependence. Its clipping
followup distinguishes the clipped population mean from the original target.


The [rare-event uncertainty lab](RARE_EVENT_UNCERTAINTY_LAB.md) evaluates finite-law
coverage when a sample contains no rare events. Empirical variance and bootstrap
intervals are contrasted with explicitly assumption-rich known-variance and
known-support controls. The [informative missingness lab](INFORMATIVE_MISSINGNESS_LAB.md)
separates observed-only targets from population targets, with known-propensity
weighting, positivity failures and observationally indistinguishable populations.

The [covariance robustness lab](COVARIANCE_ROBUSTNESS_LAB.md) compares fixed
constraints and shrinkage choices under heavy tails and future covariance shifts.
Its retained numerical-equivalence audit prevents tiny solver differences from
being interpreted as methodological wins. The [point-in-time lab](POINT_IN_TIME_LAB.md)
provides an availability-aware revision lookup with explicit missingness and
provenance, alongside delayed-publication and hindsight-revision controls.


The [delayed-label example](DELAYED_LABEL_LAB.md) separately audits which training
labels had matured at fit time. A Gaussian linear-weight reference compares safe
fitting, a complete-archive comparator and a zero baseline, retaining cases where
the contaminated fit still loses to the simple baseline.


The [constraint solver lab](CONSTRAINT_SOLVER_LAB.md) distinguishes a unique
feasible inverse solution from nonunique zero-training-risk solutions. Its exact
reuse shortcut is checked against the strict solver; timing is incremental and
measured in one environment. The [zero-fit probability lab](ZERO_FIT_PROBABILITY_LAB.md)
compares numerically checked convex-hull feasibility with geometric reference
laws, retaining correlated controls and unresolved outcomes explicitly.

The [partial identification lab](PARTIAL_IDENTIFICATION_LAB.md) separates sharp
population bounds under unknown observation propensities from finite-sample
confidence sets. Observationally equivalent populations retain the same evidence
while their latent means differ.

For conclusions across the twenty additions, start with
[toolkit integration decisions](TOOLKIT_INTEGRATION_DECISIONS.md). It distinguishes
implementation candidates from assumption-dependent examples and includes the
retained ill-conditioning and BLAS-thread counterexamples.

Add `--runtime-context` to a replay command to record
[current execution diagnostics](RUNTIME_CONTEXT.md). These observations do not
reconstruct the original runtime or change exact scientific comparisons.

[Availability workflow qualification](AVAILABILITY_WORKFLOW_FINDINGS.md) exercises
irregular decisions, release delays, outcome-encoded revisions and the actual
Polars cleaner. It includes controls that pass supplied-clock checks while using
false provenance, and states the pipeline contracts needed for useful integration.

[Revision association guardrail](PROVENANCE_GUARDRAIL_FINDINGS.md) binds exact
selected numeric values to declared revisions through cleaning and reordering.
It detects inconsistent associations while retaining counterexamples for false
metadata, incomplete history and lossy normalization.

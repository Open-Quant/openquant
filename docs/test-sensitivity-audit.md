# Test sensitivity audit (issue #40)

**Question asked of every test file:** if the library computed a wrong number, would this file
fail? Loading a fixture is not checking a value, so the answer was *measured*, not read off.

## Method

For each library module 1-6 small, realistic numeric mutations were applied to the source
(flip a sign, `n - 1` -> `n`, drop a term, `<=` -> `<`, swap a convention), that module's test
targets were run, and the outcome recorded as KILLED (some test failed) or SURVIVED (everything
still passed). Each module also got one **control**: make a core function return an obviously
wrong constant (`1234.5 + 0.0 * <expr>`). A control that survives means the test file cannot see
that function's value at all.

* Harness: `scripts/test-sensitivity/mutate.py` (the full mutation list is in that file);
  summary: `scripts/test-sensitivity/summarize.py`; raw results: `docs/test-sensitivity-results.tsv`.
* Every result in this document was measured in an isolated copy of the workspace with its own
  cargo target directory (`CARGO_TARGET_DIR` unset). A run only counts if cargo visibly rebuilt
  `openquant` with the mutation in place. An earlier pass that shared a target directory with
  another checkout was discarded in full and is not used anywhere here.
* Mutations never touch the checkout: the harness edits the copy and restores it.
* State measured: `main` at `3cc8df0` (includes #80) for everything except `labeling`,
  `sample_weights` and `util::volatility`, which #81 changed and which were re-measured after
  merging `bb1c1a6`. 36 controls + 92 realistic mutations.
* Order of work: two sessions were lost mid-task, so the new tests were committed before this
  document. The "before" column was still measured honestly: it runs only the pre-existing test
  targets against each mutation.
* Limits: this is a hand-picked sample, not exhaustive mutation testing. KILLED on 2 of 2 does not
  prove a file is thorough; SURVIVED is the hard evidence. Verdicts:
  **SENSITIVE** = every realistic mutation killed, **PARTLY** = some killed,
  **INSENSITIVE** = none killed.

## Corrected headline numbers

The earlier claim was "19 of 37 Rust test files check against fixtures". What is true:

* 19 of 37 files **load** a fixture. Of those 19, **7** killed every mutation tried, **7** killed
  some, and **5 killed none** (`cla`, `fracdiff`, `hcaa`, `risk_metrics`,
  `portfolio_optimization`) - they read the fixture and then assert structure only.
* Over all 37 files (+ the inline tests in `pipeline.rs` and `onc.rs`):
  **12 SENSITIVE, 16 PARTLY, 9 INSENSITIVE** (plus `pipeline` inline: INSENSITIVE).
* **41 of 92** realistic mutations were killed by the pre-existing suite (45%).
* **6 of 36 controls survived** when first measured - returning a constant from the function under
  test passed the whole file: `fracdiff`, `risk_metrics`, `hyperparameter_tuning`
  (accuracy), `data_processing` (gap count), `pipeline` (equity curve), `sample_weights`
  (since fixed on main by #81; re-measured: killed).
* For the eight modules this PR worked on: **6 of 32** mutations killed before, **30 of 32**
  after; the two survivors are equivalent mutants (explained below). All four surviving controls
  among them are now killed.

## Per-file table

"k/n" = realistic mutations killed / tried. `ctl` = control. Files marked (oos) belong to issue
#76 and were measured but not touched.

| Test file | Module | Tests | What the assertions are | Mutations before (k/n) | Verdict before | After this PR |
|---|---|---|---|---|---|---|
| backtest_statistics.rs | backtest_statistics | 8 | mlfinlab reference literals, tol 1e-4..1e-1 | 2/3; PSR skewness sign flip survived | PARTLY | - |
| backtesting_engine.rs | backtesting_engine | 3 | counts, enum equality, "purged > 0" | 0/5 (ddof, open/closed overlap, embargo width, sqrt(n), negated path returns) | INSENSITIVE | **5/5** via backtesting_engine_reference.rs (7 tests) |
| bet_sizing.rs + ch10_snippets.rs | bet_sizing | 17 + 32 | fixture values and inline closed forms, tol 1e-12 | 3/3 | SENSITIVE | - |
| cla.rs (oos) | cla | 20 | structure (sum to 1, bounds), one 1e-12 literal | 0/2 (EMA weight sign, sqrt dropped from sigma) | INSENSITIVE | not in scope |
| codependence.rs | codependence | 3 | reference literals, tight | 2/2 | SENSITIVE | - |
| combinatorial_optimization.rs | combinatorial_optimization | 5 | inline expected values and optimal decisions | 0/2 (impact cost sign, risk penalty abs(x) vs x^2) | INSENSITIVE | not done |
| cross_validation.rs | cross_validation | 9 | index-set properties, scores only checked to lie in [0,1] | 1/3; embargo off-by-one and F1 missing factor 2 survived | PARTLY | not done |
| data_processing.rs | data_processing | 1 | row counts on one 3-row case | 1/2; **ctl survived** (gap count never asserted) | PARTLY | not done |
| data_structures_standard.rs + data_structures_run_imbalance.rs | data_structures | 2 + 2 | inline exact bar values | 1/3; `>=` vs `>` volume threshold and signed-vs-abs imbalance survived (no boundary case) | PARTLY | not done |
| ef3m.rs | ef3m | 10 | inline expected moments/parameters | 2/2 | SENSITIVE | - |
| ensemble_methods.rs | ensemble_methods | 5 | inline values; bagging variance formula checked only by inequality | 1/3; variance formula sign and majority-vote tie rule survived | PARTLY | not done |
| etf_trick.rs | etf_trick | 3 | fixture comparison | 1/2; cost sign survived (cause not investigated) | PARTLY | not done |
| futures_roll.rs | etf_trick (roll) | 1 | reference literals | 1/1 | SENSITIVE | - |
| fast_ewma.rs | util::fast_ewma | 1 | reference literals | 1/1 | SENSITIVE | - |
| feature_importance.rs | feature_importance | 3 | orderings, `is_finite()`, sum to 1 | 0/4 (stderr exponent, log-loss sign, eigenvalue weighting, std vs var) | INSENSITIVE | **4/4** via feature_importance_reference.rs (13 tests, 4 ignored FINDINGs) |
| filters.rs | filters | 5 | CUSUM events vs fixture; z-score only structural | 1/2; z-score ddof survived | PARTLY | not done |
| fingerprint.rs | fingerprint | 5 | model-vs-model, tol 0.05 | 2/2 | SENSITIVE (loose tol, see list) | - |
| fracdiff.rs | fracdiff | 4 | lengths, "last weight == 1", "first is NaN" | 0/2; **ctl survived** | INSENSITIVE | **2/2, ctl killed** via fracdiff_reference.rs (7 tests) |
| hcaa.rs | hcaa | 14 | non-negative, sums to 1, one leaf-order literal (min-variance only) | 0/6 | INSENSITIVE | **4/6** via hcaa_reference.rs (7 tests, 1 ignored FINDING); 2 survivors are equivalent mutants |
| hpc_parallel.rs | hpc_parallel | 7 | coverage/partition properties | 1/2; linear partition boundary formula survived | PARTLY | not done |
| hrp.rs | hrp | 9 | non-negative, sums to 1, one leaf-order literal | 1/4; inverted split factor, shrinkage, 1/sd-vs-1/var survived | PARTLY | **4/4** via hrp_reference.rs (9 tests, 2 ignored FINDINGs) |
| hyperparameter_tuning.rs | hyperparameter_tuning | 3 | determinism, "accuracy > 0.9", range membership | 1/2; **ctl survived** (accuracy value never asserted); log-uniform sampler `exp`->`abs` survived | PARTLY | **2/2, ctl killed** via hyperparameter_tuning_reference.rs (4 tests) |
| labeling.rs | labeling | 9 | fixture events/labels; one tolerance is 10% of the value | 2/3; simple-vs-log return in barrier touch survived | PARTLY | not done |
| microstructural_features.rs | microstructural_features | 8 | mlfinlab literals at 1e-3 | 2/2 | SENSITIVE | - |
| onc.rs + inline onc tests | onc | 3 + 2 | cluster membership on real data + planted blocks | 4/4 | SENSITIVE | + onc_reference.rs pins silhouette VALUES vs scikit-learn (none was asserted before) |
| portfolio_optimization.rs (oos) | portfolio_optimization | 13 | `max_diff < 1.0` on weights in [0,1]; sums to 1 | 0/2 (log vs simple returns, EW mean normalisation) | INSENSITIVE | not in scope |
| risk_metrics.rs | risk_metrics | 7 | every assertion is `is_finite() \|\| is_nan()` | 0/3; **ctl survived** | INSENSITIVE | **3/3, ctl killed** via risk_metrics_reference.rs (5 tests) |
| sample_weights.rs (oos) | sample_weights | 3 | after #81: reference JSON at 1e-10 (before #81: tol 1e5) | 1/2; avg-uniqueness denominator survived | PARTLY | not in scope |
| sampling.rs | sampling | 6 | book literals at 1e-2 (book quotes 2 dp) | 2/2 | SENSITIVE | - |
| sb_bagging.rs | sb_bagging | 7 | behaviour thresholds (`mse < 0.4`) | 1/2; `max_samples` halved survived | PARTLY | not done |
| strategy_risk.rs | strategy_risk | 5 | inverse-consistency round trips | 2/2 | SENSITIVE | - |
| streaming_hpc.rs | streaming_hpc | 4 | before/after inequalities, serial == parallel | 1/3; VPIN normalisation and HHI increment survived (both sides of `==` share the bug) | PARTLY | not done |
| structural_breaks.rs | structural_breaks | 4 | mlfinlab literals at 1e-3 | 2/3; OLS variance ddof survived | PARTLY | not done |
| synthetic_backtesting.rs | synthetic_backtesting | 5 | determinism, `abs < 0.08`, qualitative | 0/2 (OU intercept sign, doubled path noise) | INSENSITIVE | not done |
| volatility_features.rs (+ daily_vol.rs from #81) (oos) | util::volatility | 1 + daily_vol.rs | mlfinlab / pandas reference | 2/2 | SENSITIVE | not in scope |
| inline `#[cfg(test)]` in src/pipeline.rs | pipeline | 2 | lengths, weights sum to 1 | 0/2; **ctl survived**; trading on the same bar's signal (look-ahead) passes | INSENSITIVE | **2/2, ctl killed** via pipeline_reference.rs (1 hand-worked test) |

### The two surviving "after" mutations are equivalent mutants

* `hcaa-2` negates cluster expected shortfall. The split factor is `1 - ES_L / (ES_L + ES_R)`;
  negating both leaves it unchanged whenever the two ES values have the same sign.
* `hcaa-4` drops `/ rows` from the annualised mean return. That rescales every expected return by
  the same constant and the Sharpe split `SR_L / (SR_L + SR_R)` is scale-free.

## Vacuous or suspicious tolerances

| Location | Assertion | Why it cannot fail (or barely can) |
|---|---|---|
| tests/portfolio_optimization.rs:82, :92, :184, :198 | `max_diff < 1.0` | weights live in [0,1]; any long-only answer passes (oos, #76) |
| tests/portfolio_optimization.rs:173 | `max_diff < 0.25` | a quarter of the whole portfolio per asset (oos) |
| tests/portfolio_optimization.rs:100, :109, :222 | `(sum - 1).abs() < 1e-2` | 1% leakage of total weight accepted (oos) |
| tests/risk_metrics.rs:55, :64, :73, :83, :96, :109, :122 | `x.is_finite()` / `x.is_finite() \|\| x.is_nan()` | true for every f64 except +-inf; control survived |
| tests/sample_weights.rs:72, :73, :93 (before #81) | `abs() <= 1e5` | tolerance of 100000; fixed on main by #81 |
| tests/labeling.rs:120 | `(trgt - 0.010166).abs() < 1e-3` | 10% of the value |
| tests/backtest_statistics.rs:105, :107 | Sharpe / information ratio `< 1e-2` | 1% of the value; reference has 6 digits |
| tests/backtest_statistics.rs:127 | min TRL `< 1e-1`, commented "loosened tolerance" | loosened to fit, no numerical justification given |
| tests/sb_bagging.rs:227-228 | `mse < 0.4`, `mae < 0.5` | threshold with no derivation; a regressor returning the constant 0.5 on a 0/1 target scores mse 0.25 |
| tests/synthetic_backtesting.rs:39 | `(phi_hat - phi).abs() < 0.08` | no standard-error justification; intercept sign flip survives |
| tests/fingerprint.rs:68, :93 | `abs() < 0.05` between two models | compares the library to itself |
| tests/hyperparameter_tuning.rs:173 | `accuracy > 0.9` | constant 1234.5 passes |
| tests/feature_importance.rs:104-108 | `is_finite()`, `-1 <= x <= 1` | any correlation-like number passes |
| src/pipeline.rs (`LeakageChecks`) | `has_forward_look_bias: false`, `inputs_aligned: true` | hard-coded constants, not checks |

`tests/sampling.rs:50-53` (1e-2) and `tests/microstructural_features.rs`, `tests/structural_breaks.rs`
(1e-3) look loose but match the number of digits the book / mlfinlab quote; they killed their
mutations.

## Findings from the new tests (library unchanged; tests are `#[ignore = "FINDING: ..."]`)

1. **hrp / hcaa weekly and monthly resampling scrambles the price matrix - library bug.**
   `src/hrp.rs:140` and `src/hcaa.rs:169` fill a `Vec` row by row and hand it to the
   column-major `DMatrix::from_vec`. Same class of defect as #79, in a code path no test
   exercised (the existing test passes `"B"`, which short-circuits). Evidence: allocating with
   `resample_by = "W"` differs from allocating on rows 4, 9, 14, ... directly; leaf order is
   completely different. Tests: `hrp_reference.rs:110`, `:123`, `hcaa_reference.rs:227`.
   `src/cla.rs:362` and `src/portfolio_optimization.rs:93` contain the same construction; they
   are out of scope here and were **not** checked.
2. **backtesting_engine embargoes before the test block as well as after - deviation from AFML
   7.4.2, conservative.** `src/backtesting_engine.rs:551` (`test_idx.saturating_sub(width)`).
   AFML embargoes only the samples that follow a test set. Not a leak; it discards training data
   (2 x h per fold). Could be a deliberate design choice, but it is undocumented and
   `cross_validation::PurgedKFold` in this same crate has the same two-sided shape, so I lean
   "unintended". Test: `backtesting_engine_reference.rs:219`.
3. **feature_importance MDI / MDA standard errors use ddof = 0 - convention difference with a
   numeric effect.** AFML snippets 8.2/8.3 call pandas `.std()` (ddof = 1). Reported std is too
   small by sqrt((n-1)/n): 18% for 3 folds. SFI (snippet 8.4, numpy `.std()`, ddof = 0) is correct
   as is and is pinned by a passing test. Tests: `feature_importance_reference.rs:64`, `:164`.
4. **feature_pca_analysis Spearman / Kendall mishandle ties - library bug.** The importance vector
   is tiled once per principal component, so ties are guaranteed whenever more than one component
   is kept; `rank_desc` (`src/feature_importance.rs:440`) assigns ordinal rather than average
   ranks and `kendall_tau` (`:456`) is tau-a over untied pairs. scipy gives Spearman -0.0123,
   the library 0.0662 (sign differs). With one component (no ties) both match scipy to 1e-9.
   Test: `feature_importance_reference.rs:308`.
5. **feature_pca_analysis weighted Kendall is not `scipy.stats.weightedtau` - wrong algorithm.**
   `weighted_kendall_tau` (`:482`) weights pair (i, j) by `1 / (1 + i + j)` using input positions.
   AFML 8.4.2 / mlfinlab use weightedtau (hyperbolic weights by rank). 0.179 vs scipy 0.354 on
   the reference case. Test: `feature_importance_reference.rs:325`.

Not findings, but worth knowing: HRP `use_shrinkage` is a fixed 10% off-diagonal shrink, not
mlfinlab's OAS estimator (design choice; the new test pins the library's own rule). HRP uses
simple returns (matches mlfinlab). The engine's per-fold `sharpe` is the t-statistic
`mean / std * sqrt(n)`, not an annualised Sharpe (pinned as such).

## What remains for #40

* INSENSITIVE and untouched: `combinatorial_optimization`, `synthetic_backtesting`; (oos) `cla`,
  `portfolio_optimization`.
* PARTLY, with a named surviving mutation to aim at: `cross_validation` (embargo off-by-one, F1),
  `data_processing` (control survives), `data_structures` (threshold boundary cases),
  `ensemble_methods` (bagging variance formula), `etf_trick` (costs), `filters` (z-score ddof),
  `hpc_parallel`, `labeling` (return convention at the barrier), `sb_bagging`, `streaming_hpc`
  (VPIN / HHI values), `structural_breaks` (OLS variance), `backtest_statistics` (PSR skewness),
  `sample_weights` (oos).
* `onc` has no reference for the re-clustering branch (`redo_clusters.len() > 2`); no test reaches it.
* The vacuous tolerances listed above that are not already owned by #76.
* Running `cargo mutants` per module would turn this sample into a census.

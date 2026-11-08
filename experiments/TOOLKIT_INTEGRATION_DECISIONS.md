# What the twenty research labs justify integrating

The strongest product result is an explicit information-availability audit. The strongest numerical result is a reason to defer a covariance shortcut: small optimization residuals do not certify equivalent allocations under ill-conditioning. The strongest inference result is that the toolkit must keep targets, observation assumptions and selection procedures explicit. None of these synthetic studies establishes investment performance.

This assessment is against public base `e3abe5198df2b8659fd8f1c5bd431b44129d115c`. The earlier private consolidated, moments and PSR checkpoints remain unrecovered. Recommendations below are implementation proposals, not claims that native CI or production integration passed.

## Integration choices

| Candidate | Decision | Reason and remaining requirement |
| --- | --- | --- |
| Source/label availability audit | Develop an additive, explicit-contract Python helper | Audit supplied information clocks before fitting; retain original indices and exclusion reasons. It cannot establish truthful provider metadata, transform isolation, independence or absence of all leakage. |
| Research replay and lossless compression | Keep as optional offline tools | Preserve exact bytes and originals, refuse overwrite and retain replay failures. Version identity does not fully describe BLAS execution settings; full JSON is loaded in memory. |
| Analytic reference fixtures and examples | Keep alongside tests/documentation | Closed-form and finite-law references expose regressions and assumptions without becoming automatic strategy-approval rules. |
| Clopper–Pearson, finite-family and sequential procedures | Keep as explicitly qualified research examples | Their different targets and observation assumptions do not justify silently replacing PSR/DSR or interpreting logged adaptive queries as a fixed family. |
| Covariance inverse shortcut | Defer production port | The existing native optimizer supports general bounds and uses ADMM with polishing, while the lab times a simplex SLSQP shortcut with an inverse already computed. Conditioning and end-to-end native benchmarks remain unresolved. |
| Shrinkage, singular tie-breaking or clipping defaults | Do not change from this evidence | Each changes the estimator or target. Future-distribution shifts and counterexamples prevent a universal preferred default. |

## Coverage of the twenty labs

The one-shot validation example already existed on the public base; it is a baseline, not counted among these twenty additions.

| Lab | Useful integration content | Limit to retain |
| --- | --- | --- |
| Adaptive holdout | Selection-versus-audit workflow | Feedback changes the candidate family. |
| Preprocessing leakage | Fold-local transform example and invariance control | Future features can help or hurt; timestamps alone cannot audit fitted transforms. |
| Adaptive sign audit | Exact sign-count regression reference | Independent sign trials, not an arithmetic-return claim. |
| Bounded finite family | Explicit bounded-score examples | Known range, IID rows and fixed family are substantive inputs. |
| Cluster sign | Dependence counterexamples | Grouping rows does not manufacture independent units. |
| Selection intervals | Simultaneous-versus-independent-audit examples | Both can fail when observations repeat. |
| Selection budget | Equal-observation-budget comparisons | Selection accuracy, width and computation have no uniform winner. |
| Sequential sign | Predetermined betting-rule example | Conditional fairness is stronger than marginal fairness. |
| Multiple sequential | Global-discovery versus candidate-attribution examples | A global result does not certify its leading candidate. |
| Covariance estimation | Exact Gaussian risk-law fixtures | Population-risk moments may not exist; sample-risk estimates can mislead. |
| Payoff target | Sign-versus-payoff counterexample | Positive signs and selected crossing paths do not imply positive expected payoff. |
| Mean robustness | Known-variance and clipping examples | Conservatism can mask violations; clipping changes the target. |
| Covariance robustness | Distribution-shift and solver-equivalence controls | Tiny numerical wins are not meaningful method improvements. |
| Rare-event uncertainty | Finite-law coverage fixtures | Empirical bootstrap cannot recover an event missing from the sample. |
| Informative missingness | Population-versus-observed target example | Propensities and positivity are assumptions; HT failure is not universal nonidentification. |
| Point in time | Revision-aware source metadata contract | Event time differs from publication time; historical provenance must be supplied. |
| Delayed labels | Separate release/maturity audit | Purging overlaps and respecting information release are distinct checks. |
| Constraint solver | Unique-versus-nonunique optimizer fixtures | Residual qualification alone cannot certify allocation equivalence. |
| Zero-fit probability | Convex-hull feasibility reference | Symmetry/general-position assumptions and numerical witness qualifications remain explicit. |
| Partial identification | Sharp bounds versus confidence-set example | Structural uncertainty can persist even with arbitrarily much data. |

## Concrete findings that matter for product behavior

- Point-in-time controls: on-time selection and event-only selection both score 0.751484 over 12,800 decisions. With delayed publication, availability-aware selection has 0.494609 correct decisions per total decision (0.502460 accuracy conditional on selection), and the event-only comparator scores 0.750469. The safe path retains 200 missing decisions and uses older events for the other 12,600; it uses no future publications. These are synthetic fixed-predictor accuracy controls, not economic returns.
- Delayed-label control: at horizon 16 and training length 16, analytic mean squared errors are 16 for the zero predictor, 26.6875 for mature-label fitting, and 11.6875 for the contaminated complete-archive fit. At horizon 4 and training length 16, the archive fit has MSE 4.171875 and still loses to zero's 4. An availability audit prevents a specific invalid comparison; it does not promise better prediction.
- Sparse events: for a Bernoulli event with probability .001, scaled to mean 1, n=128 bootstrap-percentile coverage is about .120193 versus .992524 for the known-support exact interval. At n=2048 the figures are .869877 and .981698. The exact interval uses real additional support information; this does not recommend a universal bootstrap replacement.
- Fixed clipping: the separately declared n=32,768 follow-up covered the clipped target in 500/500 replications and the original mean in 0/500. The full interval width is about .06002 (half-width .03001), while target bias is about .07950. The earlier smaller grid did not expose this failure and remains preserved.
- Repeated testing: fair-sign fixed-time testing crossed in 678/2,000 replications (33.9%) by 1,024 observations, compared with 88/2,000 (4.4%) and 86/2,000 (4.3%) for the two predeclared betting rules; the corresponding finite-sum references are .340129, .044447 and .039936. At horizon 256 across 32 independent null streams, uncorrected candidate claims occurred in 633/1,000 replications versus 30/1,000 after correction. These rates concern the declared synthetic nulls.
- Payoff selection: with positive-sign probability .6, horizon 64, unit wins/losses and cost .25, expected stopped payoff is about -2.8555 overall but +7.3155 conditional on crossing. Reporting only successful crossings reverses the apparent conclusion.
- Missingness: observationally equivalent populations have different latent means. For equivalent_A and equivalent_B in the partial-identification study (shared observed counts, 2,000 replications per setting), the structural identified-set width is .75; the average outer confidence-set width shrinks approximately 1.2815 to .8007 as n increases from 32 to 4,096. At the larger n, observed whole-set coverage .949 has Monte Carlo standard error .00492; plug-in bounds cover the whole set only .191 of the time. These are not simultaneous coverage claims across all settings.

Scientific reports, frozen protocols, exact replay verdicts and independent reviews are preserved in checkpoint B and its linked full-data archive. See the accompanying integration review for the new conditioning counterexample and runtime investigation. Do not infer a market-data guarantee from any of these controls.

## Integration uncertainties resolved in the follow-up

**Residuals are not allocation-error certificates.** For training covariance
`[[1, 0.99999999], [0.99999999, 1]]`, the condition number is approximately
200 million. The frozen experimental shortcut accepts externally supplied weights
`[1, 0]` at its residual tolerance, although the unique inverse allocation is
`[.5, .5]`. Maximum allocation error is .5. Under the diagnostic future covariance
`diag(1,100)`, variances are 1 and 25.25; reversing the diagonal reverses the
favorable direction. The internally computed candidate differs from the inverse
reference by only about 2.76e-10 here. This counterexample limits the helper's
arbitrary-candidate qualification, not the retained grid, which used internally
computed candidates. It was reproduced independently. No frozen source or result
was silently repaired.

**Exact replay needs more than package versions.** Six fresh runs of the same
p8,T68 representative configuration used actual BLAS pools of five (default), one,
and two threads, twice each. Within-setting repeats were exact; default and two
threads matched the original report. One thread differed in 16,198 floating
leaves, with maximum absolute allocation difference 5.27e-16 and future variance
difference 3.55e-15. Across 1,200 records per run, statuses, training hashes,
future-risk argmins and all ten future-risk pair indicators were unchanged.
Eight raw training-risk argmin labels flipped between mathematically equivalent
sample-GMV and long-only solutions. The lab does not select a method using those
labels. This limited sensitivity check neither proves harmlessness on other
configurations nor justifies silently relaxing exact replay. Runtime diagnostics
should describe the current process, with unknown original settings stated as
unknown.

These follow-ups narrow integration scope rather than establish new preferred
allocations. A production fast path would need internally computed candidates,
actual native bound/constraint checks, a conditioning-aware accuracy requirement,
and end-to-end benchmarks including factorization and fallback costs. None was
measured by the existing incremental SciPy benchmark.

# Fixed-schedule capacity-scaling lab

## Fixed plan, recorded before implementation and outcomes

Date: 2026-11-08. Base: `9740596e2d846d9e3c6653aec201be34bb69f37d`.

**Goal:** Add an offline, synthetic, additive dollar-PnL lab that separates a
model-implied zero-net scale from a trade-participation feasibility cap.

**Architecture:** A validated fixed schedule is evaluated independently of any
portfolio optimizer or self-financing ledger. The lab retains every cell of a
fixed Cartesian grid and writes schema-1 reports with replayable configuration
and source provenance. Standard-library Decimal arithmetic with 80-digit working
precision avoids ordinary binary64 intermediate overflow; JSON numeric output
still has explicit binary64 representation limits. This is numerical arithmetic,
not certified interval arithmetic or a real-market capacity estimator.

**Files:** `experiments/capacity_scaling_lab.py`, this document, and
`python/tests/test_capacity_scaling_lab.py`. Replay is registered in `replay_offline_report.py`.
No production optimizer changes, docs-site edits, tuning, installs, native builds,
remote writes, trading, or spending.

**Source/design:** Study 2 of the private implementation-cost design note dated
2026-11-08; original elementary algebra under the assumptions below. Boyd et al.,
[Multi-Period Trading via Convex Optimization](https://web.stanford.edu/~boyd/papers/pdf/cvx_portfolio.pdf),
sections 2.2–2.6, motivates explicit units and a separable 3/2-power cost model.
This lab does not reproduce the paper's optimizer or infer universal calibration.

### Contract

For a fixed schedule with signed dimensionless trade fractions `d`, scale `A > 0`
is in dollars and trades are `q = A*d`. Each execution bucket has known positive
*dollar* volume `V`, nonnegative fractional spread `s`, nonnegative dimensionless
impact coefficient `eta`, and nonnegative fractional volatility `sigma`.
The same horizon/bucket convention must apply throughout. Costs are

`sum(s*abs(q) + eta*sigma*abs(q)^(3/2)/sqrt(V))`.

With `L = sum(s*abs(d))` and
`I = sum(eta*sigma*abs(d)^(3/2)/sqrt(V))`, expected additive net PnL per dollar of
scale is `G-L-sqrt(A)*I`. `G` is the supplied finite expected gross PnL per dollar
of scale across the entire horizon. It may be negative. This is not compounded
wealth and does not assert solvency when costs exceed capital.

- If `G > L` and `I > 0`, zero-net scale is `((G-L)/I)^2`. Positive net is below
  that crossing, not at it.
- If `G <= L`, there is no positive-net scale, including the all-scale zero-net
  case `G=L, I=0`.
- If `G > L, I=0`, there is no finite crossing. Fixed-bps costs alone do not
  identify a finite impact-capacity threshold.
- Separately, participation `p` in `[0,1]` permits
  `A <= min(p*V/abs(d))` over nonzero trades. Zero participation blocks any
  nonzero trade; an all-zero schedule has no participation limit.
- Finite mathematical thresholds outside the binary64 output range are marked
  above/below range, never confused with an absent crossing or zero.
- All schedule arrays must have the same nonempty 1-D or 2-D shape. Only `p` may
  be a scalar broadcast to the schedule. Reject malformed shapes, booleans,
  strings, complex numbers, nonfinite values, negative coefficients, and unknown,
  zero, or negative volumes, even for zero trades. Zero trades with valid inputs
  contribute exactly zero and are excluded from participation minima.
- Bare numeric arrays cannot detect shares mistaken for dollars, mismatched
  currencies, inconsistent horizons, or economically implausible coefficients.
  Units are a caller responsibility, not a validated inference.

### Frozen grid and reporting

Keep exactly the suggested grid, with no calibration or post-outcome selection:

- `A = [1e3, 1e4, 1e5, 1e6, 1e7]` dollars.
- Dollar-volume multipliers `[0.1, 1, 10]`.
- Uniform `eta = [0, 0.1, 0.5]`.
- `G = [0, 0.001]`.
- `d = [0.1, -0.1]`, `s = [0.0001, 0.0001]`,
  `sigma = [0.02, 0.02]`, `V = [1e6, 4e6]`, `p = 0.01`.

All 90 cells will be retained, including negative net PnL and participation
violations. No Monte Carlo repetitions, standard errors, fitted costs, best-cell
selection, forecasts, live market data, or capital recommendations. Unit-error
and omitted-normalization negative controls will be labeled deliberately wrong;
showing a numeric effect cannot prove that an input has correct units.

### Implementation and verification plan

- [x] Write tests before code and record the expected missing-implementation red
  run. Pin a hand-derived one-trade case: `A=100`, `d=.25`, `s=.01`, `eta=.5`,
  `sigma=.2`, `V=100`, `G=.1`, `p=.1`: linear cost `.25`, impact `1.25`, total
  `1.5`, normalized net `.085`, zero-net scale `6084`, participation cap `40`.
- [x] Add hand-derived no-trade, no-alpha, linear-only, zero-margin, negative
  gross, multi-bucket, zero-participation, and binary64 limit cases. Check exact
  shape/type rejection and missing/zero volumes.
- [x] Implement `evaluate_schedule(scale, trade_fractions, spread, impact,
  volatility, dollar_volume, gross_return, participation)` returning costs,
  normalized PnL, separately classified thresholds, and representation status.
- [x] Add a deterministic direct-dollar oracle in tests, independently summing
  costs from `q` rather than reusing `L` and `I`. Verify doubling scale gives
  factors `2` and `2**1.5`, scaling `A,V` together preserves normalized impact,
  reversing trade signs changes nothing, and zero gross cannot give positive net.
- [x] Implement `run_lab` with fully serialized inputs, source/NumPy provenance,
  all cells, controls, and limitations. Validate custom grid values before use.
- [x] Implement strict JSON CLI with exclusive output creation. Test refusal to
  overwrite existing files and symlinks without changing their content.
- [x] Run focused tests, available lint, and the repository's bare pytest command
  using the existing interpreter only. Record every blocker/failure separately.
- [x] Run the frozen grid once, inspect retained cells/controls, save report,
  command logs, source hashes, and findings under the private capacity evidence
  directory. Report results and limitations without empirical capacity claims.

## Numerical refinement and verification record

The grid and economic assumptions above were not changed after inspecting any
outcomes. Arithmetic review added two regression-driven refinements:

1. A cost of `1 + 1e-100` followed by subtraction from gross `1` must retain the
   small negative net, even though the total cost rounds to `1` as binary64.
   Linear coefficients and `G-L` now use exact rational arithmetic for the
   supplied binary64 inputs. The impact debit is subtracted from that margin,
   rather than subtracting a rounded total cost from gross PnL.
2. Decimal precision, rounding mode, exponent limits, traps, and flags are fixed
   explicitly. A caller changing either its active Decimal context or the
   process's `DefaultContext` before module import does not change results.

Inputs are interpreted as their supplied binary64 values, not reinterpreted as
short decimal strings. Remaining 80-digit square-root arithmetic and reported
thresholds are **numerical approximations**, not certified exact signs or bounds
at arbitrarily ill-conditioned cancellation boundaries. A displayed binary64
threshold can round to the queried scale while the unrounded working value is
slightly smaller. Near-boundary decisions require more careful domain-specific
analysis; this lab does not authorize execution.

The focused suite checks an independent 110-digit direct-dollar oracle, exact
hand-derived cases, zero trades, no alpha, linear-only costs, cost exponents,
array/type/volume rejection, participation, range limits, large/small intermediate
powers and ratios, context isolation, serialized configuration replay, strict
JSON, and exclusive CLI output creation. The initial 77-test missing-feature run
failed; those 77 passed after implementation. Two cancellation regressions failed
before the rational-margin fix. The hostile DefaultContext regression failed
before context pinning. The focused suite has 88 passing tests after the independent review described below.

An initial full repository run completed with 2,131 passed, 37 subtests passed,
2 skipped, 11 warnings and 3 failed notebook-kernel tests. The failures were:

- `test_notebook_execute_script.py::test_execute_notebook_writes_to_explicit_output_path`
- `test_notebook_runner.py::test_cell_error_raises_and_keeps_the_traceback`
- `test_notebook_runner.py::test_rich_outputs_are_kept_and_metadata_is_normalised`

All three failed while starting a Jupyter kernel in this restricted environment,
matching previously reproduced baseline failures. This full run began
before the last eleven focused regressions were added; the combined integration run records the final full-suite result. No dependency installations or native builds
were attempted.

## Frozen-grid observations

The initial report retains all 90 cells. These are deterministic consequences of
the supplied assumptions, not empirical estimates or Monte Carlo results:

- `L = .00002`, or 0.2 basis points per dollar of scale, in every cell.
- All 45 zero-gross cells have no positive-net scale. Of the 45 positive-gross
  cells, 15 have no finite crossing because `eta=0`, and 30 have a finite
  model-implied crossing. There are 42 positive-net cells in the chosen scale grid.
- At base volume and `G=.001`, model-implied crossings are approximately
  `$106,711,111` for `eta=.1` and `$4,268,444` for `eta=.5`. These are **not**
  recommended capital amounts. The separately specified 1% participation cap
  rounds to `$100,000`, much lower under these assumptions.
- The participation caps at volume multipliers `.1`, `1`, and `10` round to
  `$10,000`, `$100,000`, and `$1,000,000`. With the declared exact-binary64 input
  interpretation, `.01/.1` is infinitesimally below `.1`: the queried cells at
  those printed boundaries are outside the unrounded cap. Thus the initial
  report has 36 participation-feasible cells and 18 cells both feasible and
  positive-net. Treat the boundary sensitivity as a numerical/input-convention
  caveat, not an economic finding.
- Doubling scale multiplies linear dollars by `2` and impact dollars by
  `2.8284271247461903`. Scaling scale and volume together preserves normalized
  impact. Linear-only cost has no finite impact crossing, and zero gross never
  produces positive net.
- The deliberately wrong shares-as-dollar-volume control changes the one-trade
  impact debit from `$1.25` to `$8.838834764831844` while passing numeric validation.
  Omitting scale normalization falsely labels `$1.5` as the normalized cost
  instead of `.015`. These controls illustrate hazards; neither checks real units.

## Running and replay

Run with an existing environment containing NumPy; no native extension is needed:

```sh
python experiments/capacity_scaling_lab.py --out /new/path/capacity.json
python -m pytest python/tests/test_capacity_scaling_lab.py -q
```

The output path must be new. Existing files and symlinks are refused, and
exclusive creation protects against overwrites after the preliminary check.
All report JSON forbids NaN and Infinity. API callers can replay with
`run_lab(**report['config'])`; performance metadata is not scientific identity.
Source SHA-256 and NumPy version are recorded for the standard offline
report replayer. Replay integration itself is maintained separately.


## Independent-review numerical hardening

An independent adversarial witness, outside the frozen grid, used 1,000 terms
`eta_i=2**(-i)`, `d=sigma=V=A=G=1`, and zero spreads. Its exact normalized net is
`2**(-1000)`, about `9.33e-302`; summing the impact terms at only 80 digits had
reported zero. A failing regression captured that issue before the fix.

- Whenever every relevant radical is a perfect rational square, impact and net
  are now summed as exact Fractions before binary64 output conversion. The
  scale-adjusted radicals are checked independently, since normalized impact
  can be rational even when its scale-independent coefficient is irrational.
- General radicals still use 80-digit numerical approximations. If the computed
  net magnitude is at most `1e-70` times the larger of the linear-adjusted gross
  margin and normalized impact, `net_evaluation` is
  `near_cancellation_unresolved`, net/sign outputs are null, and `precision_limits`
  identifies those unavailable outputs. This conservative warning region is not
  a certified error bound; the other approximate results remain explicitly
  labeled `numerical_approximation`. Exact rational results are labeled
  `exact_rational`. No uncertainty is disguised as a valid zero or sign in the
  detected near-cancellation region.
- Participation minima and feasibility comparisons now use exact Fractions.
  `participation_cap_exact` retains the rational threshold, and
  `participation_boundary_rounded` flags queried scales that equal the rounded
  threshold but differ from the exact-input threshold. The original boundary
  exclusions are preserved. No favorable tolerance or input reinterpretation
  was added.

The four review regressions failed before this change and then passed: exact
1,000-term cancellation, an irrational near-cancellation example, a
scale-adjusted exact radical, and participation display-boundary ambiguity.
The original v1 report and its exactly matching source snapshot were retained;
the corrected report is a separate v2 artifact. These are numerical fixes, not
post-outcome changes to the grid or economic assumptions.

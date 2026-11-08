# Signal persistence and turnover lab

## Frozen plan, written before implementation or results

Date: 2026-11-08. Base: `9740596e2d846d9e3c6653aec201be34bb69f37d`.
This third independent lab asks how signal persistence and a prespecified trading
speed change additive PnL and turnover while one-step forecasting quality stays
fixed. It does not optimize a portfolio or identify a winning speed.

### Model and chronology

The signal is stationary Gaussian AR(1), `s_t=rho*s_(t-1)+sqrt(1-rho²)*z_t`,
with variance one and `-1<rho<1`. Future return is
`r_(t+1)=beta*s_t+sigma*epsilon_t`. All innovations are independent standard
normal within a path. Positions obey
`x_t=(1-eta)*x_(t-1)+eta*k*s_t`, with `0<=eta<=1`; eta zero means cash.
After observing s_t, update x_t, charge turnover from x_(t-1), then earn
`x_t*r_(t+1)`. Epsilon_t is not observed by the policy. Report additive
unit-notional PnL, never compounded wealth. Positions and Gaussian returns are
unbounded; no leverage, financing, margin, solvency, or feasible-capital claim.
The initial position is already held in stationarity, with no inception charge;
all subsequent changes are charged and no terminal liquidation is included.

Use separate cost models: proportional `c*abs(delta_x)` and quadratic
`(c/2)*delta_x²`. Their c parameters have different dimensional interpretations;
matching numeric c does not match economic costs. These fixed adjustment rules
are not an optimal controller. Quadratic-control optimality does not transfer to
proportional fees. No risk-adjusted utility or utility ranking is reported.

### Elementary derivation and independent reference

Write `a=1-eta`, `D=1-a*rho`, `C=E[x_t*s_t]`, `V=E[x_t²]`.
For eta positive, stationarity gives
`C=a*rho*C+eta*k` and
`V=a²*V+2*a*eta*k*rho*C+eta²*k²`. Therefore:

- `C=eta*k/D`.
- `V=eta*k²*(1+a*rho)/((2-eta)*D)`.
- `Var(x_t|s_t)=eta*k²*a²*(1-rho²)/((2-eta)*D²)`.
- `Var(delta_x)=2*eta²*k²*(1-rho)/((2-eta)*D)`.
- Expected gross PnL is `beta*C`.
- Expected absolute turnover is `sqrt(2/pi)*sqrt(Var(delta_x))`.
- Expected proportional cost is c times absolute turnover; expected quadratic
  cost is `(c/2)*Var(delta_x)`.
- Forecast error MSE is `sigma²`; signed signal–return correlation is
  `beta/sqrt(beta²+sigma²)`, independent of rho. The all-zero return case has
  undefined correlation and will be labeled explicitly.

Eta zero is the distinct degenerate cash law x=0, not an arbitrary inherited
position. Eta one gives `C=k`, `V=k²`, `Var(delta_x)=2*k²*(1-rho)`, absolute
turnover `2*abs(k)*sqrt((1-rho)/pi)`. Formulas are original elementary algebra,
not attributions to unread book chapters. A test-only discrete Lyapunov solve
for state `(s,x)` supplies an independent covariance reference. Rational
arithmetic for analytic moments preserves supplied binary64 coefficients and
avoids spurious intermediate overflow/cancellation. Nonrepresentable nonzero
moments, or nonfinite simulation arithmetic, raise explicit arithmetic errors,
not economic conclusions. Numerical square roots and simulation remain binary64.

### Frozen experiment

- Seed 2026110823; 256 independent replications; 512 scored periods each.
- rho `[-.5,0,.5,.9]`; eta `[0,.1,.5,1]`.
- beta `[0,.001]`; c `[0,.0001,.001]`; k 1; sigma .02.
- Proportional and quadratic costs, each separately named: all 192 grid cells.
- SeedSequence `[seed,replication]` yields a standard-normal initial signal,
  an independent residual normal, and a chronological horizon-by-2 matrix of
  signal and return innovations. Common primitives across every grid cell,
  independent streams across replications. Preserve primitive hashes.
- Initialize each policy jointly stationary:
  `x_prev=C*s_prev+sqrt(Var(x|s))*residual`. No burn-in or initial-zero bias.
- Keep every cell's per-replication path-mean metrics, analytic reference,
  empirical mean, pointwise Monte Carlo SE using independent whole paths, and
  deviation from theory. Serially correlated dates are never independent units.
  Pointwise errors are descriptive, not simultaneous bounds or test thresholds.
- Beta zero is the negative control: expected gross is zero, expected net cannot
  exceed zero under either nonnegative cost. Realized samples may be positive.
- Do not rerun for favorable outcomes, select a speed, or use statistical
  significance as a pass criterion. The default grid is run once per retained
  source version; deterministic replay is a reproducibility check, not new data.

### Test-first implementation sequence

1. Write hand-derived full-adjustment and cash tests, a separately solved
   Lyapunov covariance, chronology/future-suffix invariance, stationarity,
   zero-cost and zero-alpha controls, range/type/finite arithmetic tests,
   configuration replay and CLI overwrite refusal. Record missing-code red run.
2. Implement validated analytic moments, primitive-driven simulation and the
   fixed grid runner in `signal_turnover_lab.py`, with no production API edits.
3. Verify all focused tests and available lint with the existing interpreter.
   Do not install, build native code, run the full suite, or write remotely.
4. Retain frozen report, exact source snapshot, hashes, runtime, command logs,
   and all results in the private signal evidence directory. Replay registration and final combined checks are separate integration steps. Change only this document, lab
   module and its dedicated Python test file.

### Source/access qualification

Study 3 of the 2026-11-08 implementation-cost design note supplies context.
The author-hosted March 2012 Gârleanu/Pedersen draft,
https://nbgarleanu.github.io/DynTrad.pdf, was consulted in
model section 2 and propositions 1–3. Its partial-adjustment optimality assumes
linear predictors, a mean-variance objective and quadratic costs. This lab
neither implements that objective nor claims that theorem for its fixed rules.
Paleologo publisher contents identify topics (including AR(1)), but chapter
text was not accessed. No Paleologo theorem is claimed or reproduced. The
source-access record documents unavailable previews without bypassing access
restrictions. This is synthetic methodological work, not a live trading result.

## Retained default run and focused verification

The frozen grid was run once using Python 3.12.14 and NumPy 2.4.6. All 192 cells
retain nine path-mean metrics for each of 256 independent replications, primitive
hashes, stationary references, pointwise Monte Carlo errors and deviations.
No source-grid tuning or outcome-based selection was performed. The measured
lab kernel took 0.682176 seconds and shell wall time was 1.226 seconds on this
executor; these are measurements, not performance promises. The optional
`/usr/bin/time` command was unavailable, so the shell timing builtin was used.

The analytic full-adjustment example at beta .001, sigma .02 and k 1 has the
same forecast MSE .0004, signal–return correlation about .04994, and gross mean .001
for every rho. At proportional c .001:

| rho | Expected absolute turnover | Analytic net PnL/period | Sample mean | Path-based MC SE |
| --- | ---: | ---: | ---: | ---: |
| -.5 | 1.381976598 | -.0003819766 | -.0004375933 | .0000550210 |
| 0 | 1.128379167 | -.0001283792 | -.0002509065 | .0000532616 |
| .5 | .797884561 | .0002021154 | .0000743043 | .0000527803 |
| .9 | .356824823 | .0006431752 | .0005552841 | .0000567424 |

This is a conditional model counterexample to forecast quality determining
trading costs, not evidence about a market or a preferred speed. Partial
adjustment also changes exposure covariance and position risk, so its lower
turnover is not a free reduction in cost at unchanged strategy risk. Both cost
models and all four adjustment speeds are retained without ranking. The
quadratic model at equal numeric c has different costs and different units.

Sampling uncertainty is visible and is not treated as a failure or hidden:
at beta zero, rho zero, eta one and zero cost, the retained gross mean is
-.0001209895 with Monte Carlo SE .0000534524, versus an exact expectation of
zero. Shared innovations make deviations across cells dependent. No significance
screen was used, and no seed was changed. A Monte Carlo error is not an
out-of-sample performance guarantee or a simultaneous interval.

Focused verification before the default run: **71 tests passed**, with Ruff
check and format check passing. Tests include an independent discrete Lyapunov
solve, analytic initialization covariance, deterministic chronology, future
suffix and return-innovation invariance, independent primitive reconstruction,
path-based SE, strict config replay, numerical range failures, and CLI refusal
to overwrite files or symlinks. The initial 69 missing-implementation failures
were retained as the test-first red run. One first-implementation fixture
incorrectly expected overflow for a finite partial-adjustment gross mean; the
fixture was corrected to full adjustment, where its gross mean is outside
binary64. No production code, native build, install or full-suite run was done
during focused implementation. Replay registration and independent review are
separate final checks.

The exact source SHA256 for this retained run is
`0d5b94c9d2836c4620a93b9b6cbc179e7789ae434e50cc4641693ef67d71454c`.
The report SHA256 is
`cfcd9279c26097ecce68b80f22cfbfd86342cd55d792aa784d8a82505de94458`.
A source snapshot accompanies the report, since the reported Git base alone
does not identify uncommitted lab source. These artifacts are reproducibility
records, not an authenticated publication or an assertion of production fitness.


## Independent numerical review refinement

Independent review reproduced an extreme-domain defect in the first retained
source: beta `1e-300`, sigma `1e100` has a nonzero signal–return correlation of
about `1e-400`, but the binary64 division returned zero. This contradicts the
lab's explicit nonrepresentable-nonzero output contract. A regression first
failed against that source; the corrected calculation checks representability
and raises `ArithmeticError` instead. No economic result is fabricated from
that out-of-range quantity. Original source, tests, report and failed regression
remain retained as first-version evidence.

The compatibility field `forecast_correlation` means **signed
`Corr(s_t,r_(t+1))`**. It is not `Corr(beta*s_t,r_(t+1))`, whose sign would be
nonnegative and which is undefined when beta is zero. The signed signal–return
quantity is zero at beta zero when sigma is positive, and undefined when both
return components are zero. Tests explicitly pin these distinctions.

Reviewed source SHA256:
`1ee5d1713328855a32c7a4f1b5f36a347fc552f55381929118e14a2eaab907d9`.
Reviewed report SHA256:
`a14582842312e18b347d257149ced65a0457b56614ea6f4d9661e511bf9f3737`.
The original report's serialized config was replayed unchanged once under the
corrected source, retaining exactly the same scientific output. This was the
same sample, not a fresh draw or a search for favorable results. The replayed
kernel took 0.774164 seconds. **73 focused tests passed**; Ruff check and format
check passed. Final combined-suite and replay acceptance are recorded separately. Remaining
arithmetic is binary64 numerical work, not certified interval arithmetic or a
proof of exact signs near arbitrary cancellation boundaries.

## Independent reconstruction and replay

A separate implementation reconstructed all 256 primitive hashes and all 442,368
stored path-metric values across the 192 cells. Maximum absolute path-metric
difference was 6.22e-15; an independently solved stationary Lyapunov reference
agreed within 3.34e-16. Whole-path Monte Carlo errors and all descriptive paired
contrasts against full adjustment were recomputed. These are numerical consistency
checks, not evidence that the assumed process describes real returns.

The 73 focused tests plus two replay-integration tests pass. Replay rejects
altered per-path outputs and altered Monte Carlo errors, and a fresh-process
replay of the reviewed report matches its scientific output exactly. The optional
post-run contrast table retains every cell and explicitly uses paired whole-path
differences; it introduces no new draws, selected winner or simultaneous coverage
claim. Full-suite acceptance is recorded at the final committed checkpoint.

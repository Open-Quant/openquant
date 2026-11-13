# Execution timing and alpha decay: offline research lab

This synthetic, prespecified acquisition experiment compares timing, not optimized
execution or investment performance. It complements `signal_turnover_lab.py`
(stationary signal smoothing) and `capacity_scaling_lab.py` (fixed-schedule scaling).
No empirical forecasts, calibrated fills, parameter fitting, policy selection,
uncertainty intervals or trading recommendations are produced.

## Model, units and timeline

There are T equal-duration slots indexed t=0,...,T-1. Starting inventory is zero.
All policies must acquire exactly Q>0 shares by the start of the final interval.
Prices, cash and costs use one currency. Alpha a>=0 is the **absolute currency per
share expected price increment in the first slot**, not a return or total horizon
alpha. Decay rho in [0,1] is dimensionless per slot. The increment in slot t is
mu[t]=a*rho^t, including mu[0]=a when rho=0. The alpha clock begins at t=0 and never
restarts when trading starts.

At t, buy q[t] shares at the pre-increment mid-price plus temporary execution cost,
then hold x[t]=sum(q[0:t+1]) shares over [t,t+1]. Costs are
s*q[t]+eta*q[t]^2, with s in currency/share and eta in currency/share^2 for this
fixed slot duration. There is no permanent impact. Changing slot duration without
rescaling inputs is a different model.

Latency L is an integer number of blocked initial slots, 0<=L<T. Three schedules
are fixed before results are computed:

- front: all Q at t=L;
- uniform: Q/(T-L) in every slot from L through T-1;
- back: all Q at t=T-1.

Latency compresses uniform execution into remaining slots. It does not silently
drop orders or relax Q. L>=T is rejected as terminal-completion infeasible.
No participation or fill constraints are imposed, so completion is assumed rather
than an empirical execution guarantee. Back-loading is intentionally unchanged by
latency until no slots remain.

Expected marked PnL is sum(x[t]*mu[t])-s*Q-eta*sum(q[t]^2).
The common terminal inventory is Q and **is not liquidated**. There is no terminal
sale or sale fee, financing, solvency or stochastic risk model. Comparing incomplete
orders would be unfair; this lab rejects that alternative by construction.

The immediate, costless benchmark is Q*sum(mu[t]). Opportunity cost is benchmark
minus gross marked PnL; implementation shortfall adds both execution costs.
It is a counterfactual benchmark, not another tradable strategy.

## Exact accounting references

The implementation uses exact rational arithmetic after converting scalar inputs
to finite binary64. Report `exact` fields are rational strings; float display fields
are rounded only at the end. Underflow/overflow yields null plus
`representation_limits`, never a misleading zero or Infinity. Exact refers to the
binary64 inputs, not a claim that decimal literals or economic assumptions are exact.

Hand fixture: Q=2, T=3, a=1, rho=1/2, s=1/4, eta=1/2, L=0.
Uniform trades are (2/3,2/3,2/3), holdings (2/3,4/3,2), and increments
(1,1/2,1/4). Gross PnL=11/6, spread=1/2, impact=2/3, net=2/3.
The benchmark is 7/2 and opportunity cost=5/3. Front execution at L=2 instead earns
only 1/2 gross, while still acquiring both shares and paying impact=2.

Tests independently reconstruct purchase cash flows from a starting price of 100,
then add terminal marked inventory. This telescoping ledger matches the exposure
formula for all schedules and latencies. Negative controls illustrate three errors:
restarting the alpha clock, dropping blocked orders, and applying price increments
to pretrade rather than posttrade holdings. All grid cells are retained; no winner
is selected. Results are deterministic apart from elapsed time and provenance.

## Run and integrate

From the repository root, with the existing offline-research environment:

```sh
python experiments/execution_timing_lab.py --out /tmp/timing-report.json
python -m pytest python/tests/test_execution_timing_lab.py -q
```

The output must not already exist. `run_lab` accepts complete keyword defaults and
emits the standard schema-1 synthetic report with config, cells, controls,
negative_controls, limitations, provenance and performance. The smoke configuration
is `experiments/configs/offline-smoke/execution_timing.json`.

`execution_timing` is registered to `execution_timing_lab.py` with no sibling
dependencies in the common runner/replayer. No extra runtime dependency is needed.
NumPy version is recorded solely to comply with the existing replay identity
contract; arithmetic uses the standard library.
The standalone CLI and direct `run_lab` import are also supported. Review
configurations before running: exact rationals and unbounded horizon/grid sizes
are not a resource sandbox. Numeric validation cannot detect inconsistent units.

## Scope and reference

The quadratic temporary-cost framing is related to the execution literature, e.g.
Almgren and Chriss, *Optimal Execution of Portfolio Transactions*, Journal of Risk
(2001), DOI: https://doi.org/10.21314/JOR.2001.041. This lab does not implement their
full optimal execution model, solve an optimal control problem, or reproduce a
published empirical result. It is a small acquisition-accounting research example.

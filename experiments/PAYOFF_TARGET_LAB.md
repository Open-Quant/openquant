# Positive signs and arithmetic payoff are different targets

This study applies fixed payoff maps to the same IID Bernoulli sign paths. It
shows why evidence for a positive-sign probability above one half is insufficient
for positive expected gross or net arithmetic payoff. Payoff units are arbitrary.
There is no compounding, capital constraint, leverage, market model or trading
recommendation. A positive indicator is a gross win; fixed cost is subtracted
from every observation actually included, including the stopping observation.

For X~Bernoulli(p), a gross win pays +w and a loss pays −l, with w,l>0 and cost
c>=0 per row. Gross positive-sign probability is p for every such magnitude map,
while the expected net payoff is

`mu = p*w - (1-p)*l - c`.

If c=w, gross wins become net ties; if c>w, even gross wins are net losses. The
reported evidence always concerns gross signs and is not silently relabeled as
net-sign evidence. The module reports both sign targets explicitly.

## Bounded stopping reference

The frozen sequential mixture supplies the first evidence-crossing time. Set
`tau=min(first_crossing,H)`, taking tau=H when no crossing occurs. Every path is
retained. The fixed-H comparator always includes H observations. The stopping
rule sees past and current signs, never future signs or payoff-map selection.

For IID integrable payoffs and tau<=H,

`E[sum_{i=1}^tau Y_i] = mu * E[tau]`.

This signed bounded version follows directly by writing the sum as
`sum_{i=1}^H Y_i * 1{tau>=i}`. The event `{tau>=i}` depends only on observations
before i, so its indicator is independent of Y_i and each expectation factors.
The sum of those event probabilities is E[tau]. This derivation needs neither
positive payoffs nor an unbounded-time limit. [Koutsoupias's Oxford lecture notes](https://www.cs.ox.ac.uk/people/elias.koutsoupias/pc2018-19/lectures.html)
states the bounded optional-stopping theorem; applying it to the centered payoff
sum gives the same identity. Its separately stated nonnegative Wald theorem is
not cited as if it directly covered these signed payoffs.

Consequently, stopping at gross-sign evidence does not reverse the sign of
unconditional expected stopped net payoff. A negative mean multiplied by positive
expected observation count remains negative. This concerns expected arithmetic
payoff, not realized losses on every path. Conditional means among paths that
cross are descriptive and need not equal mu*E[tau|cross]. The no-cross paths and
their contribution are essential; deleting them changes the population analyzed.

## Frozen protocol and independent checks

Predeclared horizons {64,256}, 5,000 paths per (p,H), seed23, alpha=.05 and the
frozen equal mixture q=(.55,.65,.8). Eight fixed payoff maps are evaluated:

| p | Win | Loss magnitude | Costs |
|---|---:|---:|---|
| .60 | 1 | 1 | 0 and .25 |
| .60 | 1 | 2 | 0 and .25 |
| .60 | 2 | 1 | 0 and .25 |
| .85 | .2 | 1 | 0 and .25 |

At a given (p,H), all magnitude/cost maps share exactly the same signs, stopping
times and gross-sign evidence; only arithmetic payoffs change. No winning map is
selected after seeing results. This is a controlled target comparison, not a new
multiplicity-adjusted inference procedure across maps.

An independent absorption recurrence tracks terminal (time,positive-count,
crossed-status) probabilities. It removes crossing paths at each time, then
retains all remaining no-cross paths at H. Expected terminal net payoff is
computed directly from that distribution and compared with the Wald identity
using expected tau from the frozen sequential forward program. A separate fixed-H
terminal law checks H*mu. Small exhaustive binary paths independently validate
terminal probabilities, costs, endpoint p values, and signed Wald expectations.
These finite-law sums are evaluated with floating arithmetic, not certified exact
arithmetic. Exact evidence boundaries come from the frozen dependency.

Reports retain every stopping time, positive count at stopping/H, arithmetic payoff
under each map, no-cross contribution, conditional descriptive means, Monte Carlo
standard errors, canonical sign-data hashes, source/dependency hashes and config.
Magnitude/time combinations that cannot be represented finitely are rejected;
sample means and standard errors use scaling to avoid unnecessary overflow.

```sh
OPENBLAS_NUM_THREADS=1 python experiments/payoff_target_lab.py --out payoff.json
```

## Retained findings

The frozen protocol completed 20,000 shared sign paths across four (p,H) groups,
with 16 payoff-map/horizon scenarios. All 32 stopped/fixed mean checks were retained;
maximum finite-sum versus Wald/fixed expectation discrepancy was 2.85e-14, and the
largest absolute Monte Carlo standardized error was 2.215. Errors across maps are
correlated because they deliberately reuse paths. Final source SHA256 is
`ebb2a3b7d66c53469e1caa18c0bbfe4e9e215dc9892accc14e41e0151b225b7e`, with sequential
dependency `0ae963b6ff5a8a6c4a03dc900200dc78b71700945fcccc93f9947c8e3421a116`.

At p=.6,H=64, every magnitude/cost map had the same gross-sign crossing probability
.221201 and expected observation count 57.1102. The win 1/loss 1 map had expected
stopped payoff +11.4220 without costs and −2.8555 with cost .25. Changing only the
loss magnitude to 2 gave −11.4220 even without costs. Thus neither sign evidence
nor evidence-based stopping identified the sign of expected arithmetic payoff.

Discarding no-cross paths produced a further failure. For win 1/loss 1/cost .25,
mean stopped net payoff among crossing paths was +7.3155, versus −5.7444 among
no-cross paths and −2.8555 unconditionally. For win 1/loss 2/cost 0 the corresponding
values were +6.8667, −16.6166 and −11.4220. Those positive selected-subset means
do not contradict Wald's unconditional identity.

At p=.85,win .2/loss 1/cost .25, every net observation was negative although the gross
positive-sign probability was .85. At H=64 the gross-sign crossing probability was
.999884 and expected stopped net payoff −3.5860. Positive gross evidence remained
correct for its stated target while failing to establish the different economic
target. No compounding or real-market conclusion follows from these examples.

Private evidence includes `payoff-target-final/report.json`, `summary.json`, the
reproducible `summarize_and_plot.py`, and PNG/PDF target-mismatch and no-cross
selection figures. Raw stopped/fixed payoff arrays and no-cross contributions
are preserved. All maps were fixed in advance and all outcomes were retained.

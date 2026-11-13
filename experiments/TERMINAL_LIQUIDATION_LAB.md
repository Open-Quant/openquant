# Terminal inventory and liquidation accounting lab

Research-only synthetic accounting spike. This is not a trading strategy, optimizer,
execution simulator, profitability estimate, or evidence of market realism. It extends
the finite-horizon question excluded by `SELF_FINANCING_LAB.md`: what changes when
terminal inventory must become cash? The existing ledger and numerical core are unchanged.

## Contract and event order

Initial capital is one unit of currency, held in cash. Returns are dimensionless
simple returns strictly greater than -1; arrays have shape (periods, risky assets).
Each period first rebalances at that period's opening mark, then applies that
period's returns. Final liquidation occurs **after the last return**, at that same
terminal mark, once. There is no additional period, target rebalance, price movement,
or charge on cash at exit.

Prespecified policies are cash, equal-weight buy-and-hold, and equal-weight daily
rebalance. Cash has zero interest and never trades. The other policies enter fully
at period zero and remain fully invested. Entry fees and ongoing buy/sell fees are
separate scalar fractions of actual traded risky notional, both in [0,1). Exit fees
are a third independently declared fraction in [0,1), charged on sales only. Thus
0.001 is 10 basis points, not 0.001 percent. Fees, holdings, and traded notionals in
reports are **currency units per initial capital**, not a sum of weight fractions.
Cumulative fees are uncapitalized cash charges at their event dates; they do not
measure the difference from a counterfactual fee-free terminal portfolio.

No borrowing, shorts, partial fills, bid/ask spread, price impact, minimum fees,
taxes, settlement lag, cash yield, stochastic liquidation prices, or bankruptcy
recovery is represented. No signals, parameter fitting, policy selection, hypothesis
tests, confidence intervals, or annualization are performed.

## Equations and independent oracle

Let pretrade wealth be W and risky weights w. For a fully invested postfee target
v, existing `self_financing_lab.rebalance` solves

    alpha + c * sum(abs(alpha*v_i - w_i)) = 1.

The new ledger uses posttrade risky notionals W*alpha*v_i and fee
W*c*sum(abs(alpha*v_i-w_i)). Cash is explicitly set to zero on entry, never inferred
from a rounded sum of risky weights. The reused solver's roundoff/simplex policy
is unchanged; its source hash is a required replay dependency.

For terminal risky notionals h_i >= 0 and explicit cash C >= 0:

    H = sum(h_i)
    marked wealth = C + H
    exit sale notional = H
    exit charge = e * H
    realized cash = C + (1-e) * H
    realized cash + exit charge - marked wealth = 0 (up to floating point).

Multiplying the surviving inventory by (1-e) rather than subtracting the charge
from total wealth avoids catastrophic cancellation near e=1. Cash is not taxed.
The public `liquidate` helper also supports mixed risky/cash terminal inventories;
the fixed investment policies in this study do not claim that broader allocation
support during the path. Zero terminal inventory is allowed; zero total terminal
wealth is meaningful for the helper but not for a simulation's log-wealth output.
Nonfinite or nonpositive aggregate wealth in the simulation is a failed row,
never clipped wealth. Individual holdings can underflow while aggregate wealth
remains positive; this lab does not certify every extreme-scale component.

Hand-checkable flat-price round trip: entry c=1/10 buys 10/11 units of risky
notional, entry fee=1/11. Exit e=1/5 costs 2/11 and realizes 8/11. The ongoing
fee has no role for a one-period holding. The test uses a different ongoing fee
to detect accidental conflation.

The two-period independent Fraction oracle starts with holdings [5/11,5/11].
After returns [1/4,0], they are [25/44,5/11]. At ongoing fee 1/5, daily equal
weighting trades toward equal dollar holdings x; the sell and buy amounts sum
to 5/44, so the fee is 1/44 and x=1/2. After [-1/5,1/10], terminal marked wealth
is 19/20. Exit fee 3/10 realizes 133/200. This oracle uses dollar holdings and
rational arithmetic, not the production root solver or a replica of its code.

## What the controls can establish

- For cash, marked and liquidated wealth are identically one and all fees zero.
- Flat prices isolate entry and exit costs. A common small positive return can
  put an invested policy above cash when marked and below cash after selling.
- With both policies fully invested and a common scalar exit rate, realized
  wealth is (1-e) times marked wealth. Liquidation **cannot reverse their mutual
  ranking**, though it can reverse either policy's comparison with cash.
- Changing only the exit rate cannot alter any preterminal holdings or records.
- Terminal fee burden varies with actual terminal inventory, not initial capital
  or stale target weights. Charging initial capital at exit is the wrong model.

Scenarios are flat prices, common small gains, opposing trend, reciprocal
alternation, independent zero-mean signs, and common zero-mean signs. Random paths
use SeedSequence(seed, family index, replication), paired across policies and exit
rates. They are illustrative controls, not calibrated market models. Comparisons
use literal floating-point signs; near ties should be inspected, not described as
statistically significant. A `cash_comparison_reversal` is strictly marked>1 and
realized<1; equality is not a reversal.

## Run and replay

    python experiments/terminal_liquidation_lab.py --out /tmp/liquidation.json

The standalone command uses a modest default study and refuses to overwrite.
`run_lab(**config)` accepts horizon, repetitions, entry_fee, trade_fee,
exit_fee_rates, and seed. Scientific paths and report rows are deterministic;
wall-clock timing is not. The offline smoke config is
`experiments/configs/offline-smoke/terminal_liquidation.json`.

Schema v1 follows the existing offline runner/replayer: lab name
`terminal_liquidation`, config, NumPy/source/dependency provenance, performance,
explicit paths with little-endian binary64 SHA-256 hashes, and paired report rows.
The shared registry registers `terminal_liquidation_lab.py` and declares
`self_financing_lab.py` as a replay dependency. No new packages beyond
NumPy and the existing standard library are needed.

## Implementation-ready follow-up, not implemented here

1. Preserve distinct endpoint fields in any downstream comparison UI. Never
   relabel marked inventory as realizable cash or compare unlike horizons.
2. If cash allocation policies are introduced, track explicit cash and inventory
   throughout the path; do not reconstruct cash as one minus rounded weights.
   Add exact mixed-inventory and cash-yield oracles before supporting them.
3. If per-asset exit rates or liquidation price shocks are needed, declare both
   separately and use cash + sum((1-e_i)*h_i*exit_price_relative_i). Add reversal,
   zero-inventory, failed-exit, and near-total-loss tests before changing the API.
4. Model spread, impact, minimum fees, settlement delay, or partial fills as new
   assumptions and independent controls. This proportional mark-price exercise
   cannot justify those capabilities or claims of executable performance.

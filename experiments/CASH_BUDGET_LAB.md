# A cash budget is conditional on execution assumptions

This original, offline research slice connects a desired buy quantity to the
[exact fill ledger](FILL_LEDGER_LAB.md). It adds an explicit pre-trade budget and
an audit of its assumptions. It does not alter `run_ledger`, any production API,
or any broker order. The helper is not an optimal position-sizing rule.

## Run and replay

```bash
python -m experiments.run_offline_lab run \
  --lab cash_budget \
  --config experiments/configs/offline-smoke/cash_budget.json \
  --out /tmp/cash-budget-new.json
python -m experiments.replay_offline_report /tmp/cash-budget-new.json --lab cash_budget
```

The default study retains 72 scenarios: three starting cash amounts, two price
caps, two order-lot sizes, two fixed fees and three fill patterns. The smoke
configuration has 12. Each report retains the budget, complete ledger and audit;
zero orders and incomplete orders remain visible. There is no ranking or
selection among scenarios. The runner refuses an existing output, hashes the
lab and ledger source, and exact replay compares scientific contents. Replay
is not proof of authenticity or statistical validity.

## The missing contract

The ledger correctly rejects a candidate buy that requires more cash than is
available. It does not silently resize it. A calculation that divides cash by
today's mark is insufficient: the later ask can differ, one order can fill in
several pieces, and each piece can incur a fixed fee. Rounding an order up to a
lot boundary can also make it unaffordable.

We considered three paths: silently clip candidate ledger fills, adapt arbitrary
CPCV paths, or expose a separate pre-trade budget. The last is the limited choice
here. Clipping changes the declared order policy. CPCV composition additionally
requires model/split provenance and boundary-capital decisions already listed in
[the segment guide](EXECUTION_SEGMENTS_LAB.md). Neither is smuggled into this lab.

## Exact envelope and proof

Inputs are available cash C, protected cash R, desired shares D, positive order
lot L, positive execution-price cap P, fixed fee f per positive fill, proportional
notional fee r, and positive maximum fill count N. C, R, D, f and r are
nonnegative; R cannot exceed C. All are exact fractions of the supplied decimal
values. N is an actual positive Python integer, not a float or boolean.

For an order with q > 0 shares, reserve:

    W(q) = q * P * (1 + r) + N * f.

The chosen quantity is:

    q = L * min(floor(D / L), floor(max(0, C - R - N*f) / (L*P*(1+r)))).

If q = 0, no order is created and W(0) = 0. No fixed fee is charged for doing
nothing. Otherwise `cash_floor = C - W(q)`. This is the largest quantity on the
declared lot grid, at most D, satisfying this particular conservative envelope.
It is not necessarily the largest quantity affordable along the realized path.

Suppose all actual executions are positive buys, at prices at most P, total
filled quantity at most q, no more than N positive fills, and commissions exactly
f + r times notional for each fill. Then total spend is at most W(q), since
all quantities, prices and fee rates are nonnegative. Every prefix also costs no
more than the total. Therefore, with no other cash consumer, cash never falls
below C-W(q), which is at least R. This statement does not require the order to
complete. Conversely, a larger permitted lot quantity either exceeds D or fails
the declared W bound; the tests check maximality by enumerating candidates rather
than repeating the formula.

This proof is conditional. A market order does not enforce P or N. A limit price
can constrain executions, but it does not by itself bound the number of fills or
the eventual fee schedule. A backtest that uses future maximum ask to construct
P has introduced hindsight even if it stamps an earlier timestamp on the value.

## Runnable hand example

```python
from datetime import UTC, datetime
from experiments.cash_budget_lab import size_buy

at = datetime(2026, 1, 1, tzinfo=UTC)
b = size_buy(
    cash=100, desired_shares=20, price_cap=10, lot_size=1,
    fixed_fee=1, fee_rate="0.01", max_fills=2,
    information_at=at, decision_at=at,
)
print(b.shares, b.worst_case_spend, b.cash_floor)
```

```text
9 929/10 71/10
```

Two fills of 4 and 5 shares at 10 cost 40 + 1.4 and 50 + 1.5. Total spend is
92.9, leaving 7.1. A single fill costs less because only one fixed fee is paid.
No fills cost zero. These are accounting identities, not investment returns.

`cash_reserve=20` reduces the planned order to 7 shares. Protected cash is simply
unavailable to this order; it does not claim to cover a liquidation, tax or future
fee without a separately established bound.

## Failure controls and completion

The report explicitly retains four negative controls:

- An order budgeted for at most price 10 executes 9 shares at 11 in two fills.
  Spend is 101 against cash 100. The original ledger rejects the second fill.
- Cash 95, fee 5 and a budget for one fill permits 9 shares at 10. Splitting into
  two fills requires 100. Again the ledger rejects the second fill.
- Ignoring two fixed fees gives a 10-share order at price 10 that actually needs
  102 rather than 100.
- Rounding up with cash 105, lot 2 and price 10 produces 12 shares costing 120;
  rounding down permits 10.

`audit_fills` retains each broken price/count/fee/quantity/timing assumption. It
reports actual spend, cash, remaining quantity and completion separately. In the
72-scenario study, partial orders can satisfy the affordability envelope while
remaining incomplete. A zero-sized request is complete without an order, which
must not be confused with a filled investment opportunity.

The helper assumes an unmodified `BuyBudget` from `size_buy`, and all genuine
ledger fills allocated to that isolated budget. Manually fabricating a dataclass,
omitting a fill, or rewriting records is outside that contract. Neither the
budget nor the audit authenticates a data source or proves complete fill history.
The single `information_at` is declared metadata for the input set, not proof
that each input was genuinely known then. Timestamps must be aware UTC and cannot
place information after the decision; actual fills must follow the decision.

## Integration requirements before a production adapter

1. **Scope and ownership.** Reserve cash atomically per account, currency,
   instrument and order. Prevent two orders from each spending the same C-R.
   This helper has no reservation book and carries no instrument/currency IDs.
2. **Source provenance.** Store each bound's source/version and availability time,
   the raw desired quantity and the reason for any size reduction. Do not
   determine assumptions from the execution tape being evaluated.
3. **Enforcement.** Specify a price-limit/fee-cap/fill-count policy. Distinguish
   assumptions from venue-enforced constraints, and account for cancellation
   races. If a bound fails, reject/re-plan explicitly rather than assert safety.
4. **Quantity semantics.** The lot grid is for submitted order quantity only.
   Fractional partial fills are allowed in this synthetic ledger. A venue with
   per-fill lot rules, minimum notional or residual-dust rules needs another
   contract and its own tests.
5. **Fee semantics.** Fees here are nonnegative fixed-plus-proportional charges
   in the same cash currency per positive execution. Order-level fees, tiers,
   rebates, minimums, foreign-currency or asset-denominated fees are unsupported.
6. **Terminal state.** Entry affordability does not imply a liquidating sale is
   feasible or that a segment can hand off cash. Keep the existing explicit
   terminal bucket and cash-handoff requirements; do not turn marked shares into
   spendable cash.
7. **Evaluation.** Keep no-orders, partials, bound breaches and rejected scenarios.
   Any later comparison must use consistent obligations, data chronology and
   fees. These arithmetic checks establish no profitability or capacity.
8. **Numerics and runtime.** Fractions model declared decimal inputs, not venue
   precision or binary float expansions. Trusted local configs have no workload
   budget; huge denominators or grids can consume resources. Introduce a separate
   bounded-input policy before processing untrusted workloads.

## Primary sources and attribution

- Giuseppe A. Paleologo, *The Elements of Quantitative Investing* (Wiley, 2025),
  [official publisher sample, chapter 1](https://catalogimages.wiley.com/images/db/pdf/9781394265459.excerpt.pdf),
  especially printed pages 4–5 and 16–18. The accessible sample connects liquidity
  to size, time and cost and distinguishes pre-trade analysis, construction and
  post-trade evaluation. This motivated separating a declared budget from actual
  fills and their audit. The sample was readable on November 11; this is not a
  claim of access to the full book or implementation of a later book chapter.
- Cristiano Arbex Valle, [Portfolio optimisation: bridging the gap between theory
  and practice, arXiv:2407.00887v2](https://arxiv.org/abs/2407.00887), abstract,
  revised September 30, 2024. It distinguishes desired portfolio weights from
  implementable trades with costs and integral lots. That supports treating order
  conversion as a separate contract. We do not claim to implement or reproduce
  the paper's optimizer or empirical results.
- NautilusTrader's [official position guide](https://github.com/nautechsystems/nautilus_trader/blob/develop/docs/concepts/positions.md),
  read November 11, documents fill-driven position updates and commission
  currencies. This is useful prior art for stating this lab's narrower fee and
  currency assumptions. No Nautilus code was copied or compatibility claimed.

The envelope, proof, failure controls and implementation are original. No paid
content, unofficial book mirror or paywall bypass was used. No copyrighted
source text or full PDFs are included in the deliverable.

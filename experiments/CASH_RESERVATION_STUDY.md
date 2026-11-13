# Shared cash: reservation and execution receipts

Offline experimental continuation, 2026-11-12. No production integration.

## Decision and contribution

Keep the existing isolated [cash envelope](CASH_BUDGET_LAB.md), but do not treat
its output as account-wide buying power. Two individually affordable $92 orders
can encumber $184 against the same $100. This study implements a separate,
locked, in-memory reservation account and reproducible adversarial histories.
It is accounting research, not alpha, a fill forecast or a live risk service.

Alternatives considered: document the gap only; add a broker adapter; implement
this small offline state machine. The third gives falsifiable evidence without
credentials, execution costs or changing production APIs. Durable reservations,
provenance and real cancellation reconciliation are deliberately not implied.

## Conditional invariant

Let C be cash, L the account floor, and R the sum of all outstanding envelopes.
Available funds are C − L − R. The initial state has R = 0. Reservation sizes
inside a single-object lock and increases R by at most available funds. For a
partial buy q at price p no greater than cap P, proportional fee r and fixed
fee f, cash decreases by q p (1+r)+f and R decreases by q P (1+r)+f. Therefore
availability cannot decrease on a compliant fill. Completion removes unused
fixed-fee reserve too. Finalization releases residual R only under the declared
complete-history assumption. Each transition preserves C ≥ L + R ≥ L.

`CashReservations` exposes immutable snapshots. Caller snapshots are never
accepted as spendable balances. Exact Fraction arithmetic avoids tolerance-based
cash creation. Order IDs are unique, including zero-size decisions; execution
IDs are idempotent only when the full payload agrees. Conflicting redelivery,
unknown orders, too many fills, overfills, bad chronology, prices and fees are
rejected before mutation. All legitimate cash consumers must share one object.

### Finalization is not cancellation

`finalize` means a synthetic history is complete, not “cancel requested.” There
is no timeout-based release. Fill-count exhaustion does not imply cancellation.
A late novel fill after finalization is rejected, but a prior exact event retry
is harmless. In a real system rejecting an actual execution cannot undo it:
quarantine/reconciliation and a safe stop would be required. A cancel/fill race
can invalidate a naive release policy. No sequence-watermark, external broker,
persistence or crash-recovery guarantee exists in this prototype.

## Run and inspect

From the repository root with NumPy installed (inherited cash-budget import):

```bash
python -m unittest discover -s python/tests -p test_cash_reservations.py
python -m experiments.cash_reservation_study > /tmp/reservations.json
python -m experiments.cash_reservation_study /tmp/reservations.json
```

The runner's default 100 cases use seed 4815, three waves of four competing
orders, fractional fills, varied caps/fees, zero orders, incomplete executions,
redelivery and explicit terminal histories. Receipts include every account
snapshot. An independently maintained arithmetic reference checks cash and
remaining obligations at every transition. Replay reconstructs the entire report
and requires exact equality, not only a checksum or summary match. This does not
authenticate inputs or establish real historical availability.

The tests also exercise simultaneous reservations, duplicate delivery races,
finalize/fill serialization, rejected-event nonmutation, exact fractional
boundaries and a mutated report. The study is not registered as a return/PnL
ranking lab: ordering policies and outcomes must not be presented as comparable
strategy performance. Lock acquisition order is intentionally not deterministic;
the invariant is independent of which order wins, but allocations are not.

## Sources checked November 12

- [Paleologo, official Wiley chapter-one sample](https://catalogimages.wiley.com/images/db/pdf/9781394265459.excerpt.pdf), pp4 and16–18: liquidity and costs belong to the investment process. This motivates separating pre-trade controls and post-trade evidence. This prototype's invariant is our derivation, not attributed to the book. Only the public sample was read, not the full book.
- [Alpaca official order documentation](https://docs.alpaca.markets/us/docs/orders-at-alpaca), Buying Power and Bracket Orders: open orders consume buying power and rapid market events can outrun cancellation. This supports the gap and warns against a naive cancel-request release; this toy does not reproduce Alpaca rules.
- [QuantConnect official order events](https://www.quantconnect.com/docs/v2/writing-algorithms/trading-and-orders/order-events): partial fills are distinct order events. Consulted as prior art; no novelty claim for reservation accounting.

## Next bounded experiment and acceptance

Before any separately authorized production integration, specify a durable
event log with one account/currency owner, globally unique execution IDs,
external sequence provenance and replay checkpoints. Model missing/late/reordered
executions and crash between persistence and acknowledgment. Accept only if
recovery cannot double-debit, lose an acknowledged execution, or release cash
before an authoritative terminal watermark. Fail closed with an explicit
reconciliation state on conflicts. Keep sell proceeds, settlement, multi-currency,
margin and external consumers out until their contracts exist. Do not add a
broker adapter or treat this model as venue enforcement.

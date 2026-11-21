# Separate quotes, executable capacity and visible marks

Private research slice, November 20, 2026. This continues the
[sell/expiry diagnostic](BACKTRADER_SELL_QUOTE_PROBE.md) without changing the
portable fixtures or promoting the [capability screen](EXECUTION_CAPABILITY_PROFILE.md).
It is a declared synthetic contract, not a Backtrader adapter or market model.

## Why this slice

An OHLC slippage setting cannot establish that an order used a separate bid or
ask, respected its limit, and was valued at an independently observable mark.
The new replay makes those inputs and their availability explicit. Its value is
falsifiable accounting and timing behavior, not another strategy leaderboard.

Alternatives considered: keep tuning the previous OHLC proxy, or add a live
adapter. The former cannot identify these distinctions; the latter has unresolved
cash, capacity and lifecycle semantics. This narrow exact-rational replay and a
restricted existing-ledger comparison keep those unresolved questions visible.

## Run

The new replay and its demonstration use only Python's standard library:

```sh
python -m experiments.quote_visibility_contract
python -m experiments.quote_visibility_contract experiments/fixtures/quote_visibility_demo_v1.json
python -m pytest -q python/tests/test_quote_visibility_contract.py
```

The CLI accepts one local JSON file, bounded to 256,000 bytes, and rejects
duplicate fields. A successful exit means a replay was produced, not that an
engine was qualified. All outputs state `synthetic_only=true` and
`external_engine_validated=false`. Unknown fields, nonfinite/float monetary
inputs, ambiguous timestamps and unsupported encodings fail rather than being
coerced. Monetary inputs are at most 64 ASCII characters of integer, decimal or
fraction syntax, with explicit magnitude limits. Exponents are not accepted.
This bounded diagnostic is not a security sandbox for hostile inputs.

## Exact conventions

- One externally supplied order; one synthetic asset/currency; whole shares;
  long-only and no cash borrowing. This does not prove how a strategy chose its
  order or that the order itself was free of future information.
- Integer timestamps share one relative clock. At a timestamp, data arrivals
  precede the decision view. An order still cannot fill at its decision timestamp.
  Earliest-fill time and deadline are inclusive, subject to that strict rule.
- Each bucket is one **instantaneous declared capacity opportunity**. `at` is
  its observation/opportunity time. `available_at` cannot precede `at`. A bucket
  unavailable at its opportunity is missed permanently. Its capacity is neither
  backfilled when the quote arrives nor carried into a later bucket.
- This conservative consumer-visible opportunity rule is an explicit modeling
  choice. A resting real broker order could execute before a trader receives
  market data. The replay does not model that broker, latency, queue or exchange.
- A buy's price predicate is `ask <= limit`; a sell's is `bid >= limit`.
  `limit_eligible` reports that price predicate only, even outside the order's
  window or after completion. It is null for an invisible quote. `eligible`
  combines price, time, visibility and remaining order size, but **not capacity**.
  An eligible order with zero capacity produces no fill and no fee.
- A fill is the minimum of remaining shares and that bucket's capacity, at the
  declared ask for a buy or bid for a sell. Fees are fixed per nonzero fill plus
  a fraction of absolute notional. Cash/inventory shortfall refuses the whole
  replay with `ValueError`, not an affordable-size clip or simulated broker event.
- Mark records have independent observation and availability times. At each
  row, use the latest observed mark that has actually arrived. An older mark
  arriving late cannot replace a newer known observation. Duplicate observation
  timestamps are rejected: revisions need a separately specified contract.
- Mark price need not lie inside the spread. It is a valuation input, never a
  fill or cash flow. Without a known mark, held inventory has unknown equity;
  flat inventory still has equity equal to cash. Mark provenance times remain
  visible, rather than silently interpolating or borrowing a future value.
- `reference_close` is a diagnostic input only. It cannot determine a fill or
  valuation. Changing only it leaves the complete replay unchanged.
- A bucket strictly after the deadline reports an expired unfilled residual.
  This is the first observed post-deadline state, not an event asserted to have
  happened at the deadline. Inventory is not liquidated. A feed ending at or
  before the deadline can retain an active/partial residual. Completed stays
  completed.

Whole-document validation precedes execution. Prefix invariance applies to
successful, valid replays, not a promise to return a partial prefix after an
invalid future record or a later cash shortfall.

## Hand-checkable demonstration

Start with cash 100, no inventory, buy three with limit 10. The first bucket's
ask is 10 but capacity zero: price eligibility is not a fill. Buy two at the
next ask of 9, paying `1 + 18/1000`; cash becomes 80.982. Only mark 8 is visible,
so equity is 96.982. Buy the last share at ask 10 at the inclusive deadline,
paying `1 + 10/1000`; cash becomes 69.972. Mark 12 has now arrived, giving equity
105.972. The mark change is not cash profit or a claim of investable returns.

Holding the same reference close but changing the second ask from 9 to 11 blocks
that limit fill. Changing only the delayed mark changes equity only after its
availability, never fills or cash. A sell-limit control uses bid, not ask/mark;
it expires with inventory still held. These expected outcomes were independently
hand-calculated and reviewed, not copied from a successful implementation run.

## Existing-ledger integration boundary

`compare_accounting_subset(document)` exercises the existing local `fill_ledger`
on a directly representable subset. It refuses limit orders, delayed quotes,
missing time-zero marks and buckets at the initial-state timestamp. It does not
convert a blocked quote to zero volume or invent a mark to manufacture agreement.
Visible marks are mapped explicitly to the ledger's separate mark field.

```python
import json
from experiments.quote_visibility_contract import DEMO, compare_accounting_subset

document = json.loads(DEMO.read_text())
document["order"]["limit"] = None
document["order"]["shares"] = 5
result = compare_accounting_subset(document)
assert result["status"] == "matched_accounting_subset"
assert result["full_contract_compatible"] is False
assert result["external_engine_validated"] is False
```

The comparison checks ordered UTC row times, cash, inventory, equity, commissions
and exact fills. It does not truncate fractional quantities or subsecond times.
It discloses both lifecycle statuses separately. The existing ledger reports
`cancelled_at_horizon` or `cancelled_at_deadline` in its final order outcome, not
as a timestamped cancellation event; this new replay can remain active exactly at
the deadline or at an earlier feed end. Therefore matching financial paths are
**not full behavioral compatibility**, and cannot promote the portable screen.
Source hashes identify the compared implementation bytes, not their authenticity.

## Evidence and remaining work

Tests include hand calculations, delayed arrivals, future-record perturbations,
close/mark invariance, invalid contracts and changed-observation mutants. An
independent exact oracle covers 200 fixed-seed scenarios, including shortfall
refusals. A second corpus compares the directly representable ledger subset and
preserves its actual lifecycle differences. Results, failures and exact source
identities are retained in the private checkpoint.

Next: declare a receipt-conversion policy that preserves these terminal causes,
event times and active residuals; separately specify multi-order shared capacity
and forced liquidation. No queue, impact, live order, external quote adapter,
production API or statistical trading-performance claim follows from this slice.
Full native/package, notebook and hosted integration checks remain separate gates.

## Primary-source context

[Backtrader's official execution documentation](https://www.backtrader.com/docu/order-creation-execution/order-creation-execution/)
describes its next-bar OHLC execution assumptions and distinct order types.
Those assumptions motivated an explicit contract rather than another slippage
proxy; this new replay is not a description of all Backtrader configurations.
[Wiley's public Paleologo chapter](https://catalogimages.wiley.com/images/db/pdf/9781394265459.excerpt.pdf)
motivates understanding trading mechanisms before choosing quantitative tools.
Neither source supplies this implementation, validates these modeling choices,
or establishes engine parity. No full-book access, copied exercises, theorem
constants or paid data are assumed. Boyd's Stanford source was unavailable during
this run (HTTP 502); no new claim is attributed to a rereading of it.

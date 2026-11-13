# Offline execution accounting: implementation-ready follow-up

Status: first offline slice implemented on 2026-11-10 in `fill_ledger.py`, with
the bounded semantics in [FILL_LEDGER_LAB.md](FILL_LEDGER_LAB.md). Same-timestamp
fills, cash/inventory clipping, live trading and market calibration remain out of
scope. Original proposal date: 2026-11-09.
The [cash-only segment handoff](EXECUTION_SEGMENTS_LAB.md) is also implemented;
arbitrary CPCV path execution remains a proposal requiring a separate adapter.
Reviewed baseline: `ac8c6097104d514dddc86e47eeaf8e5df05d547b`.

## Why this is a product gap

`python/openquant/research.py::run_flywheel_iteration` explicitly describes its
cost estimate as a heuristic: position changes are charged in return units, the
first bar is free, and estimated aggregate costs are subtracted from a compounded
gross return. `pipeline.py` supplies signals and marked backtest frames;
`backtesting_engine.py::run_cpcv` consumes already-computed split returns. None is
an order/fill ledger or a guarantee that an allocation is executable. Preserve
those APIs and their historical outputs while developing the separate interface.

The existing `self_financing_lab` solves a fully invested proportional-fee model
but excludes terminal liquidation. `signal_turnover_lab` uses stationary additive
PnL and excludes entry and liquidation. `capacity_scaling_lab` evaluates a fixed
schedule and assumed volume without generating fills. The new timing and
liquidation labs are narrow controls for these boundaries, not a common model:
one uses share-price increments and an acquisition target, the other long-only
cash-funded fractional wealth. Do not combine their coefficients or outputs.

## Proposed first slice: one instrument, one currency, offline only

New experimental module `experiments/fill_ledger.py`, with immutable validated
records, has no broker, network, order-submission or optimizer integration.
Inputs are explicit rather than inferred from a price series:

- Decision: unique ID, information-available timestamp, signed requested shares,
  earliest allowed fill timestamp, deadline, and cancellation policy.
- Market bucket: timestamp, mark price in currency/share, executable buy and sell
  prices, available share volume, and source identifier. Missing/nonpositive
  prices or unknown volume are errors, not zero-cost or unlimited liquidity.
- Fill: decision ID, timestamp, signed executed shares, execution price, commission
  in currency, and a reason for any unfilled remainder. One instrument and currency
  are named at the run level; reject inconsistent identifiers.
- Initial state: cash currency amount, signed initial inventory, initial mark.
  First slice rejects shorts, borrowing and negative initial cash; later support
  requires explicit financing, collateral and insolvency rules.
- Terminal policy: `mark_only` or `liquidate_at_declared_bid`. The latter requires
  an executable terminal quote and volume, and reports incomplete liquidation.
  No silent sale at a mid price or erasure of residual inventory.

## Timeline and accounting contract

All timestamps use a single timezone-aware UTC convention. Require
information_available_at <= decision_at <= earliest_fill_at <= fill_at <= deadline.
A same-timestamp decision/fill is allowed only when an explicit event sequence
places information, decision and quote before fill; otherwise default to the next
bucket. A close-derived signal cannot earn that just-finished bar's return.

Within each bucket: prior inventory exists; eligible fills execute at their
declared price; fees debit cash; remaining inventory is valued at the closing
mark. Changing the event convention requires a new schema version.

For signed fill q and execution price p: inventory' = inventory + q,
cash' = cash - q*p - commission. Marked equity = cash + inventory*mark.
Reject orders that make cash or inventory negative under the first-slice policy.
For each interval, reconcile equity change to prior inventory mark movement,
fill mark-to-execution differences and commissions. Entry charges and terminal
charges are always separate fields; do not amortize them into missing bars.

Unfilled quantity is a state, not a zero-price fill. The initial implementation
uses FIFO by decision timestamp then unique ID, fixed participation limit, and
cancel-at-deadline. No same-bucket volume is reused across orders. A zero volume
bucket produces zero fills and a recorded residual; unknown volume is rejected.
No interpolation of future quotes. Record fill rate and completion alongside PnL.

## Outputs and comparisons

Return immutable decision, fill, cash/inventory and terminal ledgers, plus
configuration, source hashes and precise units. Keep marked equity, realized
terminal cash, residual inventory and total implementation cost distinct. Report
both unsuccessful and complete paths. Compare policies on identical decisions,
available quotes, horizons, initial inventory and terminal policy. Rank neither
policies with unequal completion nor mark-only and cash-liquidated outcomes as
if their economic obligations were identical.

Do not feed a stitched CPCV path through one stateful execution ledger: adjacent
segments can originate from independently fitted splits with incompatible
inventory. First define a reset/transition contract and charge its transitions,
or execute each coherent chronological split separately. Existing CPCV statistics
accepting returns do not validate those accounting decisions.

## Acceptance fixtures before implementation

1. Start with cash 100, buy 2 shares at 10 with fee 1, then mark 11. Cash 79,
   inventory 2, marked equity 101. Sell both at bid 10.5 with fee 1: cash 99,
   inventory 0. No-buy cash remains 100. This is an accounting example only.
2. A signal available at t=1 cannot fill in t=0 or earn the t=0 to t=1 return.
   Mutating future quotes leaves the earlier fill prefix byte-identical.
3. Two orders requesting 6 shares each with bucket volume 10 and participation
   0.5 fill a total of 5, never 10; FIFO's first order retains 1 unfilled and the
   second retains 6. Record both residuals through the deadline.
4. A terminal sale with only one share of sellable capacity liquidates one of
   two shares and preserves the other. `completed` must remain false.
5. Different timestamp units, duplicate decision IDs, malformed fees, nonfinite
   prices, negative volume, overselling, unsupported shorts, and negative cash
   fail with input-naming errors. Valid zero fills are retained.
6. A rational arithmetic oracle checks every ledger row; independent summation
   checks the equity reconciliation. Check both exact cents and binary64 scaling
   limits, without claiming that floating-point tolerances are market precision.
7. Saved reports retain all scenarios, refuse overwrites, and reject changed
   source/config/scientific results under the existing exact replay protocol.

## Implementation order and explicit non-goals

Write fixtures 1–5 first, then the smallest pure ledger transition, then the FIFO
capacity allocator, then terminal actions. Only after independent accounting
review add a runner/replayer entry and an example adapter from a fixed synthetic
decision tape. Keep `research.py`, `pipeline.py`, and Rust default behavior unchanged.
Run source-specific tests, fresh-process replay and supported full Python gates.

Market calibration, order-book queues, stochastic fills, adverse selection,
temporary/permanent impact dynamics, tax lots, funding, shorts, portfolio margin,
multi-currency conversion, corporate actions, live orders and profitable strategy
selection are excluded. Each needs its own contract and evidence. The repository
production-readiness brief also excludes live trading/execution; this proposal is
offline research only, not authorization to broaden that scope.

## Sources and attribution boundary

- [Paleologo, The Elements of Quantitative Investing, publisher](https://www.wiley-vch.de/en/areas-interest/finance-economics-law/the-elements-of-quantitative-investing-978-1-394-26545-9):
  readable publisher description frames portfolio construction, transaction costs
  and post-trade analysis as connected research topics. This motivates the gap,
  not any equation or claimed replication here.
- [Paleologo's author-maintained links](https://linktr.ee/paleologo): verified book
  and lecture pointers. The linked lectures were not readable in this session.
- [Almgren and Chriss, Optimal execution of portfolio transactions](https://doi.org/10.21314/JOR.2001.041):
  readable publisher abstract distinguishes execution costs from volatility risk.
  These simple fixed-schedule controls are not that paper's optimal strategy.

Wiley's US landing page and sample PDF returned access errors. No full book,
paywalled chapter or unavailable lecture was copied, inferred to have been read,
or used as an undocumented implementation source. All numerical examples and
interface choices above are original proposed controls.

## Conditional cash-budget follow-up (November 11)

[CASH_BUDGET_LAB.md](CASH_BUDGET_LAB.md) implements an isolated pre-trade buy
budget and falsification lab. Price, fee and fill-count assumptions are explicit;
order-lot rounding is distinct from a per-fill lot rule. The original ledger still
rejects infeasible fills rather than silently clipping them. Its conditional cash
bound does not implement a reservation book, terminal liquidation or CPCV adapter.
The earlier access-error note above describes the prior research run. The official
publisher's chapter-one sample was readable on November 11; the new guide cites
only the sections actually inspected and still makes no full-book-access claim.

# Offline partial fills and cash accounting

This original synthetic study implements a bounded first slice of the
[execution contract](EXECUTION_CONTRACT_SPEC.md). It is separate from the public
OpenQuant APIs, a broker, an order book, a portfolio optimizer and live trading.
Nothing here estimates achievable returns or market capacity.

## Run and reproduce

From the repository root with the existing offline environment:

```bash
python -m experiments.run_offline_lab defaults --lab fill_ledger
python -m experiments.run_offline_lab run \
  --lab fill_ledger \
  --config experiments/configs/offline-smoke/fill_ledger.json \
  --out /tmp/fill-ledger-new-report.json
python -m experiments.replay_offline_report /tmp/fill-ledger-new-report.json \
  --lab fill_ledger
```

Use a fresh output path. The smoke configuration retains 12 scenarios; defaults
retain all 48 combinations of four participation limits, two fixed commissions,
two proportional fees and three terminal volumes. Each scenario retains its
input decisions, quotes, source identifiers, execution settings, nonzero fills,
eligible zero/partial-fill attempts, cash/inventory rows and order remainders.
Both completed and unsuccessful obligations remain visible. There is no policy
selection or ranking across different completion levels.

## What the fixtures demonstrate

- Buying two shares at 10, paying 1, and marking at 11 leaves cash 79 and marked
  equity 101 from initial cash 100. Selling both at a declared bid of 10.5 and
  paying another 1 leaves cash 99. Marked equity is not terminal cash.
- Two six-share decisions compete for a single capacity of five shares. FIFO
  fills the first five, leaving signed remainders one and six. Assigning five
  shares to each order would reuse the same market volume.
- If two shares are held but terminal capacity is one, one share remains. With
  initial cash 100, a two-share purchase at 10, no commission and terminal bid
  10.5/mark 11, cash is 90.5, equity is 101.5 and liquidation is incomplete.
- Zero capacity creates a recorded attempt without a fill or fee. Cash remains
  100 in the corresponding control, even when fixed fees are positive.

The negative controls deliberately implement each incorrect interpretation as
an explicit competing answer. Correct arithmetic on invented inputs is only an
accounting test, not evidence about trading in an actual market.

## Exact numeric and event contract

The reusable module is `experiments.fill_ledger`. Input records are frozen
dataclasses: `InitialState`, `Decision`, `MarketBucket`. `run_ledger` returns a
frozen `LedgerResult`, whose collection fields are tuples. Amount fields normalize
Python integers, finite floats, rational/decimal strings and `Fraction` to exact
fractions. Typed callers can pass `Fraction` explicitly. A float means its Python
decimal representation, so `0.1` becomes 1/10. This intentionally differs from the
older timing lab's exact-binary64-input convention. Do not pool their coefficients.

JSON reports encode quantities and money as rational strings such as `181/2`.
This avoids silent binary64 underflow, overflow or cent rounding. It does not
provide currency tick/lot rounding, materiality or market precision. All inputs
are trusted local data; large rationals or grids can require large resources.

All datetimes must have a zero UTC offset and be timezone aware. Decisions satisfy
information_at <= decision_at <= earliest_fill_at <= deadline. Quotes must be
strictly increasing after the initial state. A fill quote is strictly later than
decision_at, at/after earliest_fill_at and at/before deadline. Same-timestamp
execution is deliberately unsupported because no intra-timestamp ordering was
declared. Quote mutation after a prefix must not change earlier fills or balances.

Per regular bucket, eligible orders are FIFO by decision_at then ID. Capacity is
participation times declared volume, shared across all orders as gross shares.
Opposing buys/sells get no capacity credit from netting. Execution uses ask for
buys and bid for sells. A later closing mark may lie outside that spread.
Nonzero fills pay fixed_fee + abs(shares) * execution_price * fee_rate.
Candidate fills that make cash or inventory negative reject the run. The ledger
does not silently reduce the order to available cash or stock.

For each row, exact equity change equals prior inventory times mark movement,
plus signed fill shares times (closing mark minus execution price), minus fees.
The independently reviewed transition uses cash -= shares*price + fee and
inventory += shares. Cash and stock are never inferred from equity alone.

Order remainders are cancelled at their deadline or the regular horizon, including
orders that did not become eligible before that horizon. A liquidation quote must
be separate and strictly after the regular horizon, with its own volume. It applies
the same participation/fees and sells only available capacity. `terminal.completed`
means completion of the selected terminal policy: for `mark_only` it is true even
with inventory; for liquidation it requires zero remaining inventory. It is not a
substitute for `decisions_completed`. The lab reports their conjunction separately.

## Evidence and boundaries

Tests pin hand amounts, chronology, cash/short rejection, malformed amounts,
duplicate IDs, inconsistent units, fees, zero capacity and scales beyond binary64
result range. Independent review additionally recomputes balances from inception
rather than using the implementation's transition helper. Retained research
evidence includes that audit, reports, source hashes and test logs.
Replay checks source/dependency identity and scientific output, and tests mutate
config, balances and both source hashes. These checks establish consistency, not
authenticity or economic validity.

The ledger has no financing, shorts, adverse selection, stochastic fill probability,
queue priority, permanent/temporary price impact, taxes, corporate actions or
multiple instruments/currencies. It is not safe to use it on stitched CPCV
segments without a separately specified state/reset policy and transition charges.

## Sources and attribution

[Paleologo's publisher description](https://www.wiley-vch.de/en/areas-interest/finance-economics-law/the-elements-of-quantitative-investing-978-1-394-26545-9)
motivates connecting portfolio decisions, costs and post-trade analysis. It does
not supply these equations. The publisher-linked sample returned 403 on 2026-11-10;
no full-book access, quotation or formula replication is claimed.

The [author-linked September 2024 BAML talk](https://docs.google.com/presentation/d/1AoS83uXEqf9XRraO4ZwvNKPKC9b0JICnWn5MCR83H-0/htmlpresent)
was readable through the presentation's native HTML view. Slides 6–9 warn about
reusing holdouts, future information and searching large strategy families.
That supports retaining all prespecified scenarios and separating accounting
checks from strategy evidence. Mathematical expressions missing from the HTML
extraction were not reconstructed, implemented or claimed to have been verified.

The [Almgren–Chriss publisher abstract](https://doi.org/10.21314/JOR.2001.041)
provides execution cost/risk context. This ledger is not their optimal-execution
model. All numeric fixtures and the interface design here are original.

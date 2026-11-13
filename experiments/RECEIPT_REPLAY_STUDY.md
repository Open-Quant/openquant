# Receipt recovery and delayed execution evidence

Offline research continuation, November 13, 2026. No production API or broker
integration. This extends the [shared-cash model](CASH_RESERVATION_STUDY.md).

## Question and choice

Can a small local journal preserve the cash-reservation invariant across process
exit, duplicate deliveries, and an out-of-order terminal receipt? An in-memory
snapshot alone cannot distinguish an execution already applied before a crash
from one not yet applied. A cancellation notice alone cannot prove that fills
are complete. We choose a SQLite command journal, replayed deterministically,
over a broker adapter or a hand-written filesystem append protocol. This is an
offline model of the contract, not a new database or a claim of novel accounting.

These implementation choices are research decisions, not approved production
design.

## Contract

One account/currency and one trusted source stream are modeled. Source sequences
start at 1 and are gap-free. A terminal event asserts that every execution for
that order is in the prefix through that sequence. It is **not** a cancel-request
or cancel-ack event. Those guarantees must come from an external adapter before
this could represent anything real. We do not claim that an ordinary broker API
supplies this contract.

Every local command is committed before acknowledgment. Reserve commands are
idempotent by a caller-supplied ID and exact payload. Source events are idempotent
only if ID, sequence and complete payload agree. A missing source sequence holds
all available funds at zero, even though some accounting state remains known.
Contiguous arrivals are processed in source order, so a terminal event cannot
release a reservation while an earlier receipt is missing.

Conflicting identities/sequences and execution-envelope violations are retained
in the journal and put the account into permanent quarantine. Further source
evidence can be retained, but reservations are blocked. The displayed cash is
the **last valid prefix**, not a claim about the actual account after a breach.
There is deliberately no 'clear quarantine' button, automatic retry order, or
reconciliation guess. The next design must supply authoritative reconciliation.

SQLite BEGIN IMMEDIATE serializes independent connections before replay and
insertion; there is no stale process cache used for sizing. Full replay for each
append is intentionally conservative and unsuitable for high-rate tick streams.
Snapshot availability is advisory, never authority for another cash consumer.

## Validation and interpretation

Run from the repository root with Python and NumPy:

```sh
python -m unittest discover -s python/tests -p test_receipt_journal.py
python -m experiments.receipt_replay_study > /tmp/receipt-study.json
```

Tests terminate a real child process immediately before and after SQLite commit,
reopen the file, and retry the command. This exercises lost acknowledgment and
rollback recovery, **not power-loss/filesystem corruption**. Concurrent-connection
tests check the exercised schedules, not fairness or all thread interleavings.
All six arrival orders of two fills plus terminal receipt produce the independently
calculated final cash. The study varies exact prices with explicit fixed fees, delivery permutations,
duplicates and restart boundaries; it separately times replay growth.

No price forecast, strategy return, Sharpe ratio or investment-performance claim
is made. Timing is local software evidence, not exchange latency. The journal has
no authentication or tamper-proof seal. A malicious editor can rewrite history;
sequence-gap detection does not prevent tail truncation or coordinated forgery.
OS/storage flush semantics, disk-full behavior and database backup/recovery need
separate qualification. Files should stay local, not on a network filesystem.

## Primary sources and roadmap

- [SQLite atomic commit](https://www.sqlite.org/atomiccommit.html): transaction
  durability depends on filesystem and storage assumptions. We reuse SQLite, not
  reimplement its journal. Local process exits do not independently verify all
  power-loss claims.
- [Alpaca orders](https://docs.alpaca.markets/us/docs/orders-at-alpaca): open orders
  consume buying power and cancellation is not the same as an execution record.
  The experiment is not an implementation of Alpaca margin or settlement rules.
- [Paleologo, Wiley chapter-one sample](https://catalogimages.wiley.com/images/db/pdf/9781394265459.excerpt.pdf):
  practical investing involves market mechanisms and costs as well as mathematical
  models. This motivates explicit execution contracts; the state machine is our
  design, not a method attributed to the book. Only the public sample was read.

Go/no-go: no broker integration until a real source's sequence/completeness and
reconciliation rules are specified, schema migration/retention is designed,
cash consumers share one authoritative owner, and replay throughput meets a
declared workload. Checkpointing could reduce replay cost but must bind schema,
prefix and replay code version, and survive corruption tests before acceptance.
Live trading, selling, margin, settlement, FX and multi-host ownership remain out
of scope. Reconcile the cumulative research branch against current main and
separate code/docs work before any integration.

## November 13 local result

The 31 receipt/reservation unit tests passed. Independent review found that
recreating missing metadata over surviving commands could silently change the
initial balance; a failing regression preceded rejecting incomplete schema or
metadata. Exact duplicate deliveries now leave journal length unchanged.

One deterministic synthetic run (seed 20261113) completed 10/30/60 orders with
30/90/180 shuffled source receipts, doubled delivery and periodic restarts.
Independent cash arithmetic matched in every case. Local elapsed times were
0.108/1.432/12.320 seconds respectively. This is a **negative scalability result**,
not a benchmark: the replay-every-command design is unsuitable for live streams.
No real tick dataset was downloaded or evaluated in this continuation. Receipt
recovery needs authenticated source semantics before market data would validate
anything beyond the toy arithmetic. Next experiment: a versioned checkpoint
whose replay matches this slow oracle, tested against missing/corrupt checkpoints
and lost acknowledgments, before considering a real adapter.

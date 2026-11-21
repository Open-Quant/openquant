# Bounded receipt batches before persistent checkpoints

Offline product-research continuation, November 14, 2026. This is a synthetic,
single-account experiment, not a broker adapter, trading service or package API.

## Recovery and product choice

The November 13 private continuation proposed a version-bound checkpoint after
full replay took 12.32 seconds for one 60-order synthetic run. Its receipt journal,
study and tests are now present on main through PR #287. This work starts from
`5c2750fc2dc3b54f5a2b3bcd56de5ed4033f1529`, retaining the merged typing/formatting
updates rather than applying a stale cumulative patch.

Three approaches were considered:

1. **Keep individual appends.** Smallest audit surface, but repeated replay and
   commits are unnecessarily expensive for already-available offline receipts.
2. **Batch a bounded set of receipts in one transaction.** Chosen here. It reuses
   the existing replay/state machine, does not trust serialized derived state,
   and allows exact journal and boundary-state comparisons.
3. **Persistent state checkpoints.** Deferred until the full acceptance contract
   below is implemented. A deserialized cash number alone would omit pending
   events, deduplication history and quarantine, potentially releasing funds.

These are research implementation choices made for this experiment, not approval
for a production architecture or a change in the project's live-trading scope.
No broker implementation is justified by this result.

## API and acknowledgment boundary

`ReceiptJournal.receive_batch(receipts)` accepts an iterable of mappings with
the same arguments as `receive`: `event_id`, `sequence`, `kind`, `order_id`, and
optional `fill`.

- At least one and at most 256 receipts. Iteration stops upon consuming item 257.
- Aggregate canonical JSON is at most 1 MiB in encoded UTF-8 bytes. This bounds
  retained encoded batch payload, not all CPU/memory used to construct arbitrary
  caller objects. This remains a trusted, local research interface.
- Each receipt is syntactically validated and frozen immediately, before asking
  an iterator for its next item. Reusing a mutable dictionary cannot rewrite an
  earlier yielded receipt.
- The entire batch is frozen before a database transaction starts. A syntax
  error, iterator exception, or size error writes **none** of the batch.
- Fully syntactically valid arrivals are applied in input order, not sorted.
  Contiguous source sequences are processed by the existing state machine.
- Exact duplicate payloads do not add durable commands, including duplicates
  within a batch and whole-batch retries after restart.
- Semantic conflicts or envelope breaches are durable evidence, not syntax
  rejection: they quarantine the account, block new reservations and retain
  later source evidence in the same batch.
- One `BEGIN IMMEDIATE` encloses replay, duplicate checks and inserts. One commit
  precedes the returned **final** snapshot. There are no intermediate success
  acknowledgments.

A rejected or interrupted batch is **unacknowledged**. The caller must retain
the original receipts until it obtains a successful result, then retry the same
batch when commit outcome is uncertain. This does not justify retrying an order.
If a malformed item is discovered after a real receipt, the preceding receipt is
not stored by this rejected batch. The caller must preserve it for explicit
repair/reconciliation, not drop it or claim the source was acknowledged.

Consequently, batch/single-receipt equivalence is claimed only for fully
syntactically valid inputs. Sequential calls could persist a prefix before a
later syntax error; batch prevalidation deliberately does not.

## Runnable example

From the repository root, with NumPy installed:

```python
from datetime import UTC, datetime, timedelta
from fractions import Fraction
from pathlib import Path
from tempfile import TemporaryDirectory

from experiments.fill_ledger import Fill
from experiments.receipt_journal import ReceiptJournal

t = datetime(2026, 1, 1, tzinfo=UTC)
with TemporaryDirectory() as folder:
    journal = ReceiptJournal(Path(folder) / "receipts.sqlite", cash=100, cash_floor=5)
    try:
        journal.reserve(
            "reserve-a", "a", desired_shares=10, price_cap=10, lot_size=1,
            fixed_fee=1, fee_rate=0, max_fills=2, information_at=t, decision_at=t,
        )
        state = journal.receive_batch([
            {"event_id": "done", "sequence": 2, "kind": "terminal", "order_id": "a"},
            {"event_id": "fill", "sequence": 1, "kind": "fill", "order_id": "a",
             "fill": Fill("a", t + timedelta(seconds=1), Fraction(2),
                          Fraction(9), Fraction(1), False)},
        ])
        assert (state.cash, state.reserved, state.available) == (81, 0, 76)
    finally:
        journal.close()
```

The terminal event is a **trusted completeness assertion**, not an ordinary
cancel acknowledgment. The example's receipt contract does not establish that
any real broker supplies this property.

## Validation protocol

```sh
python -m unittest discover -s python/tests -p 'test_receipt*.py'
python -m experiments.receipt_batch_study --orders 10 30 60 \
  --batch-sizes 1 16 64 256 --repeats 3 > /tmp/receipt-batch-study.json
python -m experiments.receipt_batch_study --orders 30 60 \
  --batch-sizes 1 16 64 --restart-every 64 > /tmp/receipt-batch-restarts.json
```

For each paired workload, a seeded synthetic receipt sequence is delivered twice
and shuffled. The single-receipt path is the semantic oracle for ordered command payloads
and state at every batch boundary. It shares the accounting state machine, so it
is **not an independent accounting implementation**. A separate integer-cent
calculation checks final cash, using price cents, quantities and per-fill fees
without calling the budget, audit or journal calculation.

The benchmark excludes oracle time from candidate timings. Ingestion includes
the candidate's required restarts; total also includes initialization, sequential
reservations, final snapshot and close. Restart boundaries are measured in
**delivered receipts**, not batch count, so large batches do not quietly receive
fewer restart penalties. Batches split at each restart boundary.

Tests cover all delivery permutations and partitions of three receipts; gaps;
duplicate identities; permanent quarantine; mutable iterators; iterator failure;
count/byte limits; SQL failure rollback; independent connection retries; and real
child exits mid-insert, before commit and after commit. These tests do not
simulate power loss, faulty disk flushes or all possible process schedules.

Full replay still grows with history. Reserving each order separately remains
expensive; command history and deduplication state are not bounded. Batching
trades acknowledgment latency and lock duration for fewer replays and commits.
There is no live-stream latency target or throughput acceptance claim here.

## Checkpoint continuation contract (not implemented)

A future checkpoint prototype should remain derived, disposable state:

1. Bind format version, replay-code version, initial account configuration,
   durable command count and an exact ordered-prefix digest. Define domain
   separation and canonical encodings rather than hashing ambiguous string joins.
2. Preserve exact rational cash/floor, every active and completed order, receipt
   identities and payloads, pending source sequences, processed watermark,
   reservation IDs/payloads, and the first quarantine reason. Do not infer missing
   deduplication history from only the last source sequence.
3. Write the checkpoint and its prefix identity atomically. Never delete command
   history merely because a checkpoint exists; retain replay as the oracle.
4. On absent, stale, unsupported, malformed, digest-mismatched or incomplete
   checkpoints, ignore the checkpoint and replay the intact journal. If the
   journal itself is corrupt, fail closed rather than treating a cache as truth.
5. No pickle or executable payload deserialization. Validate exact types, bounds,
   identity uniqueness and cash invariants before accepting a structured format.
6. Prove equivalence for every retained adversarial case and randomized schedule,
   including corrupt/missing checkpoint, lost acknowledgment, mixed-version
   readers, independent writers, committed tail arrival, truncated history and
   quarantine with pending gaps.
7. Measure cold reopen separately from warm ingestion and full end-to-end time.
   Prefix verification can still be O(history). Report that cost rather than
   hiding it in setup or labelling the path constant time.
8. A checksum detects accidental disagreement; it is not authentication.
   Coordinated history/checkpoint forgery and tail truncation require an external
   trust/sequence anchor. SQLite durability depends on the filesystem/storage
   assumptions. Process-exit tests are not power-loss qualification.

Before a real adapter: specify authentic source ordering/completeness,
reconciliation authority, retention, migration and one owner for every cash
consumer. Selling, settlement, margin, FX, multi-host ownership and trading remain
out of scope.

## Sources and a bounded research follow-through

- [SQLite isolation](https://sqlite.org/isolation.html) explains serialized
  writers and visibility of committed transactions. This experiment uses the
  default rollback-journal family, not WAL or SQLite's WAL checkpoint mechanism.
- [SQLite atomic commit](https://sqlite.org/atomiccommit.html) explains its storage
  assumptions. Application-level checkpointing and SQLite journal checkpointing
  are different concepts.
- [Paleologo: publisher chapter-one sample](https://catalogimages.wiley.com/images/db/pdf/9781394265459.excerpt.pdf).
  Only the legitimate public excerpt was read, not the full book. Its
  before/during/after-trade framing motivates explicit execution and post-trade
  contracts. The batching/checkpoint design here is ours, not attributed to it.
- [Yin et al., 2026 preprint](https://arxiv.org/html/2603.20319v1) compares engines
  under aligned strategies and costs, reporting cost-sensitive disagreement.
  Its abstract-page and HTML naming/sign convention for CSI differ; do not
  import that metric without checking its definition. We have not reproduced
  its dataset, engine configurations or investment results.

A second bounded artifact is now implemented: the versioned
[cost-convention fixture contract](EXECUTION_COST_CONFORMANCE.md) with five original
fixtures and one OpenQuant experimental adapter, not another strategy leaderboard.
The next step is a genuinely independent adapter. Specify
fee units, per-fill versus per-order charges, price/slippage timing, terminal
liquidation, cash/position reconciliation and exact input event identity. The hand-checkable cases are preserved in a portable JSON fixture bank; two
independent implementations have not been compared. Zero-cost agreement alone cannot validate paid execution.
Keep that extension separate from this batching patch and do not claim one
backtester validates another merely because their final Sharpe ratios agree.

### Interrupted campaigns

Add `--jsonl` to flush each completed benchmark case. A `start` record binds
the five source-file hashes and environment; only a final `complete` record
certifies that all requested cases ran. Partial output remains useful evidence,
not a successful campaign. Source changes during a campaign abort it.

Equivalence assumes the same valid receipt arrival order and no interleaved
reservation commands. A batch intentionally removes transaction interleaving
points. The digest covers serialized ordered `(seq, payload)` rows, not SQLite
file bytes. The cost-convention fixture bank is now implemented in
[Execution cost conformance](EXECUTION_COST_CONFORMANCE.md); a second independent
engine adapter remains future work.

## Local measurement, November 14, 2026

Python 3.12.14, SQLite 3.53.1, Linux x86-64; three seeds, 36 paired cases.
Median ingestion seconds (reservations/setup and independent oracle excluded):

| Orders | Single | Batch 16 | Batch 64 | Batch 256 |
|---:|---:|---:|---:|---:|
| 10 | 0.183 | 0.015 | 0.005 | 0.006 |
| 30 | 2.011 | 0.157 | 0.043 | 0.032 |
| 60 | 13.805 | 0.675 | 0.202 | 0.107 |

All cases matched the sequential accounting path, independent cash calculation,
and ordered command payload digest. A separate six-case campaign restarted at
identical 32-delivery boundaries, with the same equivalence checks passing.
These are illustrative local measurements under concurrent testing/build load,
not stable latency guarantees or a production capacity claim. Full history is
still replayed each transaction; batching amortizes it rather than removing it.
The largest batch trades acknowledgment delay and writer lock duration for
throughput. The default 16 is a study parameter, not a recommended live policy.

The first benchmark attempt exhausted shared temporary disk and produced no
usable result; the successful rerun used a roomy workspace temporary directory.
This prompted streamed per-case receipts. The 36-case run predates the streamed
CLI/source-identity addition; the six-case restart run uses the final source-bound
format. Raw results and exact continuation details are in the private bundle.

# Did the whole synthetic receipt request arrive?

The [fill comparator](RECEIPT_ORDER_CONFORMANCE.md) checks supplied executions
through a caller-declared sequence. Identical two-fill tapes can match even if
both dropped an unannounced third fill. This experiment adds an independent
manifest and a separate request-scoped completion record. It leaves the original
comparator intact and never sends events to the [receipt journal](RECEIPT_REPLAY_STUDY.md).

## Try the acceptance controls

Run this from the checkout root with Python 3.11 or newer, no extra dependencies:

```python
from fractions import Fraction as Q

from experiments.receipt_completion_witness import (
    Completion,
    RequestManifest,
    ScopedReceipt,
    assess,
)
from experiments.receipt_order_conformance import Execution, Receipt

manifest = RequestManifest("A", 3)
fills = [Execution("buy", i, i, Q(1), Q(10), Q(1)) for i in (1, 2, 3)]
rows = [ScopedReceipt("A", Receipt(i, f"r{i}", fill)) for i, fill in enumerate(fills, 1)]

assert assess(manifest, fills[:2], rows[:2])["status"] == "incomplete_missing_completion"
short = assess(manifest, fills, rows[:2], completion=Completion("A", 3))
assert short["status"] == "count_mismatch" and short["count_differences"] == ["observed"]
wrong = assess(manifest, fills, rows, completion=Completion("B", 3))
assert wrong["status"] == "request_scope_mismatch"
full = assess(manifest, fills, [*rows, rows[0]], completion=Completion("A", 3))
assert full["status"] == "matched_completed_synthetic_request" and full["redeliveries"] == 1
assert not full["terminal_authority"]

empty = RequestManifest("zero", 0)
assert assess(empty, [], [])["status"] == "incomplete_missing_completion"
assert assess(empty, [], [], completion=Completion("zero", 0))["status"] == (
    "matched_completed_synthetic_request"
)
```

The manifest must be selected upstream, not derived from the received tapes or
the marker. Each observed delivery and the marker must carry the exact request
ID. Counts reconcile the manifest, marker, reference fills and distinct observed
fill receipts. Redeliveries count once only when the complete receipt matches
exactly. A reused identity, sequence or execution key with inconsistent
evidence is rejected. Requests use contiguous fill-only sequences 1 through
the manifest count; an out-of-range receipt is not silently ignored.

No completion means no nested comparison or equality claim, even when row counts
agree. With matching scope/count/sequence and a marker, the existing exact fill
comparator still has to pass. Matching account cash or equal counts cannot
excuse a changed execution. Reference fills are selected by the caller and are
keyed by order and fill ordinal, not source sequence; their provenance is not
verified here. Work is bounded to 256 reference fills and 1,024 raw delivery
attempts, plus one lookahead per input. That bounds records read, not payload
bytes or precision. The function consumes a bounded supplied batch, not a
live state machine; it does not authenticate a source, distinguish reuse of a
request ID across runs, or establish actual order terminality. The marker
can announce completion of the request only. No funds can be released.

## Source context

- The [existing receipt benchmark producer](receipt_batch_study.py) emits a
  distinct final record with a count only when its cases finish. The consumer
  gate above is not wired into that campaign format.
- [Turquoise Post Trade Gateway, §2.1.3](https://docs.londonstockexchange.com/sites/default/files/documents/tq202-issue-5_0_1-post-trading-gateway-fix-5_0.pdf)
  describes an accepted snapshot's announced report count, request IDs on
  reports, and a final-report flag. Original ordering is guaranteed only
  within each affected partition. Our contiguous single request, separate
  marker, and explicit zero completion are **synthetic choices**, not
  assertions about this or another venue.
- [Paleologo's public Wiley introduction](https://catalogimages.wiley.com/images/db/pdf/9781394265459.excerpt.pdf)
  separates pretrade analysis from posttrade attribution. That motivates
  verifying the execution inputs before trusting attribution; it does not
  supply any of the implementation or a trading-performance claim.

## Next experiment: detect shared same-count corruption

The [synthetic content-commitment experiment](RECEIPT_CONTENT_COMMITMENT.md)
implements the next offline slice. The independent public archive smoke below
remains a proposal. If both sides agree on the same three wrong
fills, all the count and economic checks here can agree. A useful next test is
a request-scoped content commitment created independently of both consumers.

1. A synthetic producer freezes one request generation, exact scope, schema
   version and an ordered typed fill tape before the consumer receives it.
   Produce a manifest with the distinct count and SHA-256 of a canonical
   stream. Include schema, generation ID, scope, sequence, source receipt ID
   and each exact fill field; use deterministic typed encoding and distinct
   length boundaries. Rationals use reduced numerator/denominator rather
   than floats. Define an ordered empty input explicitly. The comparator must
   not compute or refresh this **expected** commitment from its two input tapes.
2. After the existing scope/count/sequence gate and exact deduplication,
   compute the observed commitment in source sequence. A positive fixture
   includes reordering in delivery and exact retries; neither may change the
   result. Mutate the same fill on both tapes without changing count and
   retain the original commitment: even an economic match must fail
   commitment comparison. Also test schema-version change, old completion
   under a reused request ID but different generation, and explicit zero
   with the frozen empty commitment. Require an unqualified outcome if no
   independent commitment exists.
3. Verify the exact canonical byte fixture independently, then test byte
   input and interpreted rows as distinct boundaries. For a separately scoped
   public archive smoke, the [publisher's Binance public-data format and
   checksum guide](https://github.com/binance/binance-public-data#checksum)
   shows independent checksum files and archive revisions. Pin the archive
   key and checksum version; separately check parsed row count, timestamp
   unit/range, and decoded sequence/order using a second decoder or
   hand fixture. A publisher checksum can confirm the version read, not all
   events observed by the venue or retained by a decoder. Review the
   [dataset terms](https://github.com/binance/binance-public-data/blob/master/TERMS_AND_CONDITIONS.md)
   before obtaining or redistributing data. No archive was downloaded here.

None of these checks establish source authenticity, losslessness of an
external feed, brokerage parity or performance. No cancel/replace, corrections,
shared capacity, reservations, live execution or investment conclusions.

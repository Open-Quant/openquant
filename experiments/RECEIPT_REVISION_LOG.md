# Synthetic revision logs and as-of attribution

A correction can change what we know about a previous fill without changing
what was knowable at an earlier decision. The research module
`receipt_revision_log.py` keeps the original fill and every revision in an
immutable event tape. Cash, quantity and fees are exact `Fraction` values.
This is an isolated experiment, not a production or broker adapter.

## Run it

```python
from fractions import Fraction as Q

from experiments.receipt_completion_witness import RequestManifest
from experiments.receipt_content_commitment import RequestScope
from experiments.receipt_order_conformance import Execution
from experiments.receipt_revision_log import Bust, Correct, Fill, freeze, replay

scope = RequestScope(RequestManifest("demo", 3), "source-run-1")
fill = Fill(1, "e1", Execution("order-A", 1, 10, Q(2), Q(10), Q(1, 2)))
correction = Correct(2, "e2", "e1", Execution("order-A", 1, 10, Q(3, 2), Q(12), Q(1, 4)))
bust = Bust(3, "e3", "e2")

# The producer pins these exact bytes; the consumer checks a separate delivery.
pin = freeze(scope, (fill, correction, bust))
tape = replay(scope, (bust, fill, correction, correction), pin)
assert tape is not None
assert tape.redeliveries == 1
assert tape.as_of(1) == (Q(2), Q(-41, 2), Q(1, 2))
assert tape.as_of(2) == (Q(3, 2), Q(-73, 4), Q(1, 4))
assert tape.as_of(3) == (0, 0, 0)
assert tape.delta_at(2) == (Q(-1, 2), Q(9, 4), Q(-1, 4))
assert tape.delta_at(3) == (Q(-3, 2), Q(73, 4), Q(-1, 4))
assert tape.as_of(2, effective_through=9) == (0, 0, 0)

old_scope = RequestScope(RequestManifest("demo", 1), "source-run-1")
old_tape = replay(old_scope, (fill,), freeze(old_scope, (fill,)))
assert old_tape is not None
assert tape.extends(old_tape)
assert tape.as_of(1) == old_tape.as_of(1)
```

## Two coordinates, one immutable history

- `known_through` is a contiguous **source sequence**, not arrival time.
  No replay object is returned for an unpinned tape or mismatched digest.
  Missing sequences, count conflicts and invalid revisions reject rather than
  turning unverified operations into zero activity. Raw delivery may be
  out of order; only identical redelivery is idempotent.
- `effective_through` applies an independent, inclusive **execution time**
  cutoff to revisions known at the chosen sequence. A late-reported
  earlier fill is absent before its source sequence, even if its execution
  time satisfies the filter. Correction and bust apply at the original fill
  time in a reprojected view, but never leak into earlier known-by views.
- A correction replaces the current active revision by its exact ID, retains
  order, order-local ordinal, effective time and buy/sell side, and may
  change quantity, price and fee. A bust retires the active revision. A
  misreferenced stale/future/retired revision, branch and duplicate original
  key are errors. This toy deliberately excludes timestamp/side changes and
  reinstatement; they need separate semantics.
- `extends` checks continuity by comparing original typed events against a
  previously qualified tape in the same request/generation, not just digest
  validity or final totals. A different authoritative generation is a new
  lineage, not an extension. This does not link prefixes that are absent.

## Evidence format and bounds

The distinct v2 digest domain is
`b"openquant:synthetic-receipt-revisions\x00v2\x00"`. The unchanged
[fill-only v1 format](RECEIPT_CONTENT_COMMITMENT.md) is not interchangeable.
All fields use a four-byte big-endian byte length followed by the field bytes.
Scope fields are request ID, generation and expected distinct event count.
Event fields in source sequence are:

- Fill: `fill`, sequence, event ID, then v1's six execution fields.
- Correction: `correct`, sequence, event ID, active target ID, then the
  six fields of the replacement execution.
- Bust: `bust`, sequence, event ID and active target ID.

Bounds match v1: at most 256 distinct events and 1,024 raw deliveries,
one lookahead, UTF-8 fields at most 256 bytes, integer magnitudes at most
128 bits and fraction denominators at most 16 bits. Decode and transport
security are not modeled. Replay builds linear-size exact balances; an
unfiltered prefix read uses a cached balance and a filtered read scans at most
the pinned prefix. The counters and bounds do not establish performance for
production-sized or adversarial data.

The [FIX Latest Trade specification](https://www.fixtrading.org/wp-content/uploads/download-manager-files/FIX-Latest-Specification-Trade.pdf)
describes separate corrections/cancels and references to prior reports.
[Wiley's Paleologo excerpt](https://catalogimages.wiley.com/images/db/pdf/9781394265459.excerpt.pdf)
treats posttrade attribution separately from pretrade analysis. These motivate
the experiment; neither supplies its format. A pinned tape requires independent
pin selection and does not prove authenticity, actual visibility at decision
time, order terminality, deployability or an investment result.

## Observer-time extension (synthetic)

The source sequence answers which revisions can be applied **in order**. It
does not say when a particular consumer could have known them. The companion
`receipt_availability.py` uses a separately pinned, named observation clock.
The source tape and observation pin must be selected independently of the
consumer delivery. A checksum is not a clock or trusted timestamp.

```python
from fractions import Fraction as Q

from experiments.receipt_availability import Observation, freeze_observer, qualify
from experiments.receipt_completion_witness import RequestManifest
from experiments.receipt_content_commitment import RequestScope
from experiments.receipt_order_conformance import Execution
from experiments.receipt_revision_log import Correct, Fill, freeze, replay

fill = Fill(1, "e1", Execution("A", 1, 10, Q(2), Q(10), Q(1, 2)))
correction = Correct(2, "e2", "e1", Execution("A", 1, 10, Q(1), Q(9), Q(1, 4)))

def fixture_checkpoint(events, observed, through):
    scope = RequestScope(RequestManifest("demo", len(events)), "source-run")
    tape = replay(scope, events, freeze(scope, events))
    assert tape is not None
    pin = freeze_observer(tape, "observer-A", 15, through, observed)
    checkpoint = qualify(tape, reversed(observed), pin)
    assert checkpoint is not None
    return checkpoint

seen = (Observation("e1", 20), Observation("e2", 40))
early = fixture_checkpoint((fill,), seen[:1], 30)
later = fixture_checkpoint((fill, correction), seen, 45)

assert later.extends(early)
assert later.at(30, effective_through=10) == early.at(30) == (2, Q(-41, 2), Q(1, 2))
assert later.at(40, effective_through=10) == (1, Q(-37, 4), Q(1, 4))
assert later.at(40, effective_through=9) == (0, 0, 0)
```

- A pin covers an inclusive start and end and exactly one first observation
  for every event in the revision tape. These are integer ticks on one named
  fixture clock, not assumed to be UTC, publication, execution or exchange
  time. The observer asserts capture starts before any included observation.
  A different clock epoch requires a different identity.
- A query outside that interval or at a cut with any available event beyond
  a gap raises an error. Tied observations enter together. An empty or
  before-first-observation result means no activity **in the fixture prefix**,
  not proof of market inactivity. Independently valid later pins are not
  necessarily extensions: same revision and observation prefixes must be
  retained, and any newly appended observation must be strictly after the
  earlier coverage end.
- The observation digest uses domain
  `b"openquant:synthetic-receipt-availability\x00v1\x00"`, then four-byte
  big-endian byte-length-framed fields: source schema (UTF-8), raw revision
  SHA-256 (32 bytes), observer ID (UTF-8), start and end ticks (ASCII decimal),
  then each event ID and corresponding first-seen tick in source order.
  Bounds and exact-redelivery handling reuse the bounded tape conventions.
- This is historical reconstruction **conditional on** the pinned capture
  record. Proving that capture in practice requires independently retained,
  coverage-aware observer records, stable source identities and explicit
  clock reconciliation. A venue timestamp, archive publication date, or
  order of replay alone cannot establish consumer availability. No sampled
  replay here establishes authenticity, complete market visibility, order
  terminality or trading results.

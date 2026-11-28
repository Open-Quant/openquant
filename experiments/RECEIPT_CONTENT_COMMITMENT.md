# Can two matching receipt tapes both be wrong?

Yes. Cutoff agreement, a valid completion witness, and exact comparisons
between two tapes cannot detect the same altered fill in both. This synthetic
experiment adds one independently frozen content digest. It does not replace
the [fill comparator](RECEIPT_ORDER_CONFORMANCE.md) or the
[completion witness](RECEIPT_COMPLETION_WITNESS.md).

## Reproduce the common-mode failure

From the checkout root, using Python 3.11+ and the standard library:

```python
from dataclasses import replace
from fractions import Fraction as Q

from experiments.receipt_completion_witness import Completion, RequestManifest, ScopedReceipt
from experiments.receipt_content_commitment import (
    FrozenCommitment,
    GenerationCompletion,
    GenerationReceipt,
    RequestScope,
    assess_committed,
    digest,
)
from experiments.receipt_order_conformance import Execution, Receipt

scope = RequestScope(RequestManifest("snapshot", 1), "generation-7")
original = Execution("order-5", 1, 20, Q(2), Q(10), Q(1, 2))
source_row = ScopedReceipt("snapshot", Receipt(1, "fill-1", original))
# In an experiment, freeze this upstream before producing the two consumer tapes.
pinned = FrozenCommitment(scope, digest(scope, [source_row]))
end = GenerationCompletion("generation-7", Completion("snapshot", 1))

changed = replace(original, at=21)
bad_row = replace(source_row, receipt=replace(source_row.receipt, execution=changed))
bad = assess_committed(
    scope, [changed], [GenerationReceipt("generation-7", bad_row)],
    completion=end, commitment=pinned,
)
witness = bad["witness"]
assert witness is not None and witness["status"] == "matched_completed_synthetic_request"
assert bad["status"] == "content_mismatch" and bad["digest_matches"] is False

valid = assess_committed(
    scope, [original], [GenerationReceipt("generation-7", source_row)],
    completion=end, commitment=pinned,
)
assert valid["status"] == "matched_committed_synthetic_request"
assert not valid["terminal_authority"] and not valid["external_engine_validated"]
```

The caller selects a request ID, generation, expected distinct count and
pinned digest independently of the compared inputs. The same generation is
required on the completion and each delivery. An old marker or row from a
different generation cannot be reused under the same request ID. This
checks labels, not the freshness or authenticity of a producer.

No canonical content digest is computed for an unqualified completion witness.
An eligible observed digest can match when only the reference is wrong; that
remains `execution_mismatch`. If an eligible observed digest differs, report
`content_mismatch`, retaining the separate witness result. No digest means
`missing_commitment`, not a promotion. A genuinely empty request needs both
the zero-count completion and the frozen empty-tape digest. The two-tape
comparison cannot regenerate its own expected digest. The producer-side
`digest` helper cannot prove the caller supplied an independent commitment.

## Synthetic v1 canonical encoding

SHA-256 input starts with the exact bytes
`b"openquant:synthetic-receipt-content\x00v1\x00"`. Each following field is
a four-byte unsigned **big-endian byte length** then its bytes; there is no
separator, newline or implicit normalization. Field order:

1. Request ID, generation, distinct count.
2. For each exactly deduplicated receipt in contiguous source sequence:
   sequence, receipt ID, order ID, order-local ordinal, execution time,
   signed quantity, price, fee.

Text uses exact UTF-8; integers use canonical ASCII decimal; all fractions use
ASCII `numerator/denominator` in reduced form, even whole numbers. Signed fill
quantities are preserved. The order of raw delivery attempts is irrelevant;
reassigning source sequence or receipt identity is not. Unsupported versions,
unencodable text, conflicting redeliveries and invalid source sequences do
not produce a qualified report. The version is part of the digest domain.

The experiment permits at most 256 reference fills, 1,024 raw deliveries and
one lookahead per iterable; text fields are at most 256 UTF-8 bytes, signed
integer magnitudes at most 128 bits and fraction denominators at most 16 bits.
Both input tapes are checked before exact aggregation. These are toy resource
limits and not a sandbox for arbitrary iterators or a claim about all Python
runtime precision policies. A reviewed literal Unicode byte vector and a
separately empty vector pin the v1 serialization in tests.

## Follow-up: corrections without rewriting the tape

The [synthetic revision-log experiment](RECEIPT_REVISION_LOG.md) uses a separate
v2 digest and supports append-only fills, corrections and busts. It keeps
source-known and execution-effective cutoffs distinct; the v1 fill format and
comparator here are unchanged.

## Sources and boundaries

[NIST FIPS 180-4](https://csrc.nist.gov/pubs/fips/180-4/upd1/final) specifies
SHA-256; the framing above is original to this synthetic experiment, not an
industry or FIPS-certified protocol. The [Binance public-data checksum and
revision guide](https://github.com/binance/binance-public-data#checksum) provides
a concrete example of archive version identity. An archive byte checksum
does not prove importer semantics or all observed events. No data downloaded.
[Wiley's public Paleologo introduction, §1.5](https://catalogimages.wiley.com/images/db/pdf/9781394265459.excerpt.pdf)
distinguishes pretrade inputs and ex-post attribution. Verifying the actual
execution tape before attributing results is our inference, not a claim that
the book specifies this design. No full-book access, market-model fidelity,
performance, routing, investment-return or trading authority claim.

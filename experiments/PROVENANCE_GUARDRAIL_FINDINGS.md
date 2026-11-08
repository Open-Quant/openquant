# Revision association guardrail: what it prevents and what it cannot prove

The additive `openquant.provenance` module keeps a selected numeric payload attached to its declared source/revision identity, event, availability and decision. It checks that association after row reordering or cleaning. This is a useful consistency guard; calling it provider authentication or a complete-history guarantee would be incorrect.

Existing runtime files—including `data.py`, `availability.py`, package initialization and native Rust—are unchanged. Integration files are the new `python/openquant/provenance.py`, its tests, and `examples/provenance_cleaning.py` with separate tests. No hardening edits are bundled into this change.

## Explicit contract

`make_revision` copies a flat named numeric payload into an immutable record. Values must be finite builtin integers or floats; unknown clocks, nulls, booleans, nonfinite values and invalid observation chronology reject globally. Event means the observed input's event, not an arbitrary future forecast target. Input/revision IDs are explicit caller claims.

`select_revisions` selects the latest eligible event and publication per declared sample/source key at its decision. Exact duplicates are idempotent; conflicting content under one revision identity and ambiguous winning clocks fail closed. Invalid latest records are never silently discarded to recover an older value. Missing eligible groups remain explicit.

A schema-tagged digest binds IDs, clocks, decision and exact typed payload. Integer and floating values differ, including 1 versus 1.0; signed floating zero differs too. Canonical integer/float hexadecimal encodings avoid decimal serialization limits and numeric coercion. The hash represents this canonical record, not the original provider response bytes.

`rejoin_selected` requires stable identities, the exact event and the exact typed payload. It revalidates retained bundle consistency, rejects stale digests and unknown/duplicate rows, and never falls back to row position. Dropped selected rows reject by default. Explicit `require_all=False` returns dropped keys, which callers must handle. A digest cannot inspect an external system or prove that its own inputs were truthful.

## Actual cleaner example and baselines

Run `python examples/provenance_cleaning.py --source-tree` for isolated module review, or omit the flag in the qualified installed package. Both paths execute real `clean_ohlcv` with Polars 1.44.2.

The adapter deliberately normalizes finite values to Float64 before binding, rejecting lossy integer conversion. It requires exact microsecond event keys and a unique selected canonical `(symbol,event)` key. Stable sample IDs encode symbol/event unambiguously; source identity remains separate. Every cleaned numeric OHLCV field is checked against its selected payload. This narrow adapter rejects unsupported inputs rather than guessing how a provider intended them.

| Control | Key-only baseline | Association guard |
| --- | --- | --- |
| Historical value 10 survives cleaning unchanged | Accepts | Accepts |
| Value replaced by 100 under the original event/identity | Accepts | Rejects exact payload mismatch |
| Selected row silently dropped | Can disappear | Rejects by default; explicit opt-in reports dropped key |
| Two source identities collapse to one canonical key | Cleaner can deduplicate | Adapter rejects before cleaning |
| Integer 2^53+1 coerced to Float64 | Becomes 2^53 | Adapter rejects lossy normalization |
| Nanosecond event would lose precision in microsecond cleaning | Key can collide | Adapter rejects before cleaning |
| Latest revision has null numeric value | Cleaner may drop it before deduplication and retain older value | Guard/adapter rejects the input batch, including future invalid revisions |
| Coordinated new false value, backdated clock and recomputed digest | Accepts | Also accepts: no external truth anchor |

The numerical-coercion control is not theoretical: real Polars cleaning maps 2^53 and 2^53+1 to the same Float64. Binding before an undeclared coercion detects a mismatch; explicitly normalizing first is acceptable only when its precision policy is understood. This adapter preserves exact representable large integers and signed zero and rejects null/overflow/nonfinite inputs.

Independent tests retain additional falsifiable limitations. Omitting a newer revision from the candidate history lets an older revision pass consistently. A manually constructed eligible older bundle also rejoins; rejoining does not prove that selection ran over a complete history. Physically swapping identical payloads while consistently changing their declared identities is observationally indistinguishable here. Frozen Python objects prevent ordinary accidental mutation, not adversarial control of the interpreter.

## Public-data value and a concrete temporal limit

Economic data vintages are a real use case for retaining observation IDs, revisions and immutable snapshots. FRED's observation API exposes real-time periods as dates, while its vintage endpoint identifies dates when data changed. Those date-valued fields alone do not supply an exact intraday availability instant. This guard therefore cannot turn them into valid nanosecond publication clocks by filling in midnight. [FRED observations documentation](https://fred.stlouisfed.org/docs/api/fred/series_observations.html), [FRED vintage dates documentation](https://fred.stlouisfed.org/docs/api/fred/series_vintagedates.html).

FRED also distinguishes a source's release date from when data becomes available on the FRED or ALFRED websites. A release calendar is not a substitute for a historical receipt/publication log. This is a concrete reason to reject unknown availability rather than manufacture precision. [FRED release dates documentation](https://fred.stlouisfed.org/docs/api/fred/releases_dates.html).

A practical prospective pipeline can retain raw response bytes and their digest, provider record/revision identifiers, an explicitly defined local first-seen clock and the normalization policy, then bind the selected numeric representation used downstream. A local capture log is evidence of what that pipeline received; it is not proof of an earlier provider publication time. Existing final archives without such history cannot be made historically complete by rehashing them. These are implementation recommendations inferred from the documented schema and tested counterexamples, not a validated FRED adapter. No API credential, proprietary dataset or external series download was used.

## Decision

Advance the module as an explicit association guard, with its strict schema and rejection behavior intact. Do not advertise it as provenance certification, silently normalize types inside it, or bypass rejection by rebuilding a bundle around already-cleaned unverified data. Real provider adapters still need separately reviewed identity, normalization, clock and history-retention contracts. Public source documentation and synthetic/actual-cleaner controls establish integration value without implying investment performance.

## Verified public offline fixture

`examples/alfred_revision_fixture.py` uses two facts from the St. Louis Fed's own
documentation: January 1990 U.S. unemployment was reported as 5.3% on February 2, 1990,
and a March 8, 1996 revision gave 5.4%. The fixture records its source and verification
date. [Official ALFRED example](https://fred.stlouisfed.org/docs/api/fred/alfred.html).

Among these two documented points, a strict-before date cutoff of 1990-02-03
selects 5.3 while naive latest-within-fixture selects 5.4, a 0.1-percentage-point
difference. Combining the older fixture identity with 5.4 fails the offline
association check. This does not assert that these were consecutive revisions,
that no other revision occurred in the interval, or that either website supplied
that value at a particular instant. Omitting the newer fixture record changes the
selection to 5.3 while leaving its internal association valid, demonstrating the
history-completeness limitation.

No UTC timestamp is invented and the nanosecond runtime API is not called by this
offline date-domain example. ALFRED documents source/provider/FRED fallback choices
for release dates and describes ingestion as typically within one business day;
this is not a guaranteed historical receipt clock. [ALFRED help](https://alfred.stlouisfed.org/help).
The fixture adds no FRED integration, API credential use or series download.

---
title: "provenance"
description: "Select declared numeric revisions at a decision time and preserve exact value identity when cleaned rows are reordered or removed."
status: authored
last_authored: '2026-11-07'
audience:
  - quant-dev
  - platform-engineering
module: "provenance"
api_surface: "python-only"
citation:
  - "OpenQuant python/openquant/provenance.py: revision selection, typed payload identity and cleaned-row rejoining."
  - "OpenQuant examples/provenance_cleaning.py: selection before cleaning and identity-based rejoining."
python_api:
  - "provenance.Revision"
  - "provenance.make_revision"
  - "provenance.SelectedRevision"
  - "provenance.Selection"
  - "provenance.select_revisions"
  - "provenance.RejoinResult"
  - "provenance.rejoin_selected"
sidebar:
  badge: Module
---

This module associates exact numeric values with a declared sample, source, revision and
decision. It detects inconsistent rejoining after cleaning; it does not authenticate the
provider, establish complete history or prove that metadata describes real availability.
Retain raw revisions and stable IDs before [cleaning](/modules/data/) discards columns or
resolves duplicate events.

## Select before cleaning

`make_revision(row_id, source_id, revision_id, event_ns, available_ns, payload)` creates a
validated immutable `Revision`. IDs and field names are nonempty builtin strings. Clocks
are exact builtin integers in signed int64 UTC Unix nanoseconds, excluding NaT, and
availability cannot precede the observed event. These clocks describe observations, not
future forecast target dates.

Payloads are nonempty flat mappings or sequences of named pairs containing only finite
builtin integers and floats. Booleans, nulls and nonfinite values are rejected. Field
order is normalized; numeric types and signed zero are retained. Integer `1` and float
`1.0` are distinct identities. Pair inputs reject duplicate names; a mapping cannot
recover duplicate fields already collapsed upstream.

`select_revisions(revisions, decisions)` takes decision cutoffs keyed by
`(row_id, source_id)`. It selects the lexicographically greatest eligible
`(event_ns, available_ns)` pair whose two clocks are at or before the decision. Event
time has priority; this is not simply the most recently published record. The policy is
latest only among the supplied history.

```python
from openquant.provenance import make_revision, rejoin_selected, select_revisions

base = 1_577_836_800_000_000_000
second = 1_000_000_000
revisions = [
    make_revision("A", "macro", "first", base, base + second, {"rate": 1.0}),
    make_revision("A", "macro", "revised", base, base + 4 * second, {"rate": 2.0}),
    make_revision("B", "market", "first", base, base + second, {"close": 20}),
]
selected = select_revisions(revisions, {
    ("B", "market"): base + 3 * second,
    ("A", "macro"): base + 3 * second,
})
# Cleaning reordered the rows but preserved their exact typed payloads.
joined = rejoin_selected(selected, [
    ("A", "macro", base, {"rate": 1.0}),
    ("B", "market", base, {"close": 20}),
])
assert joined.selection_indices == (1, 0)
assert joined.dropped_keys == ()
assert all(bundle.revision.revision_id == "first" for bundle in joined.bundles)
print("rejoined original positions:", joined.selection_indices)
```

```text
rejoined original positions: (1, 0)
```

`Selection.bundles` and `missing_keys` each retain their relative decision-mapping order.
Exact duplicate revisions are idempotent. Conflicting content under one full revision
identity raises globally, even for future or unrequested groups. Different winning
revision IDs with identical event/availability clocks are ambiguous and raise, even when
their payloads agree. Unknown groups remain missing.

## Rejoin by identity, not position

Each input row to `rejoin_selected` is `(row_id, source_id, event_ns, payload)`. It checks
the event, exact typed payload and bundle digest, returning bundles in incoming row order
and their original `selection_indices`. Duplicate or unknown keys and changed fields fail
with `ValueError`; there is no rounding tolerance or positional fallback.

The default `require_all=True` rejects any dropped selected identity. An explicit
`require_all=False` permits subsets, including an empty subset, and reports `dropped_keys`
in original selected order. It does not permit missing selection groups to become rows.

Cleaning that converts an integer to a float, rounds a value, renames a field or removes
part of a payload breaks this exact binding. Normalize deliberately, with precision-loss
checks, before constructing revisions if conversion is required. Learned feature
transforms belong after selection and within their training fold.

## What the digest cannot prove

`SelectedRevision` binds schema, decision, IDs, clocks and typed values into a SHA-256
digest. A changed record with a stale digest is inconsistent. A fabricated but internally
consistent record can receive a new valid digest; detecting that needs trusted comparison
evidence outside the changed record. Frozen Python objects are not a security boundary.

A directly constructed bundle can contain an eligible older revision, so its existence
does not attest that latest-selection policy ran. Omitted history, forged availability,
swapped physical identities with identical values, and incomplete provider delivery remain
outside the guarantee. The strict API also rejects invalid future records instead of
silently dropping them to resurrect older values.

Continue with [required feature assembly](/modules/feature-assembly/), then audit
[training availability](/modules/availability/). The independent
[point-in-time guide](/workflows/point-in-time-data/) explains the evidence boundaries.
Normal package imports still require the existing native/package dependencies.

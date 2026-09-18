# Delivery slice example: Localized export

Synthetic continuation illustrating the OUT/RQ convention from product shaping.
The earlier brownfield example's opt-in suggestion was not approval. This example
supplies a new fictional reviewed brief, `docs/design/export-brief.md`, owning
OUT-001/RQ-001. It does not rename or alter the earlier brief's investigation IDs.
No actual product, user approval, remote work item or live result is asserted.

Owner: maintainer (supplied fixture)
Status: reviewed scope in the synthetic input; delivery artifacts remain draft
Proposed local work ID: `localized-export`

## Scope and exclusions

OUT-001: Let reporting staff request a localized CSV while preserving the current
default bytes. The fixture's RQ-001 authorizes `--timezone <IANA-zone>`: selected
exports use RFC3339 timestamps with numeric UTC offsets; invalid zones exit 2
without emitting CSV. Input instants are already unambiguous. Defaults retain
UTF-8, `id,start_time,status`, and UTC behavior. Payroll changes, presets, schema
migration, UI and a broader export platform are excluded.

## Specification decision

`requires_spec=true`; `spec_reason="Adds explicitly selected CLI behavior while
preserving the existing default contract."` Author this ticket's own OpenSpec
change, `openspec/changes/localized-export`, through the selected provider. A
parent export epic has coordination links, not a gate over this child change.

## Traceability

| Canonical requirement | Formal scenario to author | Work ID | Implementation steps | Verification status |
| --- | --- | --- | --- | --- |
| export-brief#RQ-001 → OUT-001 | Default export remains byte-compatible | localized-export | Characterization test, preserve default path | Planned exact-byte comparison |
| export-brief#RQ-001 → OUT-001 | Explicit valid zone uses offset timestamps | localized-export | Acceptance test, localized conversion, docs | Planned named-zone/offset cases |
| export-brief#RQ-001 → OUT-001 | Invalid zone emits no CSV and exits 2 | localized-export | Refusal test, input validation | Planned exit/output assertions |

These are scenarios inside one change. Parser, tests, documentation, refactor and
rollout are not five independent product tickets. Keep rationale and source
approval in the canonical brief; the formal scenarios own observable behavior.

## Dependencies and interfaces

No required predecessor work is supplied: `depends_on=[]`. Existing default
fixture and export API are the compatibility baseline to inspect before editing.
A later saved-preset ticket can depend on this work ID after its interface is
reviewed; it must own a separately finishable change when it changes behavior.

## Work record draft

```toml
schema = 1
id = "localized-export"
title = "Add an opt-in localized CSV export"
scope = "Add the reviewed timezone option; preserve default bytes; exclude payroll changes, presets, UI and migration."
requires_spec = true
spec_reason = "Adds explicitly selected CLI behavior with default compatibility."
requirements = ["export-brief#RQ-001"]
depends_on = []
acceptance = [
  "Default export bytes match the characterized fixture.",
  "An explicit valid IANA zone emits the specified offset timestamps.",
  "An invalid zone exits 2 and emits no CSV.",
]
reviewed = false

[artifacts]
brief = "docs/design/export-brief.md"
spec = "openspec/changes/localized-export"
plan = "docs/design/localized-export.md"
```

## Open decisions and next action

The example's decisions above are supplied fixture facts; actual work must link
its own review. The local document destinations and formal change still need to
be authored; this example does not install work records or claim they exist.
Validate the provider-created specification and run `ai-dlc work validate
localized-export --root .` before authorized publication. Missing artifacts must
fail validation. Implementation follows review; finish requires the configured
merged-revision gates. Planned fixture cases are not live payroll qualification.

Decision: proceed — the supplied scope is bounded; author and review the formal
change and linked local work draft before implementation or publication.

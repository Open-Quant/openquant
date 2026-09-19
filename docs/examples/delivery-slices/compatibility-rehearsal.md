# Delivery slice example: Compatibility rehearsal

Synthetic follow-on to the localized-export example. Supplied inputs: the
maintainer authorizes no-code verification, the `localized-export` work record
exists, a fresh pinned-provider read says canonical `closed`, and its reviewed
spec is archived at `openspec/changes/archive/2026-09-07-localized-export`.
Payroll sandbox access is unavailable. These are exercise facts, not live evidence.

Owner: maintainer
Status: draft verification plan
Canonical source: `docs/design/export-brief.md`, OUT-001/RQ-001
Proposed local work ID: `export-compatibility-rehearsal`

## Scope and exclusions

Rehearse the existing default fixture against a controlled payroll sandbox and
record actual compatibility evidence. Exclude code changes, new output formats,
a new behavior specification and unrelated system updates.

## Specification decision

`requires_spec=false`; `spec_reason="Verifies the already specified default
contract without changing behavior."` Reference the existing archived behavior.
Do not duplicate it in a new OpenSpec change or in design documentation.

## Traceability

| Canonical requirement | Existing formal behavior | Work ID | Verification steps | Evidence status |
| --- | --- | --- | --- | --- |
| export-brief#RQ-001 → OUT-001 | Archived Default export remains byte-compatible scenario | export-compatibility-rehearsal | Record fixture/revision, execute isolated rehearsal, compare bytes and consumer outcome | Sandbox unavailable; no live result |

## Dependencies and interfaces

`depends_on=["localized-export"]` consumes the reviewed default CSV contract and
fixture. The supplied `closed` observation supports this example's dependency,
but a real `work start` must read it afresh through its pinned provider. A
cancelled/duplicate state, absent record or unavailable read would block starting.
Do not treat a local valid graph as proof that a prerequisite completed.

## Work record draft

```toml
schema = 1
id = "export-compatibility-rehearsal"
title = "Rehearse default export compatibility"
scope = "Verify default CSV compatibility in an isolated sandbox; no code, format or payroll changes."
requires_spec = false
spec_reason = "Verifies existing specified behavior without changing it."
requirements = ["export-brief#RQ-001"]
depends_on = ["localized-export"]
acceptance = [
  "Record fixture identity, revision, invocation and controlled environment.",
  "Observe exact default bytes and the actual sandbox consumer result.",
  "Retain missing access or mismatches explicitly; do not claim unperformed checks passed.",
]
reviewed = false

[artifacts]
brief = "docs/design/export-brief.md"
spec = "openspec/changes/archive/2026-09-07-localized-export"
plan = "docs/runbooks/export-compatibility-rehearsal.md"
```

## Open decisions and next action

Prepare the local runbook and confirm its source references without contacting
the sandbox. Its actual execution depends on authorized access. Run offline work
validation after the referenced files exist; do not invent a successful result.
Publish only on actual authorization and review. When accessible, execute and
record the real observations; a failed rehearsal returns evidence for triage,
not permission to change the contract. Required project checks and applicable
PR/CI/finish gates remain in force despite `requires_spec=false`.

Decision: investigate — prepare the bounded runbook and obtain authorized sandbox
access; local fixture evidence alone cannot complete this verification outcome.

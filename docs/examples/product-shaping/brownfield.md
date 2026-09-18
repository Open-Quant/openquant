# Product brief: Localized scheduling exports

Synthetic worked example; repository observations below are supplied fixture
facts. No live payroll integration or user study was executed.

Input: The checked-in CSV contract has UTF-8 columns `id,start_time,status` and
UTC timestamps by default. A payroll importer consumes that exact header. The
maintainer requests localized timestamps and renamed columns as new defaults,
while requiring byte-for-byte default export compatibility. No decision resolves
this contradiction. A fixture reproduces the existing default export.

Owner: requesting maintainer
Status: draft
Canonical brief: `docs/examples/product-shaping/brownfield.md`
Entry path: brownfield

## Audience and problem

Reporting staff may need local-time exports; payroll is an affected consumer.
The request identifies a representation preference but supplies no observation
of reporting difficulty. Preserve the existing consumer's contract while
clarifying which increment is useful and allowed.

## Evidence and assumptions

- Observed evidence: supplied checked-in contract and fixture establish UTF-8,
  exact header and default UTC behavior. The task identifies the payroll
  consumer; its live behavior has not been tested.
- User decisions: change defaults and preserve default bytes are both explicit
  requirements. Neither overrides the other; no opt-in alternative is approved.
- Hypotheses: a separate localized mode may help reporting staff without changing
  payroll output. User impact and timezone preferences remain unverified.

## Current behavior (brownfield)

Inspect the export implementation, public contract, regression fixture and known
consumer documentation before editing. The given baseline is the exact header
`id,start_time,status`, UTF-8 encoding, UTC timestamps and stable default bytes.
Affected consumers include payroll and reporting staff.

**Compatibility:** Keep those default bytes compatible. Renaming the default header or changing timestamps violates that
boundary.

**Migration/recovery:** No data migration is established as necessary. If a separate mode is
later selected, plan opt-in rollout, retained default mode, and rollback by
withdrawing the new mode; verify actual integration separately.

## Options and trade-offs

| Option | User impact and confidence | Effort and dependencies |
| --- | --- | --- |
| Separate opt-in localized export | May help reporters while retaining payroll defaults; impact still uncertain | Bounded additive change; needs maintainer scope/timezone decision and tests |
| External conversion of existing CSV | Could serve reporters without changing the product contract | Lower product change, extra reporting step; verify quoting/timezone handling and demand |
| Replace default format | Meets representation request but violates explicit compatibility | Infeasible within current constraints; would require a separately approved migration |
| Keep current export | Preserves payroll behavior; leaves possible reporting friction | No implementation effort; appropriate if no useful reporting need is established |

## Selected outcome

OUT-001: Resolve whether a separate localized export is a useful and authorized
increment while preserving the current default contract.
Reason: no implementation can satisfy both contradictory default requirements.
A compatibility-preserving option can be proposed without presenting it as an
approved acceptance fact.

## Scope and exclusions

Clarify the default/opt-in decision and reporting need. Preserve default bytes,
header, encoding and UTC behavior. Exclude payroll changes, automatic migration,
mandatory UI, tracker publication and a broader export platform.

## Success evidence

| Requirement | Outcome | Observable criterion | Evidence/source and status |
| --- | --- | --- | --- |
| RQ-001 | OUT-001 | Record the maintainer's resolution of contradictory defaults and chosen compatibility boundary | Pending decision; do not silently choose one |
| RQ-002 | OUT-001 | Record a reporting task and required timezone behavior with source, or retain the missing evidence explicitly | Pending evidence; no invented preference |
| RQ-003 | OUT-001 | Establish a repeatable baseline for exact default bytes before any changed export | Supplied fixture exists; executing it and live payroll qualification are separate checks |

## Next slice

Ask the maintainer whether localization should be separate and opt-in, and obtain
a reporting-task example. Inspect and run baseline characterization. Exit when
RQ-001 and RQ-002 resolve a bounded outcome and RQ-003 records reproducible
compatibility evidence. Then use needs-spec and the configured formal provider
before implementing changed behavior. Reuse this canonical brief's IDs in that
handoff; a PRD can elaborate rationale if needed without renumbering requirements.

## Unresolved decisions

Which requirement changes: new defaults or byte compatibility? Is local time a
verified reporting need? Which timezone and daylight-saving behavior are intended?
The maintainer owns these choices. Live payroll verification remains unperformed;
fixture success must not be reported as live qualification.

Decision: investigate — contradictory defaults and missing reporting evidence
must remain visible; the proposed opt-in mode is an option, not invented approval.

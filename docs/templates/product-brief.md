# Product brief: <bounded problem>

Owner: <decision owner; unknown if not supplied>
Status: draft
Canonical brief: <repository path; this document owns OUT-001 and RQ-001 IDs>
Entry path: greenfield | brownfield

## Audience and problem

Who encounters which task or difficulty? Separate the requested feature from the
underlying problem. State what is unknown.

## Evidence and assumptions

- Observed evidence: <source, observation, limits; say when none is available>
- User decisions: <source and exact scope of actual direction or approval>
- Hypotheses: <unverified belief and evidence needed to test it>

## Current behavior (brownfield)

- Inspected behavior and sources: <affected users/consumers and interfaces/data>
- Compatibility: <explicit boundaries, or not applicable with reason>
- Migration/recovery: <required transition/rollback, unknown, or not applicable with reason>

For greenfield: record not applicable and identify external constraints instead.

## Options and trade-offs

Compare at least two feasible approaches and doing nothing when meaningful.
Explain user impact, confidence in evidence, effort, and dependencies in words.
Include an investigation option when the value or constraints are uncertain.

## Selected outcome

OUT-001: <smallest useful outcome, or bounded learning outcome while uncertain>
Reason: <why this option is worthwhile relative to the alternatives>

## Scope and exclusions

Include the bounded slice. Exclude adjacent features and speculative expansion.
Record compatibility that must survive. UI work is optional when relevant.

## Success evidence

| Requirement | Outcome | Observable criterion | Evidence/source and status |
| --- | --- | --- | --- |
| RQ-001 | OUT-001 | <verifiable result, not an invented acceptance fact> | <existing evidence or planned check; unknowns stay visible> |

Keep these IDs stable. Downstream documents reference this canonical brief and
its IDs; they do not create a second requirement authority. If IDs already exist,
reuse them. Business baselines, user preference and live results require evidence.

## Next slice

State the smallest action, its evidence goal, dependencies, and exit condition.
If proceeding, use needs-spec and the configured specification provider before
implementation where required. A small change may use this brief alone; expand
with a PRD only when useful. Publication requires separate user authorization.

## Unresolved decisions

Name material unknowns, contradictions, decision owners and how to resolve them.
Record actual approval separately; a recommendation is not approval.

Decision: investigate — <reason; choose proceed only when scope and material
constraints are resolved within existing authorization; stop for declined,
duplicate, out-of-scope or infeasible work; investigate for missing evidence or
unresolved contradictions. This decision does not itself authorize implementation.>

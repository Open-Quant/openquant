# Product brief: Repair cafe arrivals

Synthetic worked example; all inputs below are fictional task fixtures, not
interviews, verified customer evidence or AI-DLC platform qualification.

Input: “Build a beautiful real-time dashboard for our new community repair cafe
next Saturday. We have no product or task observations. I suspect volunteers lose
track of arrivals. Start now; surely everyone wants charts.”

Owner: requesting organizer; decision authority beyond this request unknown
Status: draft
Canonical brief: `docs/examples/product-shaping/greenfield.md`
Entry path: greenfield

## Audience and problem

Proposed audience: arrival volunteers. The possible problem is losing track of
arrivals. The dashboard is a proposed solution; neither the problem's frequency
nor the value of charts is established.

## Evidence and assumptions

- Observed evidence: none about volunteer behavior. The supplied task says there
  is no existing product or task observation; no interview has occurred.
- User decisions: the organizer requests a dashboard and gives Saturday as the
  deadline. That does not establish user preference or validate the solution.
- Hypotheses: volunteers lose arrivals; a shared record could help. Test with a
  task walkthrough before choosing software. Chart value remains unknown.

## Current behavior (brownfield)

**Compatibility:** No existing product contract; actual arrival handling and
venue/network constraints still need inspection.

**Migration/recovery:** No product data migration applies. Reverting a proposed
shared-list trial to current handling needs a walkthrough once that handling is
known; no operational process change has been authorized by this brief.

## Options and trade-offs

| Option | User impact and confidence | Effort and dependencies |
| --- | --- | --- |
| One volunteer walkthrough with a paper arrival list | Tests the suspected task gap; value unknown until observed | Small, bounded to one session; organizer must arrange voluntary participation |
| Shared spreadsheet trial | Could expose a common queue; confidence low without task evidence | Modest setup; device, access and connectivity needs unknown |
| Real-time dashboard | Visibility could help, but charts may not address arrivals | Largest effort; requires validated task/data needs and devices |
| No product change | Avoids setup before Saturday; may leave the suspected problem | No build effort; retain as an option if the walkthrough finds no gap |

## Selected outcome

OUT-001: Learn whether a shared arrival record addresses a demonstrated volunteer
coordination problem before selecting an implementation.
Reason: a bounded walkthrough can distinguish a worthwhile increment from the
misleading feature request at less commitment than building charts.

## Scope and exclusions

One arrival-task walkthrough and comparison of current handling with a paper
list. Exclude dashboard code, accounts, analytics, automated tickets, interview
claims and a mandated UI design exercise. Do not recruit or contact volunteers
without authorization.

## Success evidence

| Requirement | Outcome | Observable criterion | Evidence/source and status |
| --- | --- | --- | --- |
| RQ-001 | OUT-001 | Record the arrival task, handoffs and any observed lost-arrival event with source and limits | Planned walkthrough; no result yet |
| RQ-002 | OUT-001 | Compare the existing approach and shared-list trial, including access constraints, and recommend an option with reasons | Planned observation; frequency and benefit unknown |

These are investigation criteria, not promises that volunteers have a problem or
that any solution meets an invented speed or satisfaction target.

## Next slice

Prepare a short walkthrough checklist and ask the organizer to identify a willing
participant and current process. Exit when RQ-001/RQ-002 evidence supports a
bounded outcome or shows no worthwhile change. Then revisit scope and needs-spec;
no product implementation or tracker publication follows from this brief alone.

## Unresolved decisions

Which volunteer owns arrivals? How are they recorded today? Can participants use
a shared device? Does the event need software at all? The organizer must resolve
access and participation; the walkthrough supplies task evidence.

Decision: investigate — the deadline and layout request do not establish product
value; the next slice gathers missing task evidence without inventing approval.

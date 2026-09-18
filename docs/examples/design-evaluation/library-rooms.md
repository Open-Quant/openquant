# Library room selection: original instructional cases

All products, candidates, traces and scores below are synthetic teaching material.
No UI was executed, screenshot captured or human asked to rate these examples.
Candidate revision labels are illustrative artifact identities, not Git commits.
The author-defined expected outcomes are examples, not calibration measurements.

## Brief and reviewed-contract example

Hypothetical source `docs/product/library-rooms.md` owns OUT-ROOM (reserve the
intended room) and RQ-LOCATION (preserve chosen branch), RQ-KEYBOARD (complete the
journey by keyboard) and RQ-BRAND (retain the established component library).
Delivery slice `room-selection` links those requirements; Design PM does not
redefine them. Audience: members booking a quiet room. Scope: clarify branch and
room choice; no account system, recommendation engine or navigation redesign.

Rubric `room-r1` links those requirements. In this example only, its fictional
reviewed task choices are minimum 3 for required rated criteria, one initial
candidate plus two revisions and 30 minutes. These are sample choices, not approval
or a budget for a real task. Stop at satisfaction, exhausted budget, a material
unresolved decision, or two comparable evaluations without criterion improvement
and without a required defect resolved.

| Criterion / source RQ | Required | Observable method |
| --- | --- | --- |
| FLOW-LOCATION / RQ-LOCATION | yes | With North Annex and East Hall fixtures, select North Annex, confirm, reload; booking summary must still name North Annex and the selected room. Retain state trace. |
| KEYBOARD / RQ-KEYBOARD | yes | At the agreed viewport use Tab/Enter to reserve, open and dismiss confirmation; controls remain reachable, focus visible, and focus returns to the invoking control. Retain interaction/focus trace. |
| CLARITY / RQ-LOCATION | yes, rated | Inspect choice, confirmation and summary together; branch and room identity remain distinguishable before committing. Retain annotated captures for each state. |
| FIT / RQ-BRAND | yes, rated | Compare the supplied brand components with the candidate's choices. Judge task fit and consistency, not decorative novelty. |

The proposed 0–4 scale uses 0 for contradiction and 2 for partial fulfillment.
Unobserved criteria are unverified. Criterion-specific anchors:

| Criterion | 1: major shortcomings | 3: meets with evidence | 4: exceeds usefully |
| --- | --- | --- | --- |
| CLARITY | Confirmation emphasizes an illustration while branch identity is hidden or ambiguous. | Branch and room labels remain readable with long names in all three captured states. | Those labels remain clear and a concise disambiguating location detail resolves two similarly named rooms without an extra step. |
| FIT | An unrelated navigation scheme and new buttons obscure the established booking task. | Existing type, spacing and buttons form a consistent booking sequence matching the brand. | The same components add a well-placed location cue that solves the task's ambiguity while preserving familiar actions. |

These anchors illustrate rationale; they do not establish universal taste or
comprehensive accessibility conformance. Required behavior and required evidence
remain separate from rated criteria; no weighted average can hide a failure.

## Case 1: polished but broken

Candidate `polished@p1`, rubric room-r1, synthetic trace E-P1: select North Annex,
confirm Room 4, reload; summary displays East Hall Room 4. Author-constructed visual
rating CLARITY 4 cannot override FLOW-LOCATION=fail. KEYBOARD has no trace and is
unverified; FIT is unverified without the brand comparison. Verdict needs-work.
Finding F-P1 references FLOW-LOCATION, p1 and E-P1; reproduce reload after selecting
North Annex, correct persistence, then retest both branch fixtures. Do not label
an attractive capture as a successful journey.

## Case 2: plain and usable

Candidate `plain@p2`, room-r1, synthetic trace E-P2 preserves North Annex Room 4
through reload. Synthetic keyboard trace E-K2 reaches every action and restores
focus after dismissing confirmation. Supplied instructional captures E-V2 show
readable branch/room labels at 390px and 1280px with the established controls.
Example results: required behavior pass; CLARITY 3 and FIT 3 against the anchors.
This candidate is eligible under the fictional room-r1 contract. Its simpler
visual treatment does not imply low task quality; human preference is unmeasured.

## Case 3: deliberate brand constraint

Candidate `brand@b1` uses unchanged official buttons and typography plus a location
cue beside each room. Illustrative FIT 4 is justified by solving the room ambiguity
within the existing system. Reuse is not an originality defect. Only captures
exist for b1: required interactions stay unverified and b1 is not eligible yet.
For a small spacing correction, one concise report can link the existing brief
and rubric; do not invent a new visual direction or repeat the PRD.

## Case 4: inaccessible interaction

Candidate `keyboard@k1`, room-r1, synthetic trace E-K1 opens confirmation using
Enter, but dismissal moves focus to an unreachable hidden element. KEYBOARD=fail,
severity required journey failure. Specify viewport, starting focus and the
open/dismiss sequence in F-K1; correction must be followed by keyboard retest.
A screenshot of a visible focus ring cannot override this interaction failure.
No broader accessibility compliance claim follows from this one synthetic case.

## Case 5: screenshot-only mockup

Candidate `static@s1`, room-r1, is a supplied illustrative 390px frame with a clear
booking button. Visual hierarchy may receive a reasoned rating limited to that
frame. FLOW-LOCATION and KEYBOARD remain unverified. If the current generator
reviews its own mockup, mode=self-review. Handoff to a fresh session or human with
candidate access once it exists; no independent evaluator or executed tool is
implied by writing this report. Contract verdict unverified, not accepted or 0.

## Bounded selection and continuation

A fictional run retained plain@p2 before keyboard@k1 and static@s1. It used its
initial candidate plus two revisions and all 30 minutes. Select plain@p2 under
room-r1 with its E-P2/E-K2/E-V2 evidence; preserve later findings and references.
Do not select the latest merely because it cost more. This is a synthetic decision,
not a measured user preference or real project acceptance.

If the owner proposes room-r2 with a new required capacity warning, retain r1 and
its selection. No candidate qualifies under r2 until reevaluated; do not copy r1
passes into r2. Stop with unverified/needs-work for the new contract and propose a
specific follow-up budget. An override can record a risk decision but cannot
relabel failed checks, invent captures or bypass normal implementation gates.

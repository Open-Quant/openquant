# Design PM calibration protocol — unrun

This protocol belongs to the separate design-pm-calibration work item (#16 in
the framework backlog). These instructions and synthetic examples do not prove
better interfaces, reliable evaluators, human preference or cost effectiveness.
No human labels, comparison runs or paid generation are supplied here.

## Freeze a comparison before running it

Choose a decision owner and explicit task budget: cases, repetitions, candidate
and revision ceiling, time/token/spend ceiling, allowed tools, actual model/harness
settings, stopping conditions and evidence storage. Approval to use Design PM is
not approval for this experiment or paid calls. Preserve the existing generic
skill-evaluation protocol and budget; this experiment has its own decision.

Use matched tasks with three conditions: current guidance; rubric-only generation;
rubric plus independent evaluation. Keep task requirements, generation resources,
contract versions and budgets comparable. Counterbalance run/order exposure and
hide condition labels from human raters where practical. Record deviations rather
than quietly giving a preferred condition more retries or stronger tools.

## Cases and labels

Use the original library-room cases to explain expected behaviors: polished but
broken; plain usable; required brand reuse; keyboard failure; static-only mockup.
Their written outcomes are author-defined teaching expectations, not human labels.
Prepare additional cases in distinct products and reserve a held-out partition
before tuning. Do not tune prompts on held-out outputs or expose answer keys to
evaluators. Record leakage and reclassify contaminated cases as development data.

Humans label required defects and explain preferences before prompt tuning.
Record who reviewed, consent/access limits, exact candidate/rubric identity and
rater disagreement. Missing human review remains pending. Do not replace human
labels with another model's judgment or a filled template.

## Per-run record

Retain case/condition and split, source task/RQ IDs, rubric/version, candidate IDs
and revisions, prompts/access instructions, actual tools/model/harness/session,
review mode, observed states, evidence references/digests, scores/findings,
selection/stop reason, actual elapsed time and usage when available. Mark unavailable
metrics unknown; do not infer tokens or cost from elapsed time. Separate fixture,
supplied capture and actual observed evidence. Keep secrets/private work data out.

Measure human pairwise preference, missed required defects, false-positive findings,
rater disagreement and observed cost/time. Keep required failures separate from
taste. Compare candidates only under the same contract version. Preserve repeat
variance and negative results; one favorable example is not a general improvement.
If tools or reviewers are unavailable, record not-run/unverified and the exact
missing prerequisite, not a success score.

## Review and continuation

An independent reviewer inspects raw inputs, outputs and scoring decisions,
including disagreements and held-out treatment. Record limits on generalization,
which clients were actually exercised and whether the budget was comparable.
Human review, live browser/client execution and measured quality gain are all
pending until supplied by actual runs. Calibration does not create a new project
finish gate or authorize implementation, deployment or remote publication.

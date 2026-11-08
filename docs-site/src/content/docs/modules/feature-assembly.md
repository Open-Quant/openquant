---
title: "feature_assembly"
description: "Project ordered, complete feature rows from selected source revisions and report missing inputs without imputing or fitting a model."
status: authored
last_authored: '2026-11-07'
audience:
  - quant-dev
  - platform-engineering
module: "feature_assembly"
api_surface: "python-only"
citation:
  - "OpenQuant python/openquant/feature_assembly.py: ordered projection, source lineage, missing-input exclusions and coverage counts."
  - "OpenQuant examples/required_features.py: declared source requirements, informative missingness and downstream candidate-index intersection."
python_api:
  - "feature_assembly.FeatureRow"
  - "feature_assembly.MissingFeature"
  - "feature_assembly.ExcludedRow"
  - "feature_assembly.FeatureCoverage"
  - "feature_assembly.AssemblyResult"
  - "feature_assembly.assemble_features"
sidebar:
  badge: Module
---

`assemble_features(feature_spec, decisions, selection)` projects complete rows from
[selected revision bundles](/modules/provenance/). It performs no coercion, imputation,
learned transformation or model fitting. A row with one missing required input is
excluded explicitly rather than given a misleading aggregate availability timestamp.

## Declare column and row order

The feature specification is an ordered, nonempty sequence of
`(output_name, source_id, payload_field)` triples (`FeatureSpec`). Output names must be
unique nonempty builtin strings. Sets and mappings are rejected as specifications;
aliases may intentionally project one source field into different named columns.
Aliases do not establish statistical independence.

`decisions` maps stable row IDs to exact builtin int64 UTC Unix-nanosecond decisions,
excluding NaT. Its insertion order determines output row order, after exclusions. A
required source bundle must be bound to exactly that requested decision, even if its
required field is missing.

```python
from openquant.feature_assembly import assemble_features
from openquant.provenance import make_revision, select_revisions

base = 1_577_836_800_000_000_000
second = 1_000_000_000
decisions = {"B": base + 3 * second, "A": base + 3 * second}
specification = (("rate", "macro", "rate"), ("price", "market", "close"))
revisions = [
    make_revision("A", "macro", "one", base, base + second, {"rate": 1.5}),
    make_revision("A", "market", "one", base, base + second, {"close": 10}),
    make_revision("B", "market", "one", base, base + second, {"close": 20}),
]
source_decisions = {
    (row, source): decision
    for row, decision in decisions.items()
    for source in ("macro", "market")
}
result = assemble_features(
    specification, decisions, select_revisions(revisions, source_decisions)
)
assert result.requested_count == 2 and result.complete_count == 1
assert result.rows[0].row_id == "A" and result.rows[0].values == (1.5, 10)
assert result.excluded[0].row_id == "B"
assert result.excluded[0].missing[0].reason == "missing_source"
assert [item.present for item in result.coverage] == [1, 2]
assert result.rows[0].max_available_ns == base + second
print("complete rows:", [row.row_id for row in result.rows])
print("excluded rows:", [row.row_id for row in result.excluded])
```

```text
complete rows: ['A']
excluded rows: ['B']
```

## Interpret the result

- `feature_spec` and `requested_rows` preserve the declared orders. `rows` contains only
  complete `FeatureRow` records; each `values` tuple follows feature order and preserves
  builtin numeric type and signed zero.
- Each complete row retains contributing source `bundles` once per source, even if
  several columns use it. `max_available_ns` is the maximum availability of those bundles.
- `excluded` contains `ExcludedRow` records with a `MissingFeature` for every absent
  source or payload field. Reasons distinguish `missing_source` from `missing_field`.
- `coverage` counts present, missing-source and missing-field cases per feature across
  every requested row, including excluded rows. `requested_count`, `complete_count`
  and `excluded_count` summarize rows, not the number of missing cells.

Every supplied bundle is revalidated, including unused records. Valid unused sources do
not alter the projection. Empty row requests are allowed, but an empty feature
specification is not. Invalid identities, clocks, duplicated output names, inconsistent
bundles and required-source decision mismatches raise `ValueError`.

## Keep missingness and lineage visible

Each selected revision is a complete snapshot. If its payload drops a required field,
assembly reports that field missing; it never recovers the field from an older revision.
Normalize incremental provider updates into explicit snapshots before using this API.

Complete declared inputs do not prove that the specification includes every real feature
dependency. Nor does a high coverage count establish representative sampling: dropping
rows can change the target distribution. Keep excluded IDs and reasons with the study
and assess informative missingness separately.

Use [availability filtering](/modules/availability/) to intersect original splitter
indices after aligning complete feature rows by stable IDs. Do not index the compacted
matrix with old original positions. Direct construction of output dataclasses is not a
substitute for the validation performed by `assemble_features`. Provider truth, label
maturity, purging, fold-local preprocessing and predictive quality are separate checks.
Normal `openquant` imports retain the existing native/package dependencies.

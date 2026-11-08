# Availability audit: workflow qualification and limits

The helper is useful as a strict eligibility filter inside a provenance-preserving pipeline. It is not a vintage database, provider adapter, splitter, preprocessing audit, or proof of leakage-free predictions. Tests with the actual cleaner show why those boundaries matter: truthful-looking timestamps can be attached to the wrong values or rows and pass every timestamp check.

## Reproducible workflow

`examples/availability_end_to_end.py --source-tree` runs a fixed synthetic protocol: seeds7,19,43;96 stable-ID rows per seed; irregular3–9-hour gaps; fits at positions48,64,80; eight subsequent decisions per fit. Labels resolve two hours after the historical decision; some publish36 hours later. Initial sources sometimes arrive after the decision. Final revisions deliberately contain the outcome divided by two and arrive later: this is an outcome-leakage positive control, not a plausible performance advantage.

Four training arms separate historical/final feature vintages and published/resolved-only labels. Each is scored using both historical and final prediction features. All arms retain the same pre-existing candidate exclusions. Learned centering and scalar OLS use only their selected training rows. Predictions use each test decision's own source cutoff, not the earlier model-fit cutoff. Test targets are used later for retrospective scoring, never to fit the historical model.

Across nine folds there are72 test decisions,63 with a usable historical source and nine explicitly missing. Every error comparison below uses exactly the same63 IDs, including the zero baseline. Folds share training data; these are descriptive fixed-control results with no independence, confidence-interval or generalization claim. Training counts are row uses across folds, not distinct observations.

| Training features / label policy | Historical test-feature MSE | Final test-feature MSE | Unpublished training-label uses |
| --- | ---: | ---: | ---: |
| Historical / published | 0.468827 | 0.062908 | 0 |
| Historical / resolved only | 0.465745 | 0.063296 | 9 |
| Final / published | 0.510899 | approximately0 | 0 |
| Final / resolved only | 0.510899 | approximately0 | 9 |
| Zero baseline | 3.512616 | 3.512616 | 0 |

The tiny error change from admitting unpublished labels is retained; leakage need not create a dramatic score jump. Final-trained models perform worse than the historical-trained model on historical prediction features here. Near-zero final/final error is constructed by encoding the target in the revised feature. Final training arms also use56 or65 source revisions that were not yet available even at fit. Revisions known at fit but newer than the historical decision violate this helper's historical-vintage contract; they can belong to a different, explicitly designed training policy. These two situations are reported separately.

Independent review regenerated the data and verified nine folds,36 training arms,504 predictions and72 score sets, plus input-reordering and future-revision-append invariance. This example's exclusions are explicit synthetic stand-ins, not a claim to have run the native splitter.

## Actual cleaner integration

`examples/availability_cleaning.py --source-tree` executes the repository's real `data.clean_ohlcv` with locked Polars1.44.2. It directly loads the real modules, without mocking the cleaner, but does not initialize the native package.

The historical-vintage pipeline selects the initial value10, then cleans and joins metadata by a unique canonical event key; output values are10 and20. A null row is dropped and counted. Cleaning the complete archive instead retains revised value100. Pairing100 with the original release timestamp causes both rows to pass the availability audit despite one true future revision. Joining pre-cleaning metadata positionally swaps both output row identities, yet both supplied-clock audits again pass.

All24 input permutations preserve the historical-vintage result. The naive archive's default `keep='last'` follows surviving input order, as documented; it is not a publication-time revision selector. An eligible latest revision containing a missing value is dropped rather than silently backfilled from an older revision. Ambiguous latest publication ties are rejected. The example rejects submicrosecond event keys because canonical cleaning emits microseconds; it does not silently collapse source identities at that boundary. These are restricted adapter preconditions, not changes to the cleaner.

## Conditions for usable integration

1. Preserve immutable source revision IDs, actual availability and historical decisions before cleaning. Select the historical vintage before collapsing duplicate event keys. A final archive cannot recover lost publication history.
2. Aggregate availability over every required source input. If any required clock is unknown, the aggregate remains unknown. Taking a skip-null maximum can falsely admit a row.
3. Preserve row identity through dropping, sorting and deduplication. A key join is safe only when the selected revision is unique for that key and timestamp normalization cannot merge distinct events. Matching keys alone does not authenticate the value-vintage association.
4. Intersect the audit with the splitter's existing training candidates. Treat empty retained training as an explicit failure/skip; never fall back to excluded rows. Purging, embargo and release availability are separate contracts.
5. Audit source information before fitting fold-local transforms. A deterministic reconstruction may run later; a train-only transform may be fitted at model-fit time. Neither computation timestamp should replace source availability.
6. At prediction time, check sources against each prediction decision, not merely the earlier training cutoff. Report unavailable predictions and denominators explicitly. Coverage can be informative: a separate synthetic control has selected-row accuracy.8 versus full-row accuracy.5, and selected-target mean.6 versus full mean0.
7. Treat the result as compliance with supplied clocks. Backdated metadata, false value associations, learned-transform leakage and provider errors can remain undetected. No aggregate “leakage-free” status follows.

The twenty original scientific labs and the availability helper remain unchanged. The examples, tests, full workflow report and independent reviews are preserved in the workflow checkpoint. Native package/splitter qualification is reported separately; these synthetic results do not establish investment performance.

## Installed package and native splitter qualification

The exact pinned Rust1.98.1 toolchain was verified against official distribution
metadata and installed in an isolated workspace. A locked dev-profile wheel build
completed in76 seconds. A separate Python environment ran both examples through
normal installed-package imports; outputs matched source-tree mode apart from the
cleaning example's explicit import-mode field. The installed availability source
matches the frozen helper exactly.

The targeted CV/helper/example suite passed97 cases with one optional scikit-learn
case skipped initially; installing that optional dependency in the isolated native
environment allowed the remaining case to pass. These98 cases are not a full
repository CI run. Six tests used the actual native splitter to verify closed-span
purging and publication boundaries. Independent review checked60 more compositions
across30 shuffled, irregular-clock inputs with stable-ID alignment. No maturity
option restored rows the splitter had purged.

This closes the earlier native-package and splitter gap for the tested Linux
x86_64 wheel and workflows. It does not validate arbitrary provider schemas,
all-platform packaging, performance of a release build, or the full project test
suite. The private dev wheel is preserved separately with its hash; it was not
published or deployed. Existing project manifests/locks and the original numerical
research environment were preserved.

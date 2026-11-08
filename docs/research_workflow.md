# Notebook-first research workflow

Choose the workflow by the evidence you need. The notebook runbooks test stated
hypotheses; the convenience pipeline and experiment CLI demonstrate synthetic
component calculations and artifact generation. Running the CLI after a notebook
does not transfer that notebook's fitted model, data or out-of-sample predictions.

## Choose an output

| Goal | Entry point | Result and limits |
| --- | --- | --- |
| Fit a baseline and trace held-out predictions | [Research journey](../docs-site/src/content/docs/workflows/research-journey.md), `examples/research_journey.py` | Classification report with folds and predictions; not an account-level P&L backtest |
| Reproduce a numbered research claim | [Notebook runbooks](../notebooks/python/README.md), notebooks 09–14 | Executed hypothesis, controls, trial inventory and decision; inspect each notebook's scope |
| Exercise synthetic pipeline artifacts | `experiments/run_pipeline.py` | Manifest, frames, plots and threshold decision; no predictive model is trained |
| Inspect multiple CLI configurations | Same CLI with `--grid-config` | In-sample leaderboard over the generated data; ranking does not validate a winner |

The [notebook contract](../docs-site/src/content/docs/workflows/research-notebook-contract.md)
and [site workflow](../docs-site/src/content/docs/workflows/notebook-research-workflow.md)
explain the research controls and reproduction checks.

## Local commands

Set up once from the repository root using the
[Python bindings guide](../docs-site/src/content/docs/setup/python-bindings.md).
After changes to Rust or bindings, rebuild the release extension and restart loaded
interpreters before running the checks below.

```bash
# Small component smoke; does not execute every notebook
uv run --no-sync --python .venv/bin/python python notebooks/python/scripts/smoke_all.py

# Execute one notebook and compare outputs without rewriting notebooks or figures
uv run --no-sync --python .venv/bin/python python notebooks/python/scripts/run_notebooks.py --only 12 --check

# Inspect artifact generation and shared-pipeline consistency
uv run --no-sync --python .venv/bin/python pytest python/tests/test_experiment_scaffold.py -q
```

The scaffold tests compare selected outputs from two paths through the same
implementation. They check artifact structure and deterministic plot output; they
are not an independent numerical reference or evidence of predictive performance.
A partial notebook check omits the full exported-figure comparison. Notebook code
can still write caches and configured registries during `--check`.

For an artifact run, choose a new output root and preserve it with the result:

```bash
uv run --no-sync --python .venv/bin/python python experiments/run_pipeline.py --config experiments/configs/futures_oil_baseline.toml --out experiments/artifacts/review-001
uv run --no-sync --python .venv/bin/python python experiments/run_pipeline.py --config experiments/configs/futures_oil_baseline.toml --grid-config experiments/configs/futures_oil_grid.toml --out experiments/artifacts/grid-review-001
```

Replace these example directory names if they already exist. The CLI writes to
configuration-hashed child directories and can overwrite a prior run at the same
root. The hash is not a unique execution identifier. See the
[artifact readback example](../docs-site/src/content/docs/examples/catalog.md#run-a-configuration-grid-and-read-its-artifacts)
for manifest and row-alignment checks.

## Read the decision with its assumptions

The CLI's `decision.md` reflects threshold flags from `run_flywheel_iteration`.
`passed_alignment_guard` checks timestamp ordering; `passed_event_order_guard`
checks event-index ordering. The returned `inputs_aligned` and
`has_forward_look_bias` fields are deprecated constants, not measured leakage
diagnostics. Input-length mismatches raise errors; neither constant can detect
future information embedded in supplied probabilities.

Before treating a research result as evidence, record:

- A hypothesis and invalidation rule fixed before inspecting the result.
- Actual data, features, labels, fit boundaries and prediction provenance.
- Label spans, purge/embargo settings and training-only preprocessing.
- All attempted configurations and the choices that define their identities.
- Gross and net metrics with cost, return-frequency and risk-free conventions.
- The source revision and dirty state, interpreter and extension paths, dependency
  versions, seeds and retained data or immutable references.

`TrialRegistry` counts distinct explicit configuration keys, replacing the record
when a key repeats. It does not infer a changed dataset or model setting. The
[trial identity contract](../docs-site/src/content/docs/workflows/research-notebook-contract.md#trial-identity-is-part-of-the-method)
describes the implications for adapted notebooks. A timestamp or hash printed in a
footer does not reconstruct the inputs or prove that the loaded extension matches
source. Preserve the evidence needed to replay the run.

Passing threshold flags or execution checks does not authorize trading or establish
out-of-sample validity. Use the held-out evaluation and controls appropriate to the
research question, and report failures and unmeasured assumptions with the result.

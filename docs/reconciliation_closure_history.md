# Beads Closure Reconciliation (OQ-mef.10)

Date: 2026-02-13

## Scope
Reconcile closed issue claims against repository contents on `main` and align tracker/docs state
to what is actually delivered.

## Evidence Summary

### Python bindings and package track (`OQ-u5h` subtree)
- `OQ-caa` scaffold claim: present.
  - Evidence: `crates/pyopenquant/Cargo.toml`, `crates/pyopenquant/src/lib.rs`, `pyproject.toml`.
- `OQ-xhk` API surface/conversions claim: present.
  - Evidence: `crates/pyopenquant/src/lib.rs`, `python/openquant/__init__.py`.
- `OQ-8yl` pytest harness claim: present.
  - Evidence: `python/tests/test_bindings_contract.py`, `python/tests/test_pipeline_api.py`.
- `OQ-pqz` CI smoke workflow claim: present.
  - Evidence: `.github/workflows/python-bindings.yml`.
- `OQ-alv` install/API docs claim: present.
  - Evidence: `docs/python_bindings.md`, `README.md` links.

### Notebook-first platform track (`OQ-ruu` subtree)
- `OQ-2ub` notebook starter pack claim: present.
  - Evidence: `notebooks/python/*.ipynb`, `notebooks/rust/*.rs`, folder READMEs.
- `OQ-gz9` adapters/viz claim: present.
  - Evidence: `python/openquant/adapters.py`, `python/openquant/viz.py`.
- `OQ-jmo` pipeline claim: present.
  - Evidence: `crates/openquant/src/pipeline.rs`, `python/openquant/pipeline.py`.
- `OQ-1na` experiment scaffold claim: partially present.
  - Present: config + deterministic tabular artifacts + manifest/decision note.
  - Evidence: `experiments/run_pipeline.py`, `experiments/configs/futures_oil_baseline.toml`,
    `python/tests/test_experiment_scaffold.py`.
  - Gap: acceptance explicitly included plot artifacts; current runner does not emit plot files.
- `OQ-7uk` notebook/example smoke CI claim: present.
  - Evidence: `.github/workflows/notebooks-examples-smoke.yml`,
    `crates/openquant/examples/research_notebook_smoke.rs`.
- `OQ-vuy` workflow docs claim: partially present.
  - Present: `docs/research_workflow.md`, README links.
  - Gap: acceptance explicitly required a dedicated docs-site workflow page linked from
    getting-started/examples; no such page exists under `docs-site/src/pages/`.

## Discrepancies and Tracking Actions
- Missing docs-site workflow page/linkage:
  - Follow-up issue created: `OQ-ojp`.
- Missing plot artifacts in experiment scaffold outputs:
  - Follow-up issue created: `OQ-det`.

Both follow-ups are linked as `discovered-from` `OQ-mef.10`.

## Conclusion
- Closed-issue claims for bindings, notebooks, pipeline, and CI smoke are materially present in tree.
- Two acceptance-criteria gaps remain and are now explicitly tracked as open issues.
- `docs/project_status.md` has been updated to reflect this reconciled state.

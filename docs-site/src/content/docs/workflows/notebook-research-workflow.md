---
title: Notebook Research Workflow
description: How the research notebooks are structured, run, checked in CI and published to the research gallery.
status: authored
last_authored: '2026-09-26'
audience:
  - quant-dev
  - platform-engineering
sidebar:
  order: 3
---

Research in OpenQuant happens in Jupyter notebooks under `notebooks/python/`, committed with
their outputs. The notebooks numbered 09 and up are **runbooks**: each tests one claim from
AFML on data where the answer is known and ends with a decision about what later work should
adopt. Their results are collected in the [research gallery](/runbooks/).

## Structure of a runbook

A runbook's sections, in order:

1. **Setup**: imports, parameters and the data source. `OPENQUANT_RUNBOOK_SOURCE` points a
   runbook at your own file instead of the committed SYNTHETIC sample.
2. **Hypothesis**: numbered claims (`H1`, `H2`, ...) with their pass/fail rules, written before
   the full run.
3. **Data** and **Method**: what is simulated or loaded, the splits, the costs and the trial grid.
4. **Results** and **Analysis**: tables and figures, then the verdict on each hypothesis.
5. **Promotion decision**: what is promoted (usually a procedure, not a strategy), under which
   conditions, and what is not.
6. **Self-review checklist** and **Reproducibility**: versions, data hashes, seed, git commit.

The gallery page is generated from these sections, so a runbook that follows them appears there
without anyone copying text.

## Running the notebooks

```bash
uv venv --python 3.13 .venv
uv sync --group dev
uv run --python .venv/bin/python maturin develop --release --manifest-path crates/pyopenquant/Cargo.toml

just notebooks-run              # execute every NN_*.ipynb in place
just notebooks-run --only 12    # one notebook
just notebooks-verify           # compare the outputs with the committed ones
```

Figures are drawn with `nbfigures.figure(name, draw)`, which shows the plot inline and writes a
light and a dark SVG to `docs-site/public/figures/notebooks/` for the docs pages to embed.

## What CI checks

`.github/workflows/notebooks.yml` executes every notebook on pull requests that touch
notebooks, Python or the crates, and nightly. A cell error fails the job, and so does any
difference between the fresh outputs or figures and the committed ones beyond float noise. After
changing a notebook, re-run it, commit the notebook and its figures, and regenerate the gallery
with `python3 scripts/docs/generate_site_pages.py --write`; the docs job fails while the gallery
disagrees with the notebooks.

The rules a runbook's result is held to (purged splits, trial registries, controls with a known
answer) are on the [Governance](/project/governance/) page.

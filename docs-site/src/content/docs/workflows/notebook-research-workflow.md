---
title: Notebook Research Workflow
description: How the research notebooks are structured, run, checked in CI and published to the research gallery.
status: authored
last_authored: '2026-11-07'
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

1. **Setup**: imports, parameters and the data source. Read the notebook's own setup before
   supplying data: several accept `OPENQUANT_RUNBOOK_SOURCE`, while notebook 10 uses its
   `SOURCE` parameter. The runner does not provide one universal data-source override.
2. **Hypothesis**: numbered claims (`H1`, `H2`, ...) with their pass/fail rules, written before
   the full run.
3. **Data** and **Method**: what is simulated or loaded, the splits, the costs and the trial grid.
4. **Results** and **Analysis**: tables and figures, then the verdict on each hypothesis.
5. **Promotion decision**: what is promoted (usually a procedure, not a strategy), under which
   conditions, and what is not.
6. **Self-review checklist** and **Reproducibility**: data hash and seed, with the git commit and
   versions on stderr.

The gallery page is generated from these sections, so a runbook that follows them appears there
without anyone copying text.

The full rules are the [notebook contract](/workflows/research-notebook-contract/): what each
section must contain, the `nbrepro.footer(...)` reproducibility cell every notebook ends with,
and the API tours (notebooks 01 to 08), which make no claims and so have no hypothesis or
decision. `just notebooks-lint` checks the headings and the footer on every pull request.

## Running the notebooks

```bash
uv venv --python 3.13 .venv
uv sync --locked --group dev --no-install-project
uv run --no-sync --python .venv/bin/python maturin develop --release --manifest-path crates/pyopenquant/Cargo.toml

just notebooks-run              # execute every NN_*.ipynb in place
just notebooks-run --only 12    # one notebook
just notebooks-verify           # compare current files with HEAD; does not execute
just notebooks-run --check      # execute to temporary files and compare with this checkout
```

Use a checkout with the intended local extension and dependencies. The runner inherits
environment variables, including notebook-specific data and registry overrides. Clear
unintended overrides before reproducing the committed synthetic results.
Execution uses the Jupyter kernel named `python3`, with the notebook's directory
as its working directory. If you have custom kernel registrations, inspect the
kernel's interpreter and import paths rather than assuming they match the shell.
Temporary `--check` outputs do not sandbox other writes: a notebook may still
populate its data cache or write to a configured trial registry.

Figures are drawn with `nbfigures.figure(name, draw)`, which shows the plot inline and writes a
light and a dark SVG to `docs-site/public/figures/notebooks/` for the docs pages to embed.

An in-place full run deletes the existing notebook SVGs first and regenerates them. A run
with `--only` leaves other figures in place; a partial `--check` also omits SVG comparison.
Use a full check to detect missing, extra or changed exported figures. `--only` matches a
filename prefix or full stem; use the two-digit number and inspect `--list --only 12` if
the selected scope is uncertain.

## Inspect the interpreter inside the kernel

A shell import check proves what the launcher imports, not what the named Jupyter kernel
will import. From the repository root, the following diagnostic creates a temporary
one-cell notebook, executes it through the same single-notebook runner, and reads the
paths printed **inside** that kernel. It uses the notebook development dependencies
already installed by setup. It does not register kernels or modify research notebooks.

```python
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

import nbformat

runner = Path("notebooks/python/scripts/execute_notebook_cells.py").resolve()
assert runner.is_file(), "run this diagnostic from the repository root"
cell = '''import json, sys
from pathlib import Path
from importlib.metadata import version
snapshot = {
    "executable": sys.executable,
    "prefix": sys.prefix,
    "python": sys.version.split()[0],
    "working_directory": str(Path.cwd()),
}
try:
    snapshot["distribution"] = version("pyopenquant")
except Exception as error:
    snapshot["metadata_error"] = f"{type(error).__name__}: {error}"
try:
    import openquant
    import openquant._core
    snapshot["package"] = str(Path(openquant.__file__).resolve())
    snapshot["extension"] = str(Path(openquant._core.__file__).resolve())
except Exception as error:
    snapshot["import_error"] = f"{type(error).__name__}: {error}"
print(json.dumps(snapshot))
'''
with TemporaryDirectory(prefix="oq-kernel-probe-") as directory:
    source = Path(directory) / "probe.ipynb"
    output = Path(directory) / "executed.ipynb"
    nbformat.write(nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(cell)]), source)
    subprocess.run([sys.executable, str(runner), str(source), "--out", str(output)], check=True)
    notebook = nbformat.read(output, as_version=4)
    stdout = "".join(item["text"] for item in notebook.cells[0].outputs
                     if item.output_type == "stream" and item.name == "stdout")
    snapshot = json.loads(stdout)
    assert Path(snapshot["working_directory"]).resolve() == Path(directory).resolve()
    print("launcher:", sys.executable)
    print(json.dumps(snapshot, indent=2))
```

The paths and versions are intentionally environment-specific. Compare `launcher` with
`executable` and `prefix`, and check both `package` and `extension` against the installation
you intended. If `import_error` appears, the receipt still identifies the kernel that failed: fix that
environment's installation or kernel selection before running research code. `metadata_error`
separately records missing or unreadable distribution metadata. The absence of package paths
is not a successful import check. A wheel normally points into its environment's `site-packages`; an editable
checkout normally points into `python/openquant`. A matching version string alone does
not prove matching source or a recently rebuilt native binary. The working-directory
assertion verifies the runner's notebook-directory contract for this temporary fixture.

For an unexpected interpreter, inspect the `python3` registration from the same prepared
environment with `uv run --no-sync --python .venv/bin/python python -m jupyter kernelspec list`.
The executor explicitly requests `python3`; changing only notebook metadata does not select
a different kernel here. Resolve the registration/environment mismatch before research runs,
and restart an already running kernel after rebuilding the extension.

Ordinary import and metadata exceptions are recorded instead of preventing the receipt.
A kernel that cannot start, crashes or times out may produce no receipt; inspect the runner
error in that case. This probe tests a clean first cell. A real notebook can subsequently modify `sys.path` or
load another package during setup; inspect the same fields after its setup when diagnosing
that notebook. The temporary output files are removed when the context exits. Save the
printed diagnostic with your run evidence if needed; it is an environment receipt, not proof
that every feature, label or dependency came from the recorded source revision.

## Preserve trial identity

`TrialRegistry` counts distinct explicit configuration keys. Recording the same key again
replaces its return summary; it does not append another attempt. The
[trial-identity contract](/workflows/research-notebook-contract/#trial-identity-is-part-of-the-method)
includes a runnable example of this behavior. Current notebook keys have specific limits. Notebooks 11, 13 and 14 use
`OPENQUANT_RUNBOOK_REGISTRY`; notebook 12 uses `OPENQUANT_RUNBOOK_REGISTRY_DIR`:

| Notebook | Persistent registry override | Identity currently recorded |
| --- | --- | --- |
| 11, meta-labeling | Supplied path for the planted study; sibling `<stem>-control.json` for the control | Planted label is fixed. Control label contains 12 characters of the dataset hash. Both add the trial-grid fields but omit global choices such as model penalty and some event/cost settings. |
| 12, CPCV | Directory containing `nb12-null.json`, `nb12-planted.json` and `nb12-data.json` | MA-grid fields plus a dataset descriptor. Simulations identify name, strength and repetition, omitting seed and other generator settings; fetched data includes source and full hash but not all protocol settings. Monte Carlo paths are not appended to these files. |
| 13, bet sizing | Supplied path for the control; planted registry remains temporary | Control label contains 12 characters of the dataset hash plus sizing-grid fields, not all model/event settings. |
| 14, fracdiff classifier | Supplied path for the control; planted registry remains temporary | Only data label, feature-set name and cost. Custom files all use `control-user data`; their content and requested date range are not in the key. |

For an adapted study, construct the configuration passed to **every** `record` call from
one shared identity plus the candidate's choices. Include the full input-content hash,
symbol universe and time range, preprocessing and label settings, model parameters,
split/embargo settings, costs, and the implementation/protocol revision. Include stochastic
settings when they distinguish candidate choices. Keep a rerun of the same experiment under
the same identity; use a separate execution log for timestamps, machine details and repeat
runs. A fresh timestamp on every key would turn ordinary reruns into extra selection trials.
The reproducibility footer does not fill in missing registry fields for you.

Before running, preserve the current registry and record the intended comparison family and
trial inventory. Use a single writer for each registry: atomic file replacement does not
merge concurrent writers. Inspect the recorded configurations and count afterward; count
alone cannot reveal a replaced trial. Keep failed, abandoned and post hoc choices in the
research inventory, even when they do not have usable returns for `record`.
Do not reset an ongoing study's registry to discard unsuccessful trials, or combine unrelated
studies just to increase its count. DSR still depends on the assumptions of the return stream
and the trial family; complete keys do not validate the statistical design.

To reproduce committed synthetic outputs, clear unintended source, range, symbol, repetition
and registry overrides and use the notebook's default temporary registries. For an adapted
study, choose explicit registry and figure paths outside the checkout and retain them with
the input snapshot and executed notebook. These controls preserve evidence; they do not
change the existing notebook keys listed above. Update those keys before relying on a
persistent registry across new data or protocol choices.

## What CI checks

`.github/workflows/notebooks.yml` executes numbered notebooks on matching pull requests,
matching pushes to main, nightly and manual dispatch. Its path filter includes notebooks,
Python, crates, notebook figures, dependency manifests/lockfiles and the runner's workflow.
A cell error fails the job. The following comparison checks sources, text outputs and SVGs,
normalizing decimal output numbers to four significant digits; it ignores stderr and raster
image bytes. Passing it is not pixel-level visual validation. After
changing a notebook, re-run it, commit the notebook and its figures, and regenerate the gallery
with `python3 scripts/docs/generate_site_pages.py --write`; the docs job fails while the gallery
disagrees with the notebooks.

The rules a runbook's result is held to (purged splits, trial registries, controls with a known
answer) are on the [Governance](/project/governance/) page.

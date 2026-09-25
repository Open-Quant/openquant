# Python Research Notebooks

Notebook starter pack for the OpenQuant mid-frequency research flywheel.

## Notebooks

1. `01_event_labeling_and_pipeline.ipynb`
2. `02_purged_cv_and_seq_bootstrap.ipynb`
3. `03_feature_diagnostics.ipynb`
4. `04_portfolio_construction.ipynb`
5. `05_risk_overlays_and_reality_check.ipynb`
6. `06_afml_real_data_end_to_end.ipynb` (full flywheel analysis on the SYNTHETIC `fetch` sample; pass your own `source=` for real data)
7. `07_feature_engineering_discovery_loop.ipynb` (candidate feature generation + screening)
8. `08_algo_wheel_experiments.ipynb` (config-wheel experiment ranking)

Every notebook runs offline. Notebook 06 reads market-shaped data through `openquant.data.fetch`, whose default
source is the committed SYNTHETIC sample (`SYN_A` to `SYN_E`, see `DATA_SOURCES.md`); the others generate
synthetic series in memory. None of the committed outputs describe a real market.

## Run setup

```bash
uv venv --python 3.13 .venv
uv sync --group dev
uv run --python .venv/bin/python maturin develop --manifest-path crates/pyopenquant/Cargo.toml
```

## Execute the notebooks (`just notebooks-run`)

```bash
just notebooks-run              # every NN_*.ipynb, in place, in a Jupyter kernel
just notebooks-run --only 06    # one notebook
just notebooks-run --check      # execute to a temp dir; fail if the committed outputs are stale
```

`scripts/run_notebooks.py` executes each notebook with nbclient + ipykernel, so tables, printed output and
figures are all saved in the `.ipynb`. Any cell error fails the run (exit status 1) after the other notebooks
have run. Outputs are normalised so an unchanged notebook re-runs to an unchanged file: no execution
timestamps, no Python patch version, positional cell ids, seeded data. A notebook can be excluded only by
listing it in `EXCLUDED` in `run_notebooks.py` with a reason; none is.

Figures are drawn with `nbfigures.figure(name, draw)`, which shows the plot inline and writes
`docs-site/public/figures/notebooks/<name>-light.svg` and `-dark.svg` in the docs-site identity colours, for
pages to embed with the `light:sl-hidden` / `dark:sl-hidden` pair. A full run rewrites that directory.

CI (`.github/workflows/notebooks.yml`) runs `just notebooks-run` on pull requests that touch notebooks,
Python or crates, and nightly, uploads the executed notebooks and figures as an artifact, and then runs
`just notebooks-verify`, which fails if the committed outputs differ from the fresh ones by more than float
noise (4 significant digits) and PNG bytes. **If you change a notebook or anything it calls, run
`just notebooks-run` and commit the notebooks and figures.**

One notebook, with the lower-level script:

```bash
uv run --python .venv/bin/python notebooks/python/scripts/execute_notebook_cells.py \
  notebooks/python/08_algo_wheel_experiments.ipynb \
  --out notebooks/python/_executed/08_algo_wheel_experiments.ipynb
```

## Smoke run (CI-friendly)

```bash
uv run --python .venv/bin/python python notebooks/python/scripts/smoke_all.py
```

The smoke script mirrors core notebook logic with deterministic synthetic futures data.

## Bar diagnostics (AFML Ch.2)

Compare time/tick/volume/dollar bar families with simple serial-dependence and heteroskedasticity proxies:

```bash
uv run --python .venv/bin/python python notebooks/python/scripts/bar_diagnostics.py
```

The script prints:
- `lag1_return_autocorr` (serial dependence proxy)
- `lag1_sq_return_autocorr` (heteroskedasticity proxy)
- `return_std` and number of bars per family

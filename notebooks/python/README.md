# Python Research Notebooks

Notebook starter pack for the OpenQuant mid-frequency research flywheel.

## Notebooks

Every notebook follows the notebook contract (docs site: Workflows > Notebook contract,
`docs-site/src/content/docs/workflows/research-notebook-contract.md`), checked by
`just notebooks-lint`. A notebook is either an **API tour** (title `# API tour: ...`: shows how a
few calls fit together on small synthetic inputs and claims nothing) or a **runbook** (title
`# Runbook: ...`: a pre-registered hypothesis, controls, a trial registry and a promotion
decision). Both end with an `nbrepro.footer(...)` reproducibility cell.

API tours:

- `01_event_labeling_and_pipeline.ipynb` (`pipeline.run_mid_frequency_pipeline_frames`, CUSUM events)
- `02_purged_cv_and_seq_bootstrap.ipynb` (indicator matrix, average uniqueness, sequential bootstrap)
- `03_feature_diagnostics.ipynb` (`viz` payloads for feature importance and regimes)
- `04_portfolio_construction.ipynb` (inverse-variance, minimum-volatility and maximum-Sharpe weights)
- `05_risk_overlays_and_reality_check.ipynb` (`research.run_flywheel_iteration`, drawdown payload)
- `07_feature_engineering_discovery_loop.ipynb` (`feature_diagnostics.feature_screen_report`)
- `08_algo_wheel_experiments.ipynb` (`research.run_flywheel_grid`)

Runbooks:

- `09_fracdiff_stationarity_memory.ipynb` (runbook #49: FFD d-sweep, ADF vs memory, checked against simulated series of known memory; SYNTHETIC by default, `OPENQUANT_RUNBOOK_SOURCE` for your own file)
- `10_hrp_vs_ivp_cla_oos.ipynb` (runbook, #51: HRP vs inverse-variance and CLA out of sample, AFML §16.6 Monte Carlo on SYNTHETIC data; `OPENQUANT_RUNBOOK_RUNS=10000` for the book's run count)
- `11_meta_labeling_triple_barrier.ipynb` (runbook, #47: CUSUM events, triple-barrier meta-labels and a meta-model in purged k-fold, primary vs meta on precision, F1 and deflated Sharpe net of costs; SYNTHETIC paths with a planted signal plus the `fetch` sample as a no-signal control; `OPENQUANT_RUNBOOK_SOURCE` for your own file)
- `12_cpcv_deflated_sharpe.ipynb` (runbook, #48: CPCV backtest with PSR and the deflated Sharpe ratio; a no-signal control and a planted signal, SYNTHETIC; `OPENQUANT_RUNBOOK_SOURCE` for your own file)
- `13_bet_sizing_from_probabilities.ipynb` (runbook, #50: pre-registered test of runbook 11's post hoc finding; Snippet 10.1-10.3 probability sizing vs a flat 0.5 meta filter, with the number of classes, step size and averaging as registered trials, net of costs, deflated by a trial registry; SYNTHETIC planted-signal paths plus no-signal controls; `OPENQUANT_RUNBOOK_SOURCE` for your own file)

Notebook 06 (a momentum heuristic with made-up probabilities, no cross-validation and no deflated
Sharpe ratio) was removed in #47; notebook 11 replaces it.

Every notebook runs offline. Notebooks 09 to 13 read market-shaped data through `openquant.data.fetch`, whose default
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
just notebooks-run --only 11    # one notebook
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

## Check the notebook contract (`just notebooks-lint`)

```bash
just notebooks-lint                                   # every NN_*.ipynb
just notebooks-lint notebooks/python/11_meta_labeling_triple_barrier.ipynb
```

`scripts/lint_notebooks.py` (standard library only) checks the title, the section headings, the
`nbrepro.footer(...)` last cell and its committed output, and that no committed output holds an
error, a machine path or a package version on stdout. CI runs it in the `python-lint` job.

The footer (`nbrepro.py`) prints the data hash and the seed on stdout and the git commit and
package versions on stderr, which `just notebooks-verify` ignores, so a dependency bump does not
make a notebook stale. Under `just notebooks-run` it prints that the containing commit pins them,
so re-running does not rewrite every footer; run a notebook any other way and it prints the real
commit and versions.

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

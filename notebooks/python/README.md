# Python Research Notebooks

Notebook starter pack for the OpenQuant mid-frequency research flywheel.

## Notebooks

Every notebook follows the [notebook contract](../../docs-site/src/content/docs/workflows/research-notebook-contract.md), checked by
`just notebooks-lint`. A notebook is either an **API tour** (title `# API tour: ...`: shows how a
few calls fit together on small synthetic inputs without making investment claims) or a
**runbook** (title `# Runbook: ...`: a pre-registered hypothesis, controls, a trial inventory
and a promotion decision). Return-based searches use a persistent trial registry; feature
and variance experiments can use an explicit inventory instead. Both end with an `nbrepro.footer(...)` reproducibility cell.

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
- `14_fracdiff_features_classifier.ipynb` (runbook, #209: runbook 09's open question; FFD features with d chosen per training fold vs returns, integer differences and the level as inputs of a purged-CV classifier, net of costs, every feature set a registered trial; SYNTHETIC planted-signal paths plus no-signal controls, with a post hoc walk-forward check; `OPENQUANT_RUNBOOK_SOURCE` for your own file)

Notebook 06 (a momentum heuristic with made-up probabilities, no cross-validation and no deflated
Sharpe ratio) was removed in #47; notebook 11 replaces it.

Every notebook runs with local data under its committed defaults. Notebooks 09 to 14 read market-shaped data through `openquant.data.fetch`, whose default
source is the committed SYNTHETIC sample (`SYN_A` to `SYN_E`, see `DATA_SOURCES.md`); the others generate
synthetic series in memory. None of the committed outputs describe a real market.

## Run setup

```bash
uv venv --python 3.13 .venv
uv sync --locked --group dev --no-install-project
uv run --no-sync --python .venv/bin/python maturin develop --release --manifest-path crates/pyopenquant/Cargo.toml
```

The runner starts a kernel named `python3`. Check its environment and imported
`openquant._core.__file__` if results suggest a different installation, and restart kernels
after rebuilding the extension. See the [notebook research workflow](../../docs-site/src/content/docs/workflows/notebook-research-workflow.md)
for interpreter and provenance checks.

## Execute the notebooks (`just notebooks-run`)

```bash
just notebooks-run              # every NN_*.ipynb, in place, in a Jupyter kernel
just notebooks-run --only 11    # matching number prefix or stem
just notebooks-run --check      # execute to a temp dir; fail if the committed outputs are stale
```

`scripts/run_notebooks.py` executes each notebook with nbclient + ipykernel, so tables, printed output and
figures are all saved in the `.ipynb`. Any cell error fails the run (exit status 1) after the other notebooks
have run. Outputs are normalised so an unchanged notebook re-runs to an unchanged file: no execution
timestamps, no Python patch version, positional cell ids, seeded data. This normalization
does not guarantee identical numerical results across every platform or dependency version. A notebook can be excluded only by
listing it in `EXCLUDED` in `run_notebooks.py` with a reason; none is.

Figures are drawn with `nbfigures.figure(name, draw)`, which shows the plot inline and writes
`docs-site/public/figures/notebooks/<name>-light.svg` and `-dark.svg` in the docs-site identity colours, for
pages to embed with the `light:sl-hidden` / `dark:sl-hidden` pair. A full default run removes existing SVGs from that directory before regeneration; a failed
run can therefore leave incomplete figure output.
`--check` directs executed notebooks and figures to temporary locations; it is not a sandbox
for data caches, registries or other file writes performed by notebook code. Kernels inherit
environment variables, so inspect source/registry overrides before running. A selection
with `--only` does not perform the full figure-directory comparison.

The configured [notebook workflow](../../.github/workflows/notebooks.yml) runs
`just notebooks-run` for matching pull-request/main-push paths, nightly and on manual dispatch.
It uploads executed notebooks and figures, then runs `just notebooks-verify`. The latter
compares files already on disk; it does not execute notebooks. Comparison rounds decimal numbers in output text
to four significant digits and ignores stderr, raster-image bytes, execution counts, cell IDs
and metadata. Full `--check` or `notebooks-verify` comparisons also check exported SVGs after normalization. Passing comparison
therefore does not certify every metadata field or rendered pixel. **If you change a notebook or anything it calls, run
`just notebooks-run` and commit the notebooks and figures.**

One notebook, with the lower-level script:

```bash
uv run --no-sync --python .venv/bin/python notebooks/python/scripts/execute_notebook_cells.py \
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
package versions on stderr, which `just notebooks-verify` ignores, so a dependency-version string alone does not
make a notebook stale; changed numerical output still can. Under `just notebooks-run` it prints that the containing commit pins them,
so re-running does not rewrite every footer; direct runs print the real
commit and versions unless `OPENQUANT_NOTEBOOK_PINNED=1` is already set. The pinned-footer marker does not verify a clean working tree or
that the loaded native extension was built from that commit.

## Smoke run (CI-friendly)

```bash
uv run --no-sync --python .venv/bin/python python notebooks/python/scripts/smoke_all.py
```

The smoke script exercises a subset of core component calls with synthetic data. It
does not execute the notebook files, verify their saved outputs, or fit and test the
runbooks' research hypotheses. Use the notebook runner for those checks.

## Bar diagnostics (AFML Ch.2)

Compare time/tick/volume/dollar bar families with simple serial-dependence and heteroskedasticity proxies:

```bash
uv run --no-sync --python .venv/bin/python python notebooks/python/scripts/bar_diagnostics.py
```

The script prints:
- `lag1_return_autocorr` (serial dependence proxy)
- `lag1_sq_return_autocorr` (heteroskedasticity proxy)
- `return_std` and number of bars per family

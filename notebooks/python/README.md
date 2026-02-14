# Python Research Notebooks

Notebook starter pack for the OpenQuant mid-frequency research flywheel.

## Notebooks

1. `01_event_labeling_and_pipeline.ipynb`
2. `02_purged_cv_and_seq_bootstrap.ipynb`
3. `03_feature_diagnostics.ipynb`
4. `04_portfolio_construction.ipynb`
5. `05_risk_overlays_and_reality_check.ipynb`
6. `06_afml_real_data_end_to_end.ipynb` (online ticker data + full flywheel analysis)

## Run setup

```bash
uv venv --python 3.13 .venv
uv sync --group dev
uv run --python .venv/bin/python maturin develop --manifest-path crates/pyopenquant/Cargo.toml
```

## Smoke run (CI-friendly)

```bash
uv run --python .venv/bin/python python notebooks/python/scripts/smoke_all.py
```

The smoke scripts mirror core notebook logic with deterministic synthetic futures data.

Execute the real-data notebook cells non-interactively:

```bash
uv run --python .venv/bin/python notebooks/python/scripts/execute_notebook_cells.py notebooks/python/06_afml_real_data_end_to_end.ipynb
```

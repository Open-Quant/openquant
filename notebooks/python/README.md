# Python Research Notebooks

Notebook starter pack for the OpenQuant mid-frequency research flywheel.

## Notebooks

1. `01_event_labeling_and_pipeline.ipynb`
2. `02_purged_cv_and_seq_bootstrap.ipynb`
3. `03_feature_diagnostics.ipynb`
4. `04_portfolio_construction.ipynb`
5. `05_risk_overlays_and_reality_check.ipynb`
6. `06_afml_real_data_end_to_end.ipynb` (online ticker data + full flywheel analysis)
7. `07_ch2_event_sampling_filters.ipynb`
8. `08_ch3_labeling_signal_scaffolding.ipynb`
9. `09_ch4_sampling_uniqueness_bootstrap.ipynb`
10. `10_ch5_fracdiff_stationarity_memory.ipynb`
11. `11_ch7_validation_leakage_protocol.ipynb`
12. `12_ch8_feature_importance_diagnostics.ipynb`
13. `13_ch10_bet_sizing_mechanics.ipynb`
14. `14_ch14_risk_reality_checks.ipynb`
15. `15_ch16_portfolio_construction_allocation.ipynb`
16. `16_ch17_structural_break_proxy.ipynb`
17. `17_ch18_microstructure_proxy_features.ipynb`
18. `18_ch19_codependence_and_regimes.ipynb`

## Run setup

```bash
uv venv --python 3.11 .venv
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

Run the full AFML chapter notebook suite:

```bash
uv run --python .venv/bin/python notebooks/python/scripts/run_all_chapter_notebooks.py
```

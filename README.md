<p align="center">
  <img src="assets/banner_v3.svg" alt="OpenQuant-rs" width="100%" />
</p>

<h1 align="center">openquant-rs</h1>

<p align="center">
  <strong>Rust-native quantitative finance toolkit for research and production workflows.</strong>
</p>

<p align="center">
  <a href="https://github.com/Open-Quant/openquant/actions/workflows/ci.yml">CI</a>
  ·
  <a href="https://github.com/Open-Quant/openquant/actions/workflows/benchmark-regression.yml">Benchmark Regression</a>
  ·
  <a href="https://github.com/Open-Quant/openquant/actions/workflows/release.yml">Release Readiness</a>
</p>

## Status
- Production baseline package: benchmarks + regression checks + release workflow are in place.
- High-coverage module-level tests, benchmark tracking, and release gates are active.

Detailed status: `docs/project_status.md`

## Quick Start
```bash
# Fast validation (default CI path)
cargo test --workspace --lib --tests --all-features -- --skip test_sadf_test

# Long-running SADF hotspot (explicit)
cargo test -p openquant --test structural_breaks test_sadf_test -- --ignored

# Benchmarks
cargo bench -p openquant --bench perf_hotspots --bench synthetic_ticker_pipeline

# Collect + check benchmark thresholds
python3 scripts/collect_bench_results.py --criterion-dir target/criterion --out benchmarks/latest_benchmarks.json --allow-list benchmarks/benchmark_manifest.json
python3 scripts/check_bench_thresholds.py --baseline benchmarks/baseline_benchmarks.json --latest benchmarks/latest_benchmarks.json --max-regression-pct 25
```

## Research Flywheel (Python + Rust)
```bash
# Python env + bindings
uv venv --python 3.11 .venv
uv sync --group dev
uv run --python .venv/bin/python maturin develop --manifest-path crates/pyopenquant/Cargo.toml

# Notebook logic smoke + reproducible experiment run
uv run --python .venv/bin/python python notebooks/python/scripts/smoke_all.py
uv run --python .venv/bin/python python experiments/run_pipeline.py --config experiments/configs/futures_oil_baseline.toml --out experiments/artifacts

# Rust notebook-companion smoke
cargo run -p openquant --example research_notebook_smoke
```

## Crate Layout
- `crates/openquant/src/`: core library modules
- `crates/openquant/tests/`: Rust test suite
- `crates/openquant/benches/`: criterion benchmarks
- `tests/fixtures/`: shared fixtures
- `benchmarks/`: baseline + latest benchmark snapshots
- `notebooks/`: Python notebooks + Rust Evcxr companions
- `experiments/`: config-driven experiment runner + artifacts

## Publish Readiness
- Publishing checklist: `docs/publishing.md`
- Stabilization + productionization checklist: `docs/stabilization_productionization.md`
- Latest benchmark report: `docs/benchmark_snapshot.md`
- Python bindings quickstart + API map: `docs/python_bindings.md`
- Notebook-first workflow + promotion checklist: `docs/research_workflow.md`

## Astro Docs Site (GitHub Pages)
A modern docs site scaffold is included under `docs-site/`.

```bash
cd docs-site
npm install
npm run dev
```

Build output is published by GitHub Actions workflow: `.github/workflows/docs-pages.yml`.

## License
MIT (`LICENSE`)

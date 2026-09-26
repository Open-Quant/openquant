set shell := ["bash", "-cu"]

# `help` is the first recipe, so a bare `just` lists the recipes.

help:
    @just --list

fmt:
    cargo fmt

fmt-check:
    cargo fmt --all -- --check

clippy:
    cargo clippy --workspace --all-targets --all-features -- -D warnings

check:
    cargo check --all-targets --all-features

test:
    cargo test --all-targets --all-features

test-fast:
    cargo test --workspace --lib --tests --all-features -- --skip test_sadf_test

test-slow:
    cargo test -p openquant --test structural_breaks test_sadf_test -- --ignored

# What nightly-validation.yml runs: everything, including ignored tests, except
# tests ignored with a "FINDING:" reason (known, expected failures).
test-nightly:
    cargo test --workspace --all-features --no-fail-fast -- --include-ignored $(python3 scripts/ci/finding_skip_args.py)

lint: fmt-check clippy

bench:
    cargo bench --all-features

bench-hotspots:
    cargo bench -p openquant --bench perf_hotspots

bench-synthetic:
    cargo bench -p openquant --bench synthetic_ticker_pipeline

bench-compile:
    cargo bench -p openquant --no-run

bench-all:
    cargo bench -p openquant --bench perf_hotspots --bench synthetic_ticker_pipeline

bench-collect:
    python3 scripts/collect_bench_results.py --criterion-dir target/criterion --out benchmarks/latest_benchmarks.json --allow-list benchmarks/benchmark_manifest.json

# Compares against the committed reference numbers, which were measured on one
# particular machine: only meaningful on comparable hardware. CI instead times
# the PR base and head on the same runner (benchmark-regression.yml).
bench-check:
    python3 scripts/check_bench_thresholds.py --baseline benchmarks/baseline_benchmarks.json --latest benchmarks/latest_benchmarks.json --max-regression-pct 35 --overrides benchmarks/threshold_overrides.json

py-develop:
    uv run --python .venv/bin/python maturin develop --manifest-path crates/pyopenquant/Cargo.toml

# Optimised build; use it for timing and for heavy calls such as structural_breaks.get_sadf.
py-develop-release:
    uv run --python .venv/bin/python maturin develop --release --manifest-path crates/pyopenquant/Cargo.toml

py-build:
    uv run --python .venv/bin/python maturin build --manifest-path crates/pyopenquant/Cargo.toml --out dist

py-import-smoke:
    uv run --python .venv/bin/python python -c "import openquant; print('openquant bindings OK')"

py-test:
    uv run --python .venv/bin/python pytest python/tests -q

py-lint:
    uv run --python .venv/bin/python ruff check python/
    uv run --python .venv/bin/python ruff format --check python/
    uv run --python .venv/bin/python mypy

# Regenerate the API reference artefacts after changing a binding: the .pyi stubs (from the
# Rust source), the rustdoc/API inventory, and the docs site's Python reference (from the
# built extension, so run py-develop first).
py-api-docs:
    uv run --python .venv/bin/python python scripts/generate_python_stubs.py
    uv run --python .venv/bin/python python scripts/generate_api_inventory.py
    uv run --python .venv/bin/python python scripts/generate_python_api_reference.py

# Check the stubs against the built extension (names, parameters, defaults).
py-stubtest:
    uv run --python .venv/bin/python python -m mypy.stubtest openquant._core

py-setup:
    uv venv --python 3.13 .venv
    uv sync --group dev

py-bench:
    uv run --python .venv/bin/python python python/benchmarks/benchmark_pipeline.py --iterations 30 --bars 2048

py-bench-data:
    uv run --python .venv/bin/python python python/benchmarks/benchmark_data_processing.py --rows-per-symbol 200000 --symbols 4 --iterations 7 --out benchmarks/data_processing/latest.json

exp-run:
    uv run --python .venv/bin/python python experiments/run_pipeline.py --config experiments/configs/futures_oil_baseline.toml --out experiments/artifacts

notebook-smoke:
    uv run --python .venv/bin/python python notebooks/python/scripts/smoke_all.py

# Execute every notebooks/python/NN_*.ipynb in a Jupyter kernel (nbclient), in
# place, and re-export docs-site/public/figures/notebooks/. Fails on any cell
# error. Needs the extension built first (`just py-develop-release` runs them far
# faster than `just py-develop`). Extra arguments go to the runner, e.g.
# `just notebooks-run --only 11`, `--check`, or `--jobs 0` (one notebook per CPU
# at a time, as CI runs them).
notebooks-run *args:
    uv run --no-sync --python .venv/bin/python python notebooks/python/scripts/run_notebooks.py {{args}}

# Fails if the working-tree notebooks/figures differ from the committed ones
# beyond float noise and image bytes; CI runs it after `just notebooks-run`.
notebooks-verify ref="HEAD":
    uv run --no-sync --python .venv/bin/python python notebooks/python/scripts/run_notebooks.py --against-git {{ref}}

research-smoke: py-develop notebook-smoke exp-run

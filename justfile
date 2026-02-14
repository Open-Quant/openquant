set shell := ["bash", "-cu"]

default := help

help:
    @just --list

fmt:
    cargo fmt

fmt-check:
    cargo fmt -- --check

clippy:
    cargo clippy --all-targets --all-features -- -D clippy::correctness -D clippy::suspicious

check:
    cargo check --all-targets --all-features

test:
    cargo test --all-targets --all-features

test-fast:
    cargo test --workspace --lib --tests --all-features -- --skip test_sadf_test

test-slow:
    cargo test -p openquant --test structural_breaks test_sadf_test -- --ignored

lint: fmt-check clippy

bench:
    cargo bench --all-features

bench-hotspots:
    cargo bench -p openquant --bench perf_hotspots

bench-synthetic:
    cargo bench -p openquant --bench synthetic_ticker_pipeline

bench-all:
    cargo bench -p openquant --bench perf_hotspots --bench synthetic_ticker_pipeline

bench-collect:
    python3 scripts/collect_bench_results.py --criterion-dir target/criterion --out benchmarks/latest_benchmarks.json --allow-list benchmarks/benchmark_manifest.json

bench-check:
    python3 scripts/check_bench_thresholds.py --baseline benchmarks/baseline_benchmarks.json --latest benchmarks/latest_benchmarks.json --max-regression-pct 35 --overrides benchmarks/threshold_overrides.json

py-develop:
    uv run --python .venv/bin/python maturin develop --manifest-path crates/pyopenquant/Cargo.toml

py-build:
    uv run --python .venv/bin/python maturin build --manifest-path crates/pyopenquant/Cargo.toml --out dist

py-import-smoke:
    uv run --python .venv/bin/python python -c "import openquant; print('openquant bindings OK')"

py-test:
    uv run --python .venv/bin/python pytest python/tests -q

py-setup:
    uv venv --python 3.13 .venv
    uv sync --group dev

py-bench:
    uv run --python .venv/bin/python python python/benchmarks/benchmark_pipeline.py --iterations 30 --bars 2048

exp-run:
    uv run --python .venv/bin/python python experiments/run_pipeline.py --config experiments/configs/futures_oil_baseline.toml --out experiments/artifacts

notebook-smoke:
    uv run --python .venv/bin/python python notebooks/python/scripts/smoke_all.py

research-smoke: py-develop notebook-smoke exp-run

docs-loop-init:
    skills/afml-docs-loop/scripts/run_afml_docs_loop.sh init

docs-loop-status:
    skills/afml-docs-loop/scripts/run_afml_docs_loop.sh status

docs-loop-next:
    skills/afml-docs-loop/scripts/run_afml_docs_loop.sh next --print-prompt

docs-loop-export:
    skills/afml-docs-loop/scripts/run_afml_docs_loop.sh export

docs-loop-evidence:
    skills/afml-docs-loop/scripts/run_afml_docs_loop.sh evidence

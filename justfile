set shell := ["bash", "-cu"]

default := help

help:
    @just --list

fmt:
    cargo fmt

fmt-check:
    cargo fmt -- --check

clippy:
    cargo clippy --all-targets --all-features -- -D warnings

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
    python3 scripts/check_bench_thresholds.py --baseline benchmarks/baseline_benchmarks.json --latest benchmarks/latest_benchmarks.json --max-regression-pct 25

# Publishing Guide

## Preconditions
- Clean working tree
- Rust stable toolchain
- Crates.io account + `cargo login`

## Local release checklist
1. `cargo fmt -- --check`
2. `cargo clippy --all-targets --all-features -- -D warnings`
3. `cargo test --workspace --lib --tests --all-features -- --skip test_sadf_test`
4. `cargo test -p openquant --test structural_breaks test_sadf_test -- --ignored`
5. `cargo bench -p openquant --bench perf_hotspots --bench synthetic_ticker_pipeline -- --sample-size 10 --warm-up-time 1 --measurement-time 1`
6. `python3 scripts/collect_bench_results.py --criterion-dir target/criterion --out benchmarks/latest_benchmarks.json --allow-list benchmarks/benchmark_manifest.json`
7. `python3 scripts/check_bench_thresholds.py --baseline benchmarks/baseline_benchmarks.json --latest benchmarks/latest_benchmarks.json --max-regression-pct 25`
8. `cargo package -p openquant`

## Tag + publish
1. Update `crates/openquant/Cargo.toml` version.
2. `git tag vX.Y.Z && git push origin vX.Y.Z`
3. `cargo publish -p openquant`

## Post-release
- Copy `benchmarks/latest_benchmarks.json` to `benchmarks/baseline_benchmarks.json` for next cycle.
- Update docs examples/changelog and GitHub Pages docs if API changed.

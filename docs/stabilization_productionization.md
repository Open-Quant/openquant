# Stabilization + Productionization Plan

## Current Stage
- Migration parity: complete (crosswalk fully ported).
- Stabilization: mostly complete.
- Productionization: active.

## Validation/Stabilization Gates
- Fast regression suite (default):
  - `just test-fast`
- Long-running SADF hotspot (explicit):
  - `just test-slow`
- Migration sync integrity:
  - `python /Users/seankoval/.codex/skills/migration-sync/scripts/check_migration_sync.py --roadmap ROADMAP.md --crosswalk openquant-rs/tests/crosswalk.md`

## Performance Gates
- Hotspot benchmarks:
  - `just bench-hotspots`
- Synthetic ticker performance showcase:
  - `just bench-synthetic`
- Collect benchmark outputs:
  - `just bench-collect`
- Enforce regression threshold vs baseline:
  - `just bench-check`
- Benchmark manifest (tracked IDs):
  - `benchmarks/benchmark_manifest.json`
- Baseline file:
  - `benchmarks/baseline_benchmarks.json`
- Initial benchmark targets:
  - `structural_breaks/get_sadf_sm_power`
  - `bet_sizing/bet_size_reserve_fit`
  - `bet_sizing/bet_size_reserve_reuse_fit`
  - `synthetic_ticker/ewma_100k`
  - `synthetic_ticker/risk_metrics_var_es_cdar`
  - `synthetic_ticker/seq_bootstrap_2k_600`
  - `synthetic_ticker/pipeline_end_to_end`

## Release Readiness Flow
1. Run local quality gates:
   - `just lint`
   - `just test-fast`
2. Run long test gate before cutting release:
   - `just test-slow`
3. Verify package readiness:
   - `cargo package -p openquant --allow-dirty`
4. Cut tag:
   - `git tag vX.Y.Z && git push origin vX.Y.Z`

## CI/Automation
- `CI` workflow runs format, clippy, and fast tests on push/PR.
- `Nightly Validation` workflow runs the long SADF test and benchmark compile check.
- `Release Readiness` workflow runs on tags (`v*`) and validates lint/test/package/bench compile.
- `Benchmark Regression` workflow runs benchmark suites, collects criterion output, and fails if runtime regressions exceed threshold.

## Known Hotspot
- `test_sadf_test` is intentionally marked `#[ignore]` to keep default CI latency bounded.
- It remains required for periodic/nightly and pre-release validation.

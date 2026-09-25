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
  - Module-to-test parity is recorded in `tests/crosswalk.md`. The script that once checked it
    lived outside this repository and `ROADMAP.md` was never part of it, so there is no
    runnable gate for this today.

## Performance Gates
- Hotspot benchmarks:
  - `just bench-hotspots`
- Synthetic ticker performance showcase:
  - `just bench-synthetic`
- Collect benchmark outputs:
  - `just bench-collect`
- Enforce regression threshold vs the committed baseline (same-machine only; see below):
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
What each workflow enforces. "Every PR" jobs have no path filter, so any of them can be made a
required status check; the other rows cannot (a path-filtered or scheduled check never reports on
some PRs).

| Workflow / job | When | Enforces |
| --- | --- | --- |
| `CI` / `lint-test` | every PR, push to `main` | `cargo fmt --all --check`; `cargo clippy --workspace --all-targets --all-features -D warnings`; `cargo test --workspace --lib --tests --all-features` (skips `test_sadf_test`, see below) |
| `CI` / `bench-compile` | every PR, push to `main` | `cargo bench -p openquant --no-run`: all four benches compile |
| `CI` / `python (3.11)`, `python (3.13)` | every PR, push to `main` | `uv.lock` is current; the extension builds; `pytest python/tests`. The 3.13 leg also runs the notebook smoke, the experiment scaffold smoke and the `research_notebook_smoke` Rust example |
| `CI` / `python-lint` | every PR, push to `main` | `ruff check python/`, `ruff format --check python/`, `mypy` (`python/openquant`, `python/tests`; config in `pyproject.toml`) |
| `CI` / `docs-checks` | every PR, push to `main` | docs build, links, content schema, API drift, contrast, coverage page, Rust examples compile, Python examples run |
| `Benchmark Regression` | PRs touching `crates/openquant`, `Cargo.*`, the toolchain, or the bench scripts/config | head vs merge base, both timed on the same runner; fails above 35% (per-bench overrides in `benchmarks/threshold_overrides.json`) or if a benchmark stops reporting |
| `Nightly Validation` | daily 07:00 UTC, manual | full Rust suite incl. doc tests and `#[ignore]`d tests (hour-long `test_sadf_test`), except tests ignored with a `FINDING:` reason; core crate tests on macOS and Windows; Python suite + smokes on 3.11 and 3.13; Rust (`cargo-llvm-cov`) and Python (`pytest-cov`) line coverage in the job summary. Coverage is reported, not gated |
| `Release Readiness` | tags `v*`, manual | lint + fast tests, `cargo package -p openquant`, all benches compile, and `test_sadf_test` |
| `Docs Pages` | push to `main` touching docs | build, links, schema, API drift; deploys the site |
| `Verify` | manual only | ai-dlc project checks (disabled on PRs until ai-dlc publishes a release manifest) |

Tests ignored with `#[ignore = "FINDING: ..."]` are known divergences from the AFML reference,
recorded as failing tests; they are expected to fail and run nowhere until fixed. The list is
printed by `scripts/ci/finding_skip_args.py` in every nightly log.

Local equivalents: `just lint`, `just test-fast`, `just bench-compile`, `just py-test`,
`just py-lint`, `just test-nightly`, `just test-slow`.

`just bench-check` compares against `benchmarks/baseline_benchmarks.json`, numbers measured on one
machine; it is only meaningful on comparable hardware. CI does not use that file.

## Known Hotspot
- `test_sadf_test` is intentionally marked `#[ignore]` to keep default CI latency bounded
  (about an hour in a debug build).
- It runs in `Nightly Validation` and in `Release Readiness` on every tag.

# Project Status

## Migration status
- Python-to-Rust module test parity is complete for the tracked mlfinlab suite.
- Source of truth: `openquant-rs/tests/crosswalk.md`.
- Roadmap remaining list is empty.

## Do we still need mlfinlab repo locally?
Short answer: not required for day-to-day OpenQuant usage, still useful for maintenance.

Keep mlfinlab if you want to:
- add new parity modules/tests in the future,
- re-generate fixtures from Python behavior,
- validate behavior drift against upstream changes.

You can archive/remove mlfinlab locally if you are:
- focused only on OpenQuant runtime/library usage,
- not planning additional parity backports,
- comfortable relying on existing fixtures + Rust tests only.

## Functional status
OpenQuant is functional for the migrated tracked package scope, with:
- passing fast/full Rust test sweeps (except intentionally isolated long SADF run in default fast path),
- benchmark baselines and regression checks,
- release-readiness CI workflow.

## Known caveats
- `test_sadf_test` is intentionally excluded from fast CI and run in dedicated slow/nightly path.
- Performance thresholds are now wired but should be tightened over time as variance stabilizes.

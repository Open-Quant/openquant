<!--
Title: a conventional-commit prefix (fix, feat, test, docs, refactor, chore, ci, design)
and the effect of the change, e.g.
  fix: weekly/monthly resampling scrambled the price matrix in four modules (#93)
See CONTRIBUTING.md for the full conventions. Delete sections that do not apply.
-->

Closes #

## What was wrong

<!-- The defect or gap, with the evidence. For features: what was missing and why it is needed. -->

## What changed

<!-- The fix or feature. Name any breaking change to a Rust or Python API. -->

## Tests

<!--
Bug fixes: the regression test, and confirmation that it FAILS on the old code
(e.g. "checked by reverting src/ and re-running").
New expected values: the independent reference they come from (AFML snippet, paper,
numpy/scipy/pandas/scikit-learn) and the generator under tests/fixtures/, if any.
-->

## Verification

<!-- The exact commands you ran and their results. -->

- [ ] `cargo fmt -- --check`
- [ ] `cargo clippy --workspace --all-targets --all-features -- -D warnings`
- [ ] `cargo test --workspace --lib --tests --all-features -- --skip test_sadf_test`
- [ ] `pytest python/tests` after `maturin develop` (if bindings or Python code changed)
- [ ] `bun run check:docs` in `docs-site/` (if docs or public APIs changed)
- [ ] Benchmarks checked against the baseline (if a hot path changed)

## Behaviour change

<!-- What a user calling the library will see differently. "None" is a valid answer. -->

## Not done / not verified

<!-- Anything left out, assumptions not checked, defects found but not fixed (with issue numbers). -->

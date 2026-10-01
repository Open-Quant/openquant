# Publishing Guide

A release publishes three things, all from `.github/workflows/release.yml` when a `v*` tag
is pushed:

| What | Where | How |
|---|---|---|
| The Python package (import name `openquant`) | PyPI, under the distribution name in `pyproject.toml` | five abi3 wheels + an sdist, trusted publishing from the `pypi` environment |
| The `openquant` crate | crates.io | `cargo publish -p openquant`, `CARGO_REGISTRY_TOKEN` secret |
| Release notes and the built files | a GitHub Release for the tag | notes extracted from `CHANGELOG.md` |

`pyopenquant` (the extension crate) has `publish = false`: it ships only inside the wheels.

## The distribution name

The PyPI distribution name is not decided yet (#58). It lives in exactly one place,
`[project] name` in `pyproject.toml` (currently the placeholder `pyopenquant`). The workflow
reads it from there and never spells it out, so choosing the name is a one-line change to
`pyproject.toml` plus the PyPI trusted-publisher entry below. The import name is
`openquant` whatever the distribution is called.

## What the workflow does

| Job | Tag push | Dispatch (dry run, the default) | Pull request touching packaging |
|---|---|---|---|
| `metadata`: name, version, pinned toolchain; the three versions agree; the tag is `v<version>` | yes | yes | yes |
| `release-notes`: the version's `CHANGELOG.md` section | required | falls back to Unreleased | falls back to Unreleased |
| `release-check`: fmt, clippy, fast tests, bench compile | yes | yes | no (ci.yml runs them) |
| `release-slow-tests`: the long SADF test | yes | yes | no |
| `crate-package`: `cargo package --list`, `cargo publish --dry-run` | yes | yes | yes |
| `wheels`: Linux x86_64 + aarch64 (manylinux_2_28), macOS x86_64 + arm64, Windows x86_64 | yes | yes | yes |
| `smoke`: each wheel in clean venvs on Python 3.11, 3.12, 3.13, running the Quickstart | yes | yes | yes |
| `sdist`: build it, build a wheel from it alone, smoke-test that | yes | yes | yes |
| `publish-pypi`, `publish-crate`, `github-release` | yes | no | no |

The pull-request run triggers on changes to `release.yml`, `scripts/release/`,
`pyproject.toml`, `Cargo.toml`, `Cargo.lock`, the two crate manifests and
`rust-toolchain.toml`. A dispatch with `dry_run=false` publishes, and is refused unless a
`v*` tag is selected as the ref; use it only to re-run a release whose tag push failed
before publishing anything.

**Wheels are abi3.** `[tool.maturin] features = ["pyo3/abi3-py311"]` in `pyproject.toml`
builds one stable-ABI wheel per platform (`cp311-abi3`), which installs on CPython 3.11 and
every later version. PyO3 0.29 and pyo3-polars 0.28 use only the limited API. The feature is
set for maturin only, so `cargo test --workspace` still builds against the full API.

**Versions and tools.** `release_info.py metadata` fails unless `pyproject.toml`,
`crates/openquant/Cargo.toml` and `crates/pyopenquant/Cargo.toml` carry the same version,
and on a tag unless the tag is `v` + that version. The Rust toolchain comes from
`rust-toolchain.toml`; maturin is pinned by `MATURIN_VERSION` in the workflow.

The same checks, locally:

```bash
python3 scripts/release/release_info.py metadata --tag v0.1.0
python3 scripts/release/release_info.py notes 0.1.0
```

## One-time maintainer setup

Done once, by a repository admin, before the first release. None of it can be done from a
pull request.

1. **Decide the distribution name** (#58) and set `[project] name` in `pyproject.toml`.
2. **PyPI trusted publisher.** On pypi.org, *Your account → Publishing → Add a new pending
   publisher* (the project does not exist yet), GitHub tab: PyPI project name = the name
   from step 1; owner `Open-Quant`; repository `openquant`; workflow `release.yml`;
   environment `pypi`. No API token is created or stored.
3. **`pypi` environment.** GitHub → Settings → Environments → New environment `pypi`.
   Recommended: *Deployment branches and tags* limited to tags matching `v*`, and a required
   reviewer, so a publish waits for a human approval.
4. **crates.io token.** On crates.io, logged in as the account that will own `openquant`:
   *Account Settings → API Tokens → New Token* with the `publish-new` and `publish-update`
   scopes, limited to the crate `openquant`. Store it as the repository secret
   `CARGO_REGISTRY_TOKEN` (Settings → Secrets and variables → Actions). After the first
   publish, add the other maintainers with `cargo owner --add`.
5. **Tag protection (recommended).** A ruleset restricting who can create `v*` tags, since
   pushing one publishes.
6. **Dry run.** Actions → Release → Run workflow on `main`, `dry_run` checked; every job up
   to the publish jobs must pass.

## Cutting a release

1. **Versions.** Set the same version `X.Y.Z` in `pyproject.toml`,
   `crates/openquant/Cargo.toml` and `crates/pyopenquant/Cargo.toml`, then refresh the lock
   files: `cargo update -w` (the workspace entries in `Cargo.lock`) and `uv lock`.
2. **Changelog.** In `CHANGELOG.md`, rename `## Unreleased` to `## X.Y.Z - YYYY-MM-DD` and
   open a new, empty `## Unreleased` above it (an empty section is fine until the next entry).
   Update the intro paragraph if it still says nothing is published. Regenerate the docs
   page: `python3 scripts/docs/generate_site_pages.py --write`.
3. **Check locally.** `python3 scripts/release/release_info.py metadata --tag vX.Y.Z` and
   `python3 scripts/release/release_info.py notes X.Y.Z` must both succeed.
4. **Pull request.** Open it, let CI and the release workflow's pull-request run pass, merge.
5. **Dry run on main (optional).** Actions → Release → Run workflow, `dry_run` checked.
6. **Tag** the merged commit on `main`:
   `git tag -a vX.Y.Z -m "OpenQuant X.Y.Z" && git push origin vX.Y.Z`.
7. **Watch the run.** It waits for the release checks, the SADF test, every wheel, the smoke
   tests and the sdist; then (after the `pypi` environment's approval, if one is required)
   publishes to PyPI and crates.io and creates the GitHub Release with the notes and the
   built files attached.

If one publish job fails after the other succeeded, fix the cause and re-run the failed job
from the Actions page; do not move the tag. PyPI and crates.io never accept the same version
twice, so a broken release is fixed by a new patch version, not by re-uploading.

## Local checks before tagging

The workflow runs all of these; they are listed for running by hand.

1. `cargo fmt -- --check`
2. `cargo clippy --workspace --all-targets --all-features -- -D warnings`
3. `cargo test --workspace --lib --tests --all-features -- --skip test_sadf_test`
4. `cargo test -p openquant --test structural_breaks test_sadf_test -- --ignored`
5. `cargo publish -p openquant --locked --dry-run`
6. A wheel and the Quickstart smoke test:
   `uv run --with maturin maturin build --release --out dist`, then install the wheel into a
   fresh venv and, from outside the repository, run
   `python scripts/release/smoke_wheel.py --dist-name <name> --version X.Y.Z --abi3`.

Benchmarks are not a release gate in CI. To refresh them:

1. `cargo bench -p openquant --bench perf_hotspots --bench synthetic_ticker_pipeline -- --sample-size 10 --warm-up-time 1 --measurement-time 1`
2. `python3 scripts/collect_bench_results.py --criterion-dir target/criterion --out benchmarks/latest_benchmarks.json --allow-list benchmarks/benchmark_manifest.json`
3. `python3 scripts/check_bench_thresholds.py --baseline benchmarks/baseline_benchmarks.json --latest benchmarks/latest_benchmarks.json --max-regression-pct 35 --overrides benchmarks/threshold_overrides.json`

## Post-release

- Copy `benchmarks/latest_benchmarks.json` to `benchmarks/baseline_benchmarks.json` for the
  next cycle.
- Once the package installs from PyPI, update the install instructions (README, the
  docs-site Quickstart and Python bindings pages).

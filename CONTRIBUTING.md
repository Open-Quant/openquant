# Contributing to OpenQuant

OpenQuant is a Rust implementation of the methods in *Advances in Financial Machine
Learning* (AFML), with Python bindings. It is pre-release (0.1.0, unpublished), so the
bar for a change is that it is **correct against an independent reference** and that
the gates below pass. This page covers setup, the gates, how fixtures are made, and the
conventions for issues and pull requests.

By taking part you agree to the [Code of Conduct](CODE_OF_CONDUCT.md). Report security
problems privately, as described in [SECURITY.md](SECURITY.md), not in an issue.

## Setup

You need:

- A Rust toolchain. The version is pinned in `rust-toolchain.toml` (currently 1.98.1,
  with `clippy` and `rustfmt`); `rustup` installs it automatically the first time you run
  `cargo` in the repository. Do not override it: `clippy -D warnings` is only
  reproducible on the pinned version.
- Python 3.11, 3.12 or 3.13 (CI tests 3.11 and 3.13; the recipes use 3.13) and
  [`uv`](https://docs.astral.sh/uv/).
- [`bun`](https://bun.sh/) if you touch the documentation site.
- Optionally [`just`](https://github.com/casey/just); every recipe below is also given
  as the plain command it runs.

```bash
git clone https://github.com/Open-Quant/openquant.git && cd openquant

# Python environment and the openquant extension (just py-setup && just py-develop)
uv venv --python 3.13 .venv
uv sync --group dev
uv run --python .venv/bin/python maturin develop --manifest-path crates/pyopenquant/Cargo.toml
uv run --python .venv/bin/python python -c "import openquant; print('ok')"
```

`maturin develop` builds `crates/pyopenquant` and installs it into `.venv`. Re-run it
after any change to Rust code that the bindings call, or the Python tests run against the
old build. `just py-develop-release` builds it optimised, which the notebooks and heavy
calls such as `structural_breaks.get_sadf` need to finish in reasonable time.

## Build and test (Rust)

```bash
cargo fmt -- --check                                                    # just fmt-check
cargo clippy --workspace --all-targets --all-features -- -D warnings    # just clippy
cargo test --workspace --lib --tests --all-features -- --skip test_sadf_test   # just test-fast
```

`test-fast` is what CI runs on every pull request. The SADF test in
`crates/openquant/tests/structural_breaks.rs` takes minutes and is `#[ignore]`d; it runs
nightly. Run it yourself when you change `structural_breaks`:

```bash
cargo test -p openquant --test structural_breaks test_sadf_test -- --ignored   # just test-slow
```

CI also builds the rustdoc with every public item documented and every intra-doc link
resolving, and runs the doctests:

```bash
RUSTDOCFLAGS="-D missing_docs -D rustdoc::broken_intra_doc_links" cargo doc -p openquant --no-deps
cargo test -p openquant --doc
cargo bench -p openquant --no-run                                       # just bench-compile
```

## Python tests, lint and types

After `maturin develop`:

```bash
uv run --python .venv/bin/python pytest python/tests -q                 # just py-test
uv run --python .venv/bin/python python -m mypy.stubtest openquant._core   # just py-stubtest
```

`py-stubtest` checks the `.pyi` stubs in `python/openquant/_core/` against the built
extension: every function, parameter name, kind and default must match.

Lint, formatting and type checks do not need the extension (the compiled module is typed
by its stubs):

```bash
uv run --python .venv/bin/python ruff check python/                     # just py-lint
uv run --python .venv/bin/python ruff format --check python/            #   (all three)
uv run --python .venv/bin/python mypy                                   #
```

`ruff format python/` fixes formatting. The ruff and mypy versions are pinned in the `dev`
group of `pyproject.toml`, and mypy's scope is `[tool.mypy]` there.

## Generated API artefacts

Three files are generated, and CI fails if a committed copy is stale. After you add,
rename or change the signature or docstring of a binding (a `#[pyfunction]` in
`crates/pyopenquant`) or a public Rust function, rebuild the extension and regenerate them:

```bash
just py-develop
just py-api-docs
```

`py-api-docs` runs these three generators; each has a `--check` mode that CI runs:

| Generator | Writes | CI check |
|---|---|---|
| `scripts/generate_python_stubs.py` | `python/openquant/_core/*.pyi`, from the Rust source | `python scripts/generate_python_stubs.py --check` |
| `scripts/generate_api_inventory.py` | `docs-site/src/data/apiInventory.ts`, the public Rust and Python API inventory | `bun run check:api-drift` |
| `scripts/generate_python_api_reference.py` | `docs-site/src/data/pythonApiReference.json`, from the built extension | `bun run check:python-reference` |

The API reference generator imports the built extension, so run `just py-develop` first.

## Benchmarks

Criterion benchmarks live in `crates/openquant/benches/`. If you change a hot path, run
them and check the result against the committed baseline; CI does the same on pull
requests that touch `crates/openquant/`.

```bash
cargo bench -p openquant --bench perf_hotspots --bench synthetic_ticker_pipeline   # just bench-all
python3 scripts/collect_bench_results.py --criterion-dir target/criterion --out benchmarks/latest_benchmarks.json --allow-list benchmarks/benchmark_manifest.json   # just bench-collect
python3 scripts/check_bench_thresholds.py --baseline benchmarks/baseline_benchmarks.json --latest benchmarks/latest_benchmarks.json --max-regression-pct 35 --overrides benchmarks/threshold_overrides.json   # just bench-check
```

A regression over 35% fails unless `benchmarks/threshold_overrides.json` allows it. Do not
update `benchmarks/baseline_benchmarks.json` in the same pull request as a slowdown
without saying why in the description.

## Documentation

The site at <https://open-quant.github.io/openquant/> is built from `docs-site/` (Astro +
Starlight). Module pages are in `docs-site/src/content/docs/modules/`.

```bash
cd docs-site
bun install --frozen-lockfile
bun run dev            # local preview
bun run check:docs     # every docs gate, in the order CI runs them
```

`check:docs` runs, in order: `build`, `stage:rustdoc` (rustdoc into the built site, so
`check:links` can check links to it), `check:links`, `check:api-drift`,
`check:content-schema`, `check:contrast`, `check:coverage`, `check:generated-pages`,
`check:examples` (every Rust block in the docs is compiled with `cargo check`),
`check:python-examples` (every Python block is executed) and `check:python-reference`.
The last two need the extension built into `.venv` (see Setup); `check:python-examples`
also accepts an interpreter named by `DOC_PYTHON` that can `import openquant`. A block
that cannot run must opt out visibly with ` ```python doc-check=skip `.

Some pages are generated from other files in the repository, and their gates fail when a
source changes without regenerating them. Both scripts are standard-library Python:

```bash
# Research gallery (from the executed notebooks), performance page (from benchmarks/)
# and changelog page (from CHANGELOG.md). Checked by `bun run check:generated-pages`.
python3 scripts/docs/generate_site_pages.py --write

# The coverage dashboard's tables. Checked by `bun run check:coverage`.
python3 scripts/docs/check_coverage.py --write
```

Run `generate_site_pages.py --write` after re-running the notebooks, changing a benchmark
file or editing `CHANGELOG.md`, and `check_coverage.py --write` after adding or removing a
public module or a docs page, or changing a page's `status`.

Claims on a page need a source: an AFML section or snippet, a paper, or a test in this
repository. Do not raise a page's status (`draft`, `authored`, `reviewed`, …) unless you
have done what that status means.

## Notebooks

The research runbooks are `notebooks/python/NN_*.ipynb`, committed with their outputs,
and the figures they export to `docs-site/public/figures/notebooks/`. The Notebooks
workflow executes all of them on pull requests that touch notebooks, Python, crates or the
lockfiles, and fails if the committed outputs are stale. If your change alters what a
notebook prints or draws, re-run it and commit the result:

```bash
just py-develop-release                  # the notebooks are far faster against a release build
just notebooks-run                       # execute every notebook in place and re-export figures
just notebooks-run --only 05             # or just one (number or stem)
just notebooks-verify                    # compare the tree with HEAD, as CI does
```

`notebooks-verify` ignores float noise and PNG bytes, but not a changed source cell, a new
output line or a missing or extra figure. The notebooks read the committed SYNTHETIC
sample through `openquant.data.fetch`, so none of this touches the network. After
re-running them, regenerate the research gallery with
`python3 scripts/docs/generate_site_pages.py --write` (see Documentation).

## Research smoke checks

```bash
uv run --python .venv/bin/python python notebooks/python/scripts/smoke_all.py        # just notebook-smoke
uv run --python .venv/bin/python python experiments/run_pipeline.py --config experiments/configs/futures_oil_baseline.toml --out experiments/artifacts   # just exp-run
cargo run -p openquant --example research_notebook_smoke
```

Every notebook under `notebooks/python/` follows the notebook contract
(`docs-site/src/content/docs/workflows/research-notebook-contract.md`): a runbook's sections run
Setup, Hypothesis, Data, Method, Results, Analysis, Promotion decision, Self-review checklist and
Reproducibility, an API tour makes no claims, and both end with an `nbrepro.footer(...)` cell.
`just notebooks-lint` checks it (CI runs it in `python-lint`).

## Tests and fixtures

A test is only useful if it fails when the code is wrong. `docs/test-sensitivity-audit.md`
records which tests in this repository did not, and how that was measured.

- **Compare against an independent reference.** Expected values come from the AFML
  snippet, a paper, or a well-known library (numpy, scipy, pandas, scikit-learn), never
  from running this library and pasting its output back in.
- **Commit the generator.** A new reference fixture goes in `tests/fixtures/<module>/`
  with a `generate.py` beside it whose docstring says what it computes, from which
  source, and the exact command that regenerates it, for example:

  ```bash
  uv run --no-project --with numpy --with scipy python tests/fixtures/hrp/generate.py
  ```

  `--no-project` keeps the generator's dependencies out of this repository's own
  environment. The generator must not import `openquant`.
- **Use the tolerance the reference supports**, and say in a comment why that tolerance.
- **Do not copy data or expected values from a project whose license does not allow
  it.** In particular, do not add anything taken from mlfinlab releases after
  March 2020: they are not open source. [`tests/FIXTURES.md`](tests/FIXTURES.md) lists
  where every existing fixture came from and under what license, including the ones
  whose provenance is still open.
- A test that pins a known defect is `#[ignore = "FINDING: ..."]` (Rust) or
  `xfail(strict=True)` (Python), with a linked issue, until the fix lands.

## Issues and planning

Work is tracked in [GitHub issues](https://github.com/Open-Quant/openquant/issues).
Each issue carries one `track:*` label (area) and one priority (`P0`–`P2`); the label set
is described in `.github/labels.yml`. The current backlog was derived from
`docs/design/production-readiness-brief.md` (requirements `RQ-*`) and
`docs/design/production-readiness-slices.md` (the slices that became issues).

The maintainer runs the backlog with [AI-DLC](AI-DLC.md) (`ai-dlc.toml`,
`docs/development-workflow.md`). For each issue there is a reviewed work record in
`.ai-dlc/work/<id>.toml` holding its scope, acceptance criteria and a specification
decision. When `requires_spec = true`, the change also needs a formal specification with
the configured provider (OpenSpec, under `openspec/`) before implementation; otherwise
the record says why none is needed. The `verify.yml` workflow runs
`ai-dlc agents render --check` and `ai-dlc work validate --all`.

You do not need the `ai-dlc` tool to contribute. Open an issue first for anything bigger
than a small fix, say which AFML section it concerns, and the maintainer will create or
update the work record.

## Pull requests

Look at recently merged pull requests for the house style. In short:

- **Branch** from `main` as `<type>/<issue>-<slug>`, e.g. `fix/110-portfolio-simple-returns`.
- **Title** uses a conventional-commit prefix and states the effect, not the activity:
  `fix: weekly/monthly resampling scrambled the price matrix in four modules (#93)`.
  Prefixes in use: `fix`, `feat`, `test`, `docs`, `refactor`, `chore`, `ci`, `design`.
  Commit messages follow the same convention.
- **Link the issue** with `Closes #N` (or `Fixes #N`). Use `Refs #N` if the pull request
  only partly addresses it.
- **Bug fixes include a regression test that fails on the old code.** Check it by
  reverting the fix locally and running the test, and say that you did.
- **The description says** what was wrong, what changed, how it was verified (the exact
  commands and results), any behaviour change a user will see, and what was not done or
  not verified. Name defects you found but did not fix, and file them.
- Keep unrelated changes out. One issue per pull request unless they cannot be separated.
- Update the docs page for any module whose behaviour you change.

Every pull request must pass CI, and the one required check is `ci-success`, which
passes only when every CI job passed (or was skipped because the pull request changes
documentation only). The jobs: format, clippy and rustdoc (`lint`), `doctests`, fast tests
(`tests`), `bench-compile`, the Python binding tests with stubtest and the smoke checks on
Python 3.11 and 3.13 (`python`), stubs, ruff and mypy (`python-lint`) and the docs gates
(`docs-checks`). Separate workflows also run the notebooks (for changes they could affect)
and, for changes under `crates/openquant/`, the benchmark regression check.

Before pushing, the quickest local equivalent is:

```bash
just lint test-fast py-lint
uv run --python .venv/bin/python python scripts/generate_python_stubs.py --check
just py-develop py-test py-stubtest
(cd docs-site && bun run check:docs)     # if you touched docs, bindings or public APIs
just notebooks-run && just notebooks-verify   # if you touched notebooks, python/ or crates/
```

## License

OpenQuant is MIT licensed (see `LICENSE`). By contributing you agree that your
contribution is licensed under the same terms, and you confirm that you have the right to
contribute it, including any data or expected values it contains.

---
title: Local Build Setup
description: Build the Rust core, run its tests, and run the documentation quality gates.
status: authored
last_authored: '2026-11-27'
audience:
  - quant-dev
  - platform-engineering
sidebar:
  order: 2
---

This is the maintainer path: compile the Rust core and run the gates CI
runs. If you are here to *use* OpenQuant rather than to change it, the
[Quickstart](/quickstart/) is shorter and ends with a result.

## Build and test the Rust core

To build only the Rust crate without the Python extension, run
`cargo build --locked -p openquant` from the repository root. The workspace-wide
maintainer checks below include the Python extension crate.

```bash
cargo build --all-features
cargo test --workspace --lib --tests --all-features -- --skip test_sadf_test
```

The `--skip` is deliberate. `test_sadf_test` is annotated `#[ignore]` in
`crates/openquant/tests/structural_breaks.rs` as a "long-running
hotspot", so it does not run by default; the explicit skip keeps the
command honest if the annotation is ever removed. Run it on its own when
you have touched `structural_breaks`:

```bash
cargo test -p openquant --test structural_breaks test_sadf_test -- --ignored
```

Lint and format, mirroring the `lint` recipe in the `justfile`:

```bash
cargo fmt -- --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
```

## Documentation quality gates

For a local preview only, install the locked site dependencies and start Astro:

```bash
cd docs-site
bun install --frozen-lockfile
bun run dev
```

Without Bun, use `npm ci` and `npm run dev`. The site itself runs on Node;
Bun is the package manager and script runner used by CI.

For the **full documentation gate**, first build the Python extension using
[the contributor setup](https://github.com/Open-Quant/openquant/blob/main/CONTRIBUTING.md#setup),
then run:

```bash
cd docs-site
bun run check:docs
```

This needs the pinned Rust toolchain, Python dependencies and the built extension.
It runs the checks declared in
[`docs-site/package.json`](https://github.com/Open-Quant/openquant/blob/main/docs-site/package.json):

1. Build the site and stage Rust API documentation with `stage:rustdoc`.
2. Run the design/accessibility, agent-index and navigation-checker tests, then
   check built paths, anchors and redirects.
3. Check API drift, page status/schema, theme contrast, coverage, generated pages and
   Rust-only module figures (including their independent reference tests).
4. Compile Rust snippets on module pages and the complete Rust core workflow,
   execute the ETF Rust example and compare its displayed output, execute runnable
   Python snippets and check the generated Python reference
   against the built extension.

The design/accessibility checks inspect rendered markup and source/style rules. They
do not exercise keyboard, screen-reader, zoom or mobile interactions in a browser.
The aggregate stops at its first failing command; later checks have not run.
Record the failing command and run unaffected checks separately when useful.

The [CI docs job](https://github.com/Open-Quant/openquant/blob/main/.github/workflows/ci.yml)
and [Pages build](https://github.com/Open-Quant/openquant/blob/main/.github/workflows/docs-pages.yml)
run their own named steps, not the local aggregate. Check the job's actual step
results before claiming a gate ran. Pages builds and stages the site, but does not
execute the documented Rust and Python examples.

With npm, invoke component scripts individually with `npm run SCRIPT`.
The aggregate and the `check:rust-output` wrapper call Bun internally.
A complete link check still requires rustdoc staging:

```bash
npm run build
npm run stage:rustdoc
npm run test:links
npm run check:links
```

### Check prose without a native toolchain

For a prose-only review without Cargo or a built extension, run this
dependency-free subset from `docs-site/` in a checkout with Git history:

```bash
node scripts/check-content-schema.mjs
node scripts/check-contrast.mjs
python3 ../scripts/generate_api_inventory.py --check
python3 ../scripts/docs/check_coverage.py --check
python3 ../scripts/docs/generate_site_pages.py --check
```

This does not build the site, prove its links work, or compile or run example code.
With the site dependencies installed, you can also exercise the rendered checks
without Rust:

```bash
npm run build
npm run test:design
npm run test:accessibility
npm run test:links
```

`test:links` tests the checker against fixtures, not against the built site.
Both `test:agent-index` and `check:links` expect staged Rust API pages;
without them, log missing targets as a coverage gap rather than a pass.

### Verify a navigation change in a browser

Start with a fresh tab on a deep module page and repeat at a narrow mobile
viewport, about 320 CSS pixels. Use the built site under `/openquant/`, not a
file URL. Capture the viewport, browser, theme and the result of each step:

1. On mobile, open the navigation menu. All section disclosures should start
   closed; a deep link must not expand its ancestors automatically.
2. With Tab, reach a section summary and check that its focus is visible.
   With Enter or Space, expand it, reach a child link, and follow that link.
3. On the new page, reopen the mobile menu if needed. The disclosures should
   start closed again. Expand the current section and confirm it marks the current
   page. Try both themes and check that code or table scrolling does not force the
   whole page to scroll sideways.

The [rendered design tests](https://github.com/Open-Quant/openquant/blob/main/docs-site/scripts/tests/design-system.test.mjs)
cover initial HTML state and the current-page mark. They do not exercise keyboard
input, focus order or viewport interaction. A targeted browser pass is not a
comprehensive accessibility audit.

### Check the four Rust-only illustrations and ETF output

The [combinatorial optimization](/modules/combinatorial-optimization/),
[ETF trick](/modules/etf-trick/), [fingerprint](/modules/fingerprint/) and
[parallel work partitioning](/modules/hpc-parallel/) illustrations have a
standard-library Python generator. Run these focused checks from the repository root:

```bash
python3 -m unittest discover -s scripts/docs/tests -p test_rust_module_figures.py
python3 scripts/docs/generate_rust_module_figures.py --check
node --test docs-site/scripts/tests/check-rust-output.test.mjs
```

These checks need neither the native extension nor Cargo. The first two verify
independent explanatory calculations and that committed SVGs match the generator;
the third tests the output comparator with synthetic strings. **None executes Rust.**
The figures show exact inventory enumeration, a roll adjustment, population
fingerprint curves and idealized partition work. They are not captured Rust results,
finite-sample fingerprint estimates or measured parallel speedups.

For an intentional figure change, edit `scripts/docs/generate_rust_module_figures.py`,
update its independent tests when the illustrated assumptions change, then run it
with `--write`. Review the four SVGs under `docs-site/public/figures/modules/` and
the page captions together; do not edit a generated SVG alone. Regeneration is not
numerical validation of the Rust implementation.

With the pinned Rust toolchain available, run the distinct execution gate:

```bash
node docs-site/scripts/check-rust-output.mjs
```

It runs `cargo run --quiet --locked -p openquant --example docs_etf_trick` and compares
stdout with **every** `text` fence in the ETF page, in page order. It normalizes line
endings and one final newline only; extra output, changed precision or a missing
block fails. A compilation pass from `check:examples` cannot replace this check.
If the output drifts, inspect `crates/openquant/examples/docs_etf_trick.rs`, the
module implementation and the page's expectations before updating either side.
Do not copy new output into the page until its calculations are justified.

These focused checks diagnose a change; they do not replace the full documentation
gate above or the Rust integration tests for the affected module.

## A reproducible docs verification receipt

Keep source identity and execution identity separate. A restored Git bundle can preserve
all commits while a virtual environment or compiled extension belongs to a different
revision. From the repository root, record these alongside the exact checks:

```bash
git rev-parse HEAD
git status --short
.venv/bin/python -c "import sys, openquant, openquant._core as core; print(sys.executable); print(openquant.__file__); print(core.__file__)"
```

Do not publish raw paths or logs containing private data or credentials. Record the
native build command and outcome after rebuilding. A package version, an import path
or a copied binary alone cannot prove that the extension matches the checked-out Rust.

| Check state | What the receipt should say |
| --- | --- |
| Passed | Exact command, revision, selected environment and observed result. |
| Failed | Command and first substantive error, including a stale-output mismatch. |
| Blocked | Missing capability or dependency, not an inferred success. |
| Not run | Later aggregate stages or browser interactions that were never exercised. |

For hosted CI evidence, save the run and job URLs, run attempt, event, candidate
head SHA, exact job checkout SHA if available, relevant step conclusions and any
named output artifact. On pull requests the default checkout can be a temporary
merge commit; record it separately from the branch head. See the
[hosted evidence guide](https://github.com/Open-Quant/openquant/blob/main/CONTRIBUTING.md#read-hosted-documentation-evidence).
Two green jobs can refer to different commits. A workflow definition describes intended checks, not an executed result;
a successful job may also contain skipped steps. A main-branch pass does not validate
uncommitted or private patch changes. Compare their source scope and rerun the affected
gates on the candidate before claiming candidate acceptance. Build success, deployment
and human acceptance remain separate claims.

In a restricted environment, notebook execution may fail because the Jupyter kernel
cannot open a local socket. Preserve the traceback and rerun the unchanged command in
an environment that supports the required operation. Do not convert an executable
fence to `doc-check=skip`, suppress the kernel failure or label the aggregate green.
Run independent gates separately to retain useful partial evidence. Likewise, a missing
Rust compiler prevents snippet compilation and rustdoc generation even if Node tests pass.

For a resumed session, recover source and the receipt first, then recheck the current
upstream revision before applying changes. Use one restore route (bundle or patch), not
both. Preserve unresolved failures and identify any independently developed changes
that have not been integrated. This receipt is evidence for maintainers, not a claim of
human review, production deployment or validated investment performance.

### Check what readers can see

First find the successful `deploy` job and its corresponding `build` job in
[Docs Pages](https://github.com/Open-Quant/openquant/actions/workflows/docs-pages.yml).
Record the revision the build actually checked out, including the ref for
manually dispatched runs. The branch tip today is not a substitute for that
revision. With the full build SHA as `expected_sha` (or use the prompt), run:

```bash
(
  if [ -z "${expected_sha:-}" ]; then
    printf 'Full SHA checked out by the deployed build: ' >&2
    read -r expected_sha || exit 1
  fi
  index=$(mktemp) || exit 1
  trap 'rm -f "$index"' EXIT
  curl -fsSL --max-time 20 https://open-quant.github.io/openquant/llms.txt \
    -o "$index" || exit 1
  awk -v sha="$expected_sha" '
    /^Source revision:/ { revisions++; print; if ($0 == "Source revision: " sha) rev_ok++ }
    /^Working tree:/ { trees++; print; if ($0 == "Working tree: clean") clean++ }
    END {
      if (sha !~ /^[0-9a-f]+$/ || (length(sha) != 40 && length(sha) != 64) ||
          revisions != 1 || trees != 1 || rev_ok != 1 || clean != 1) exit 1
    }
  ' "$index"
)
```

A zero exit status confirms only that the public index reports the expected
revision and a clean tree. Then open the changed page at its public URL and
verify the actual text, example or navigation. If the revision differs, the
index reports a modified tree or metadata is missing, record what was served
and do not mark that revision live. A newer site may have superseded the
selected run; check the newer deployment and the actual page before concluding
that the change is absent. This does not verify package publication.

## When a gate fails

**`check:links` reports broken links.** Run the site build and `stage:rustdoc` first. The link
checker reads the built output in `dist/`, not the Markdown sources, so a
stale or absent `dist/`, or missing staged rustdoc, produces failures.
The checker also verifies fragment anchors and relative links, not just files. Internal links must
resolve under the `/openquant` base path once built — a link that works
in `astro dev` can still 404 in production if it omits the base.

**`check:api-drift` fails.** Read the error first: the source-derived inventory
may be stale, or the binding documentation checks may have found a missing contract.
The generated file is `docs-site/src/data/apiInventory.ts`; the generator is under
`scripts/`. This is a symbol/source scan, not a Rust/Python behavioral parity test.
Inspect the public API diff before regenerating from the repository root:

```bash
python3 scripts/generate_api_inventory.py --check
# Only after reviewing the intended source changes:
python3 scripts/generate_api_inventory.py
```

**`check:python-reference` fails.** This is a different artifact:
`docs-site/src/data/pythonApiReference.json`. The generator checks compiled function
names and parameter names against the stubs, then combines compiled documentation
with public Python source. Confirm the selected package and extension paths before
rebuilding or regenerating; an old extension or another installed checkout can
produce a misleading result. See [bindings setup](/setup/python-bindings/).

**A Python example fails, times out or prints different output.** The checker writes
one standalone script per fence under `target/doc-python-examples/`, then runs it
from the repository root. Its final failure report names that file; inspect it and
reproduce the same inputs before changing the page. Each runnable fence has its own process,
so a variable or import from an earlier fence is not available.

You can prefer the prepared environment from the repository root:

```bash
DOC_PYTHON="$PWD/.venv/bin/python" node docs-site/scripts/check-doc-python-examples.mjs
```

Set `DOC_PYTHON` to an absolute executable path, not a shell command. The initial
import probe and the example processes can have different working directories: a
relative path such as `../.venv/bin/python` from `docs-site/` can pass the import probe
and then fail to start every example from the repository root. If the requested
executable is missing or cannot import `openquant`, the gate does not fall back
to another environment. Do not include command-line arguments in the value.
When the variable is unset or empty, discovery tries the checkout's `.venv`
interpreters, then `python3`. Read the `via ...` path in the opening summary. When a local
native extension is present under `python/openquant/`, the checker prepends the
checkout's Python source to `PYTHONPATH`; otherwise an installed package can satisfy
the import. Confirm the actual package and extension paths before treating a pass
as evidence for this checkout.

| Report | Meaning and next check |
| --- | --- |
| `FAIL` | The process exited unsuccessfully. Read the traceback and the saved script; verify package paths, fixture paths and the API contract. |
| `TIMEOUT` | A runnable block exceeded the default 180,000 ms limit, or terminated with SIGTERM. Check build mode and workload before setting `DOC_PYTHON_TIMEOUT_MS` to a larger positive millisecond value. |
| `STALE` | The process succeeded, but its stdout differs from the following eligible `text` fence after normalizing line endings, trailing spaces and outer whitespace. Check the expected numbers and their conventions; the comparison does not use a numerical tolerance. |
| `skip` | The fence received a syntax check only. Its imports, calls and displayed results were not executed. |

A Python fence's output can be matched to a later `text` fence before the next Python
fence, even with prose between them. Use `doc-output=skip` only for text that is not
program output; use the documented per-fence execution skip for examples that need
unavailable caller data. Skipping a block is a coverage exception to report, not a
fix for a runnable example that fails.

**Coverage or generated pages are stale.** Run the owning generator from the
repository root, then review its diff:

```bash
python3 scripts/docs/check_coverage.py --check
python3 scripts/docs/generate_site_pages.py --check
# After the underlying documentation/notebook/source change is ready:
python3 scripts/docs/check_coverage.py --write
python3 scripts/docs/generate_site_pages.py --write
```

Coverage counts module claims and page classifications; it does not establish
citation quality, numerical correctness or example execution. Generated-gallery
updates do not rewrite authored runbook pages. Update those separately when the
canonical notebook's interpretation changes.

**`check:content-schema` fails.** The message names the file and the rule.
The taxonomy is defined at the top of
`docs-site/scripts/check-content-schema.mjs`:

| `status` | Means | Date field it must carry |
|---|---|---|
| `generated` | emitted from the page's declared source; claims no human review | `last_generated` |
| `draft` | hand-written, known incomplete; claims nothing | **none** |
| `authored` | written against the code, not yet read by a human; not a test receipt | `last_authored` |
| `reviewed` | a human read the page end to end | `last_validated` |
| `validated` | reviewed *and* checked against the code it describes | `last_validated` |

Two rules catch most failures. A `draft` may not carry `last_validated` —
drop the field; do not invent a validation. A stamp may not predate the
page's last content change. AI-authored changes stop at `authored` with a current
`last_authored`; `reviewed` and `validated` require a human. Use `draft` when
the page is incomplete. The gate reads Git history and considers the mtime
of working-tree edits. It compares the body when a change looks stale, so
moving only frontmatter does not ordinarily require rereading unchanged prose.

Run `npm run check:content-schema` from `docs-site/` on the **committed
candidate** as well as the working copy. If an integration tool reconstructs
several commits as one, recheck that new head: the gate can now see the
new commit date as the last body-change date even when the proposed text
did not change again. Reread and accurately stamp only the affected authored
pages, or keep incomplete pages draft. Use the final head's hosted check as
evidence, not an earlier branch's pass.

The coloured pill a reader sees at the top of every page is rendered from
`status` by the `Banner` override in `src/components/DocStatusBanner.astro`,
so there is nothing to keep in step by hand.

## Next

- [Python Bindings Setup](/setup/python-bindings/) — the PyO3 extension
- [Troubleshooting](/setup/troubleshooting/) — compile and link failures

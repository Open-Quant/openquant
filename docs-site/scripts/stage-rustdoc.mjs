/**
 * Build the openquant crate's rustdoc and publish it under the site's /api/rust/.
 *
 *   node scripts/stage-rustdoc.mjs            # cargo doc, then copy into dist/api/rust/
 *   node scripts/stage-rustdoc.mjs --no-build # copy an existing target/doc
 *
 * Run after `astro build` (it writes into dist/). Module pages link each item in their
 * `rust_api` frontmatter to its page here (scripts/remark-api-reference.mjs), and
 * check:links then fails on any link whose rustdoc file does not exist, so a renamed or
 * removed item cannot leave a dead link behind.
 *
 * The same pages move to docs.rs once the crate is published there.
 */
import { execFileSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const docsSite = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const repoRoot = path.resolve(docsSite, '..');
const dist = path.join(docsSite, 'dist');
const out = path.join(dist, 'api', 'rust');

if (!fs.existsSync(dist)) {
  console.error('dist/ not found. Run the astro build first.');
  process.exit(1);
}

const targetDir = process.env.CARGO_TARGET_DIR
  ? path.resolve(repoRoot, process.env.CARGO_TARGET_DIR)
  : path.join(repoRoot, 'target');
const docDir = path.join(targetDir, 'doc');

if (!process.argv.includes('--no-build')) {
  execFileSync('cargo', ['doc', '-p', 'openquant', '--no-deps'], {
    cwd: repoRoot,
    stdio: 'inherit',
    env: { ...process.env, RUSTDOCFLAGS: '-D rustdoc::broken_intra_doc_links' },
  });
}

if (!fs.existsSync(path.join(docDir, 'openquant', 'index.html'))) {
  console.error(`${docDir}/openquant/index.html not found; cargo doc did not produce the crate docs.`);
  process.exit(1);
}

fs.rmSync(out, { recursive: true, force: true });
fs.mkdirSync(out, { recursive: true });
fs.cpSync(docDir, out, { recursive: true });
// cargo's lock file is a build byproduct, not docs.
fs.rmSync(path.join(out, '.lock'), { force: true });

// /api/rust/ itself lands on the crate root.
fs.writeFileSync(
  path.join(out, 'index.html'),
  '<!doctype html><meta charset="utf-8"><title>Redirecting…</title>' +
    '<meta http-equiv="refresh" content="0; url=/openquant/api/rust/openquant/">' +
    '<link rel="canonical" href="/openquant/api/rust/openquant/">' +
    '<a href="/openquant/api/rust/openquant/">openquant rustdoc</a>\n'
);

let files = 0;
const count = (dir) => {
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    if (entry.isDirectory()) count(path.join(dir, entry.name));
    else files++;
  }
};
count(out);
console.log(`Staged rustdoc into ${path.relative(docsSite, out)}/ (${files} files).`);

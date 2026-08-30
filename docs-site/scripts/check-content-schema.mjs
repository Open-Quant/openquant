import { execFileSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';

const docsRoot = path.resolve(process.cwd(), 'src/content/docs');

/**
 * The status taxonomy. The ordering is deliberate: each value claims strictly
 * more human attention than the one before it.
 *
 *   generated — machine-emitted from src/data/moduleDocs.ts. Nobody has read it.
 *   draft     — hand-written, known to be incomplete. Claims nothing.
 *   reviewed  — a human read the page and it stands on its own.
 *   validated — reviewed AND checked against the code it describes.
 *
 * Each status names the date field that backs it. `draft` and `generated` cannot
 * claim `last_validated`, because neither has been validated by anyone — that is
 * the whole point of the taxonomy, and the gate below enforces it.
 */
const STATUS_DATE_FIELD = {
  generated: 'last_generated',
  draft: null,
  reviewed: 'last_validated',
  validated: 'last_validated',
};
const STATUSES = Object.keys(STATUS_DATE_FIELD);
const DATE_FIELDS = ['last_generated', 'last_validated'];

function walk(dir) {
  const files = [];
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      files.push(...walk(full));
      continue;
    }
    if (entry.name.endsWith('.md') || entry.name.endsWith('.mdx')) {
      files.push(full);
    }
  }
  return files;
}

function parseFrontmatter(text) {
  const m = text.match(/^---\n([\s\S]*?)\n---/);
  if (!m) return {};
  const raw = m[1];
  const out = {};
  for (const line of raw.split('\n')) {
    if (!line.trim() || line.trim().startsWith('#')) continue;
    const kv = line.match(/^([A-Za-z_][A-Za-z0-9_]*):\s*(.*)$/);
    if (kv) out[kv[1]] = unquote(kv[2]);
  }
  return out;
}

function unquote(value) {
  const v = String(value).trim();
  if (v.length >= 2 && (v[0] === "'" || v[0] === '"') && v.at(-1) === v[0]) {
    return v.slice(1, -1);
  }
  return v;
}

/** A real YYYY-MM-DD date, not merely a string shaped like one (2026-02-31 is not a date). */
function parseIsoDate(value) {
  if (!/^\d{4}-\d{2}-\d{2}$/.test(value)) return null;
  const d = new Date(`${value}T00:00:00Z`);
  if (Number.isNaN(d.getTime())) return null;
  return d.toISOString().slice(0, 10) === value ? d : null;
}

function git(args, cwd) {
  return execFileSync('git', args, { cwd, encoding: 'utf8', stdio: ['ignore', 'pipe', 'pipe'] });
}

/**
 * Last date each doc actually changed — the newer of its last commit date and,
 * when the working tree copy differs from HEAD (or is untracked), its filesystem
 * mtime. Uncommitted edits count: otherwise you could edit a page and keep a
 * stale stamp green right up until the moment you commit.
 *
 * Returns a Map of absolute path -> YYYY-MM-DD. Outside a git checkout (a release
 * tarball, say) freshness falls back to mtime alone.
 */
function lastChangedDates(files) {
  const dates = new Map();
  const mtimeDay = (file) => fs.statSync(file).mtime.toISOString().slice(0, 10);

  let repoRoot;
  try {
    repoRoot = git(['rev-parse', '--show-toplevel'], docsRoot).trim();
  } catch {
    // Not a git checkout (a release tarball, say). Fall back to mtime for everything.
    for (const file of files) dates.set(file, mtimeDay(file));
    return dates;
  }

  // One `git log` walk over the whole docs tree rather than one subprocess per
  // file. Commits are newest-first, so the first time a path appears is its most
  // recent change.
  const log = git(
    ['log', '--pretty=format:%x00%cI', '--name-only', '--diff-filter=d', '--', docsRoot],
    repoRoot
  );
  let commitDay = null;
  for (const line of log.split('\n')) {
    if (line.startsWith('\0')) {
      commitDay = line.slice(1, 11);
      continue;
    }
    if (!line.trim() || !commitDay) continue;
    const abs = path.resolve(repoRoot, line.trim());
    if (!dates.has(abs)) dates.set(abs, commitDay);
  }

  // Anything modified or untracked in the working tree is at least as new as its mtime.
  const status = git(['status', '--porcelain', '--', docsRoot], repoRoot);
  for (const line of status.split('\n')) {
    if (!line.trim()) continue;
    const rel = line.slice(3).split(' -> ').at(-1).replace(/^"|"$/g, '');
    const abs = path.resolve(repoRoot, rel);
    if (!fs.existsSync(abs) || fs.statSync(abs).isDirectory()) continue;
    const day = mtimeDay(abs);
    if (!dates.has(abs) || day > dates.get(abs)) dates.set(abs, day);
  }

  for (const file of files) {
    if (!dates.has(file)) dates.set(file, mtimeDay(file));
  }
  return dates;
}

const files = walk(docsRoot);
const changed = lastChangedDates(files);
const errors = [];

for (const file of files) {
  const rel = path.relative(process.cwd(), file);
  const text = fs.readFileSync(file, 'utf8');
  const frontmatter = text.match(/^---\n([\s\S]*?)\n---/)?.[1] ?? '';
  const fm = parseFrontmatter(text);
  const fail = (msg) => errors.push(`${rel}: ${msg}`);

  for (const key of ['title', 'description', 'status']) {
    if (!(key in fm) || fm[key].trim() === '') fail(`missing ${key}`);
  }

  const status = fm.status;
  if (status === undefined || status.trim() === '') continue;

  if (!STATUSES.includes(status)) {
    fail(
      `status: ${status} is not one of ${STATUSES.join(' | ')}. ` +
        `Use 'generated' for machine-emitted pages, 'draft' for hand-written pages ` +
        `that are not finished, 'reviewed' once a human has read the page, and ` +
        `'validated' once it has also been checked against the code.`
    );
    continue;
  }

  // The reader-facing badge is rendered from the page's own banner frontmatter,
  // so it can drift from `status` unless something checks. This is that check.
  const badge = frontmatter.match(/doc-status--([a-z]+)/)?.[1];
  if (badge && badge !== status) {
    fail(
      `status: ${status} but the reader-facing badge says '${badge}'. ` +
        `Update the banner content so the badge matches the frontmatter status.`
    );
  }

  // Every date present must be a real ISO date, whether or not this status needs it.
  for (const field of DATE_FIELDS) {
    if (field in fm && !parseIsoDate(fm[field])) {
      fail(`${field}: ${fm[field]} is not a valid ISO date (expected YYYY-MM-DD)`);
    }
  }

  const dateField = STATUS_DATE_FIELD[status];
  if (dateField === null) {
    if ('last_validated' in fm) {
      fail(
        `status: draft must not carry last_validated — a draft has not been ` +
          `validated by anyone. Drop the field, or raise the status to 'reviewed'.`
      );
    }
    continue;
  }

  if (!(dateField in fm) || fm[dateField].trim() === '') {
    fail(`status: ${status} requires ${dateField} (YYYY-MM-DD)`);
    continue;
  }

  const stamped = parseIsoDate(fm[dateField]);
  if (!stamped) continue; // already reported above

  // The rule that makes the stamp mean something: a page cannot claim to have
  // been reviewed or generated BEFORE the last time its content actually changed.
  // Without this, `status: validated` is just a string anyone can paste, and can
  // be bulk-applied to a whole corpus — which is exactly how this site ended up
  // with 59 pages sharing one hardcoded review date.
  const changedOn = changed.get(file);
  if (changedOn && changedOn > fm[dateField]) {
    const what =
      status === 'generated'
        ? `Re-run 'node scripts/generate-module-doc-pages.mjs' to re-stamp it`
        : `Re-read the page; if it still stands, set ${dateField}: '${changedOn}'. ` +
          `If you have not re-read it, lower the status to 'draft' instead of ` +
          `bumping the date`;
    fail(
      `stale ${dateField}: page last changed ${changedOn} but claims ` +
        `${dateField}: '${fm[dateField]}'. The content moved after the ` +
        `stamp, so the stamp no longer describes this page. ${what}.`
    );
  }
}

if (errors.length) {
  console.error(`Content schema check FAILED (${errors.length} problem(s)):`);
  for (const err of errors) console.error(`- ${err}`);
  process.exit(1);
}

const tally = {};
for (const file of files) {
  const s = parseFrontmatter(fs.readFileSync(file, 'utf8')).status ?? '(none)';
  tally[s] = (tally[s] ?? 0) + 1;
}
const summary = STATUSES.filter((s) => tally[s]).map((s) => `${tally[s]} ${s}`).join(', ');
console.log(`Content schema check passed (${files.length} docs files: ${summary}).`);

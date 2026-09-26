/**
 * Remark plugin that appends an "API reference" section to every module page.
 *
 * The section is built from the page's `rust_api` / `python_api` frontmatter, resolved
 * against the generated inventories rather than written by hand:
 *
 * - Rust names resolve through `apiInventory.rustItems` (scripts/generate_api_inventory.py)
 *   to their rustdoc page under `/api/rust/`, which the Pages build fills with
 *   `cargo doc -p openquant --no-deps` (scripts/stage-rustdoc.mjs).
 * - Python names resolve through `pythonApiReference.json`
 *   (scripts/generate_python_api_reference.py) to their entry on `/api/python/<module>/`,
 *   and carry the first line of the function's docstring.
 *
 * A name that does not resolve fails the build: a listed item must link somewhere real.
 * Because the section is mdast, not HTML, it appears in the page's table of contents, and
 * remark-base-url (which runs after this plugin) adds the site base to every link.
 */
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const dataDir = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../src/data');

function loadInventory() {
  const text = fs.readFileSync(path.join(dataDir, 'apiInventory.ts'), 'utf8');
  const start = text.indexOf('{');
  const end = text.lastIndexOf('}');
  return JSON.parse(text.slice(start, end + 1));
}

function loadPythonReference() {
  return JSON.parse(fs.readFileSync(path.join(dataDir, 'pythonApiReference.json'), 'utf8'));
}

export const RUSTDOC_BASE = '/api/rust';
export const PYTHON_REF_BASE = '/api/python';

/** rustdoc URL of a module, e.g. `openquant::util::volatility` -> `/api/rust/openquant/util/volatility/`. */
export function rustModuleUrl(modulePath) {
  return `${RUSTDOC_BASE}/${modulePath.split('::').join('/')}/index.html`;
}

/** rustdoc URL of one item in `modulePath`, or null if the inventory does not know it. */
export function rustItemUrl(items, modulePath, name) {
  const kinds = items[modulePath];
  if (!kinds || !kinds[name]) return null;
  const dir = `${RUSTDOC_BASE}/${modulePath.split('::').join('/')}`;
  if (kinds[name] === 'method') {
    const [owner, method] = name.split('::');
    const ownerKind = kinds[owner];
    if (!ownerKind) return null;
    return `${dir}/${ownerKind}.${owner}.html#method.${method}`;
  }
  return `${dir}/${kinds[name]}.${name}.html`;
}

/** Python reference page for a module slug. */
export function pythonModuleUrl(slug) {
  return `${PYTHON_REF_BASE}/${slug}/`;
}

const text = (value) => ({ type: 'text', value });
const code = (value) => ({ type: 'inlineCode', value });
const link = (url, children) => ({ type: 'link', url, children });
const para = (children) => ({ type: 'paragraph', children });
const heading = (depth, value) => ({ type: 'heading', depth, children: [text(value)] });
const list = (items) => ({
  type: 'list',
  ordered: false,
  spread: false,
  children: items.map((children) => ({
    type: 'listItem',
    spread: false,
    children: [para(children)],
  })),
});

function summaryLine(doc) {
  const first = (doc || '').split(/\n\s*\n/)[0].replace(/\s+/g, ' ').trim();
  return first;
}

function resolveRust(inventory, pageModule, name) {
  const items = inventory.rustItems;
  const preferred = pageModule ? `openquant::${pageModule}` : null;
  if (preferred && items[preferred]?.[name]) {
    return { module: preferred, url: rustItemUrl(items, preferred, name) };
  }
  const hits = Object.keys(items).filter((m) => items[m][name]);
  if (hits.length === 1) {
    return { module: hits[0], url: rustItemUrl(items, hits[0], name) };
  }
  return { error: hits.length ? `ambiguous (in ${hits.join(', ')})` : 'not a public item' };
}

function resolvePython(reference, qualified) {
  const dot = qualified.lastIndexOf('.');
  const moduleName = qualified.slice(0, dot);
  const member = qualified.slice(dot + 1);
  const mod = reference.modules.find((m) => m.import === `openquant.${moduleName}`);
  if (!mod) return { error: `no module openquant.${moduleName}` };
  const entry =
    mod.functions.find((f) => f.name === member) ?? mod.classes.find((c) => c.name === member);
  if (!entry) return { error: `openquant.${moduleName} has no public ${member}` };
  return { url: `${pythonModuleUrl(mod.slug)}#${member}`, summary: summaryLine(entry.doc), mod };
}

export function remarkApiReference() {
  const inventory = loadInventory();
  const reference = loadPythonReference();

  return () => (tree, file) => {
    const fm = file.data?.astro?.frontmatter;
    const source = file.history?.[0] ?? file.path ?? '';
    if (!fm || !source.includes(`${path.sep}content${path.sep}docs${path.sep}modules${path.sep}`)) {
      return;
    }
    const pythonOnly = fm.api_surface === 'python-only';
    // On Python-only pages `rust_api` has always held the module's Python functions.
    const rustNames = pythonOnly ? [] : fm.rust_api ?? [];
    const pythonNames = [
      ...new Set([
        ...(fm.python_api ?? []),
        ...(pythonOnly ? (fm.rust_api ?? []).map((n) => `${fm.module}.${n}`) : []),
      ]),
    ];
    if (!rustNames.length && !pythonNames.length) return;

    const errors = [];
    const out = [heading(2, 'API reference')];

    if (pythonNames.length) {
      const rows = [];
      const modules = new Map();
      for (const qualified of pythonNames) {
        const hit = resolvePython(reference, qualified);
        if (hit.error) {
          errors.push(`python_api ${qualified}: ${hit.error}`);
          continue;
        }
        modules.set(hit.mod.slug, hit.mod);
        const row = [link(hit.url, [code(qualified)])];
        if (hit.summary) row.push(text(` — ${hit.summary}`));
        rows.push(row);
      }
      out.push(heading(3, 'Python'));
      out.push(
        para([
          text('Signatures, parameters, return values and errors for every function: '),
          ...[...modules.values()].flatMap((m, i) => [
            ...(i ? [text(', ')] : []),
            link(pythonModuleUrl(m.slug), [code(m.import)]),
          ]),
          text('.'),
        ])
      );
      out.push(list(rows));
    }

    if (rustNames.length) {
      const rows = [];
      const modules = new Set();
      for (const name of rustNames) {
        const hit = resolveRust(inventory, fm.module, name);
        if (hit.error || !hit.url) {
          errors.push(`rust_api ${name}: ${hit.error ?? 'no rustdoc page'}`);
          continue;
        }
        modules.add(hit.module);
        rows.push([link(hit.url, [code(name)])]);
      }
      out.push(heading(3, 'Rust'));
      out.push(
        para([
          text('Rustdoc for this commit, with every public item of '),
          ...[...modules].flatMap((m, i) => [
            ...(i ? [text(', ')] : []),
            link(rustModuleUrl(m), [code(m)]),
          ]),
          text(' (after a crates.io release the same pages are on docs.rs):'),
        ])
      );
      out.push(list(rows));
    }

    if (errors.length) {
      throw new Error(
        `${source}: API reference entries that resolve to nothing:\n  ${errors.join('\n  ')}\n` +
          'Fix the frontmatter, or regenerate the inventories ' +
          '(scripts/generate_api_inventory.py, scripts/generate_python_api_reference.py).'
      );
    }
    tree.children.push(...out);
  };
}

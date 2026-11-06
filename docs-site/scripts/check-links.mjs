import fs from 'node:fs';
import path from 'node:path';
import { parse } from 'parse5';

const root = path.resolve(process.cwd(), 'dist');
const site = 'https://open-quant.github.io';
const base = '/openquant';
if (!fs.existsSync(root)) {
  console.error('dist/ not found. Run the docs build and stage:rustdoc first.');
  process.exit(1);
}

const htmlFiles = [];
const walk = (dir) => {
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) walk(full);
    else if (entry.isFile() && entry.name.endsWith('.html')) htmlFiles.push(full);
  }
};
walk(root);

// Parse built HTML, not source Markdown. A parser handles quoted `>`, entities,
// unquoted attributes and script/comment text without inventing or hiding links.
function readPage(file) {
  const elements = [];
  const ids = new Set();
  const visit = (node) => {
    if (node.tagName) {
      const attrs = Object.fromEntries(node.attrs.map(({ name, value }) => [name, value]));
      if (attrs.id !== undefined) ids.add(attrs.id);
      if (node.tagName === 'a' && attrs.name !== undefined) ids.add(attrs.name);
      if (attrs.href !== undefined || node.tagName === 'meta') elements.push({ tag: node.tagName, attrs });
    }
    for (const child of node.childNodes ?? []) visit(child);
  };
  visit(parse(fs.readFileSync(file, 'utf8')));
  const rustdoc = elements.some(({ tag, attrs }) =>
    tag === 'meta' && attrs.name === 'generator' && attrs.content === 'rustdoc');
  return { elements, ids, rustdoc };
}
const pages = new Map(htmlFiles.map((file) => [file, readPage(file)]));
const missing = [];
function check(href, file, kind = 'link') {
  if (!href) return;
  const rel = path.relative(root, file).split(path.sep).join('/');
  const current = new URL(`${base}/${rel.replace(/index\.html$/, '')}`, site);
  let target;
  try { target = new URL(href, current); }
  catch { missing.push({ file, href, reason: `invalid ${kind} URL` }); return; }
  if (!['http:', 'https:'].includes(target.protocol) || target.origin !== site) return;
  const fail = (reason) => missing.push({ file, href, reason: `${kind}: ${reason}` });
  if (target.pathname !== base && !target.pathname.startsWith(`${base}/`)) {
    fail(`target is not under ${base}/`);
    return;
  }
  let decodedPath;
  try { decodedPath = decodeURIComponent(target.pathname.slice(base.length)); }
  catch { fail('invalid path encoding'); return; }
  let candidate = path.resolve(root, `.${decodedPath || '/'}`);
  if (candidate !== root && !candidate.startsWith(`${root}${path.sep}`)) {
    fail('target escapes built output'); return;
  }
  // Paths can contain dots (e.g. a versioned guide); inspect the filesystem
  // instead of assuming any extension means a file rather than a directory.
  if (fs.existsSync(candidate) && fs.statSync(candidate).isDirectory()) candidate = path.join(candidate, 'index.html');
  if (!fs.existsSync(candidate) || !fs.statSync(candidate).isFile()) {
    fail('file does not exist'); return;
  }
  if (target.hash && pages.has(candidate)) {
    let fragment;
    try { fragment = decodeURIComponent(target.hash.slice(1)); }
    catch { fail('invalid fragment encoding'); return; }
    // Browser text fragments do not refer to an HTML id. A preceding anchor does.
    fragment = fragment.split(':~:text=')[0];
    const page = pages.get(candidate);
    const rawFragment = target.hash.slice(1).split(':~:text=')[0];
    // Rustdoc's JavaScript resolves encoded implementation IDs and source line
    // ranges. Validate those against actual IDs rather than skipping rustdoc.
    const range = page.rustdoc && /\/api\/rust\/src\/.*\.rs\.html$/.test(target.pathname)
      ? fragment.match(/^(\d+)-(\d+)$/) : null;
    const validRange = range && Number(range[1]) <= Number(range[2]) &&
      page.ids.has(range[1]) && page.ids.has(range[2]);
    const validId = page.ids.has(fragment) || (page.rustdoc && page.ids.has(rawFragment));
    if (fragment && fragment.toLowerCase() !== 'top' && !validId && !validRange) {
      fail(`anchor #${fragment} does not exist`);
    }
  }
}

for (const [file, { elements }] of pages) {
  for (const { tag, attrs } of elements) {
    // The canonical /404/ metadata describes Astro's error route, emitted as
    // 404.html rather than 404/index.html. It is not a navigable href target.
    const canonical = tag === 'link' && attrs.rel?.toLowerCase().split(/\s+/).includes('canonical');
    if (attrs.href !== undefined && !canonical) check(attrs.href, file);
    if (tag === 'meta' && attrs['http-equiv']?.toLowerCase() === 'refresh') {
      const target = attrs.content?.match(/(?:^|;)\s*url\s*=\s*(.*)$/i)?.[1];
      if (target) check(target.replace(/^['"]|['"]$/g, ''), file, 'redirect');
    }
  }
}

if (missing.length) {
  console.error('Broken internal links found:');
  for (const { file, href, reason } of missing) {
    console.error(`- ${href} from ${path.relative(root, file)} (${reason})`);
  }
  process.exit(1);
}
console.log(`Link check passed (${htmlFiles.length} HTML files; paths, anchors and redirects checked).`);

import fs from 'node:fs';
import path from 'node:path';

const root = path.resolve(process.cwd(), 'dist');
if (!fs.existsSync(root)) {
  console.error('dist/ not found. Run bun run build first.');
  process.exit(1);
}

const htmlFiles = [];
const walk = (dir) => {
  for (const name of fs.readdirSync(dir)) {
    const full = path.join(dir, name);
    const stat = fs.statSync(full);
    if (stat.isDirectory()) {
      walk(full);
      continue;
    }
    if (name.endsWith('.html')) {
      htmlFiles.push(full);
    }
  }
};
walk(root);

const hrefRe = /href="([^"]+)"/g;
const metaRefreshRe = /<meta[^>]+http-equiv="refresh"[^>]*content="[^"]*?url=([^"'\s]+)"/gi;

// Map a site-absolute path onto the file the build actually emitted, or null
// if nothing is there.
const resolveTarget = (target) => {
  const noHash = target.split('#')[0].split('?')[0];
  if (!noHash.startsWith('/openquant')) {
    return null;
  }
  const rel = noHash.replace(/^\/openquant\/?/, '');
  if (rel.length === 0) {
    return path.join(root, 'index.html');
  }
  if (path.extname(rel)) {
    return path.join(root, rel);
  }
  return path.join(root, rel, 'index.html');
};

const missing = [];
for (const file of htmlFiles) {
  const text = fs.readFileSync(file, 'utf-8');
  let m;
  while ((m = hrefRe.exec(text)) !== null) {
    const href = m[1];
    if (!href || href.startsWith('http') || href.startsWith('mailto:') || href.startsWith('#')) {
      continue;
    }

    // Links outside the base are somebody else's problem.
    const candidate = resolveTarget(href);
    if (candidate && !fs.existsSync(candidate)) {
      missing.push({ source: file, href });
    }
  }

  // Redirect stubs are pages too: a meta-refresh target that misses the base
  // prefix 404s in production even though the alias itself returns 200.
  while ((m = metaRefreshRe.exec(text)) !== null) {
    const target = m[1];
    if (target.startsWith('http') || target.startsWith('#')) {
      continue;
    }
    const candidate = resolveTarget(target);
    if (!candidate) {
      missing.push({ source: file, href: `${target} (redirect target is not under /openquant)` });
      continue;
    }
    if (!fs.existsSync(candidate)) {
      missing.push({ source: file, href: `${target} (redirect target)` });
    }
  }
}

if (missing.length) {
  console.error('Broken internal links found:');
  for (const item of missing) {
    console.error(`- ${item.href} referenced from ${item.source}`);
  }
  process.exit(1);
}

console.log(`Link check passed (${htmlFiles.length} HTML files scanned, redirects included).`);

import fs from 'node:fs';
import path from 'node:path';

const root = path.resolve(process.cwd(), 'dist');
if (!fs.existsSync(root)) {
  console.error('dist/ not found. Run npm run build first.');
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
const missing = [];
for (const file of htmlFiles) {
  const text = fs.readFileSync(file, 'utf-8');
  let m;
  while ((m = hrefRe.exec(text)) !== null) {
    const href = m[1];
    if (!href || href.startsWith('http') || href.startsWith('mailto:') || href.startsWith('#')) {
      continue;
    }

    const noHash = href.split('#')[0].split('?')[0];
    if (!noHash.startsWith('/openquant')) {
      continue;
    }

    const rel = noHash.replace(/^\/openquant\/?/, '');
    let candidate;
    if (rel.length === 0) {
      candidate = path.join(root, 'index.html');
    } else if (path.extname(rel)) {
      candidate = path.join(root, rel);
    } else {
      candidate = path.join(root, rel, 'index.html');
    }
    if (!fs.existsSync(candidate)) {
      missing.push({ source: file, href });
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

console.log(`Link check passed (${htmlFiles.length} HTML files scanned).`);

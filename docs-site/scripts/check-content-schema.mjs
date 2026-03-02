import fs from 'node:fs';
import path from 'node:path';

const docsRoot = path.resolve(process.cwd(), 'src/content/docs');

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
    if (kv) out[kv[1]] = kv[2];
  }
  return out;
}

const files = walk(docsRoot);
const missing = [];

for (const file of files) {
  const text = fs.readFileSync(file, 'utf8');
  const fm = parseFrontmatter(text);
  const required = ['title', 'description', 'status', 'last_validated'];
  for (const key of required) {
    if (!(key in fm) || String(fm[key]).trim() === '') {
      missing.push(`${path.relative(process.cwd(), file)} missing ${key}`);
    }
  }
}

if (missing.length) {
  console.error('Content schema check failed:');
  for (const err of missing) console.error(`- ${err}`);
  process.exit(1);
}

console.log(`Content schema check passed (${files.length} docs files).`);

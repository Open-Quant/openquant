import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import test from 'node:test';

const siteRoot = fileURLToPath(new URL('../../', import.meta.url));
const root = path.resolve(siteRoot, '..');
const indexPath = path.join(siteRoot, 'dist/llms.txt');
const index = () => fs.readFileSync(indexPath, 'utf8');

test('build emits an agent index with both human and agent entry points', () => {
  assert.ok(fs.existsSync(indexPath), 'Astro must emit dist/llms.txt');
  const text = index();
  assert.match(text, /^# OpenQuant\n/);
  assert.match(text, /https:\/\/open-quant.github.io\/openquant\/quickstart\//);
  assert.match(text, /https:\/\/open-quant.github.io\/openquant\/agents\//);
});

test('agent index identifies its exact source revision', () => {
  assert.ok(fs.existsSync(indexPath), 'Astro must emit dist/llms.txt');
  const revision = execFileSync('git', ['rev-parse', 'HEAD'], { cwd: root, encoding: 'utf8' }).trim();
  assert.ok(index().includes(`Source revision: ${revision}`));
  assert.ok(index().includes(`https://github.com/Open-Quant/openquant/tree/${revision}`));
});

test('every generated site URL resolves inside the staged build', () => {
  assert.ok(fs.existsSync(indexPath), 'Astro must emit dist/llms.txt');
  const urls = [...index().matchAll(/\]\((https:\/\/open-quant.github.io\/openquant[^\s)]*)\)/g)].map((m) => m[1]);
  assert.ok(urls.length > 60, 'Index must include the documentation collection, not just a few hand-picked URLs');
  assert.equal(urls.length, new Set(urls).size, 'Do not duplicate entry points in the generated list');
  for (const url of urls) {
    const rel = new URL(url).pathname.replace(/^\/openquant\/?/, '');
    assert.ok(fs.existsSync(path.join(siteRoot, 'dist', rel, 'index.html')), `Missing index target: ${url}`);
  }
});

test('agent index states review status without copying API signatures or example outputs', () => {
  assert.ok(fs.existsSync(indexPath), 'Astro must emit dist/llms.txt');
  assert.match(index(), /\[authored\]/);
  assert.match(index(), /\[generated\]/);
  assert.match(index(), /Read exact signatures/);
  assert.ok(!index().includes('```'), 'Index should route to canonical content, not duplicate executable examples');
});

test('agent index distinguishes a modified checkout from its committed revision', () => {
  assert.ok(fs.existsSync(indexPath), 'Astro must emit dist/llms.txt');
  const dirty = execFileSync('git', ['status', '--porcelain', '--untracked-files=normal'], { cwd: root, encoding: 'utf8' }).trim();
  assert.ok(index().includes(`Working tree: ${dirty ? 'modified' : 'clean'}`));
});

test('agent index includes every authored content entry once', () => {
  assert.ok(fs.existsSync(indexPath), 'Astro must emit dist/llms.txt');
  const walk = (dir) => fs.readdirSync(dir, { withFileTypes: true }).flatMap((entry) =>
    entry.isDirectory() ? walk(path.join(dir, entry.name)) :
      /^(?!_).*\.mdx?$/.test(entry.name) ? [entry.name] : []);
  const text = index();
  const guides = text.split('## Start here')[1].split('## API reference')[0] +
    text.split('## Guides and module explanations')[1].split('## Repository sources')[0];
  assert.equal([...guides.matchAll(/^- \[/gm)].length, walk(path.join(siteRoot, 'src/content/docs')).length);
});

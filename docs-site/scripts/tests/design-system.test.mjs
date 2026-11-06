import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { parse } from 'parse5';
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const html = () => parse(fs.readFileSync(path.join(root, 'dist/index.html'), 'utf8'));
const walk = (node) => [node, ...(node.childNodes || []).flatMap(walk)];
const attr = (node, name) => node.attrs?.find((item) => item.name === name)?.value;
const text = (node) => walk(node).filter((item) => item.nodeName === '#text').map((item) => item.value).join(' ');

test('research map exposes four ordered stages with real module links without JavaScript', () => {
  const nodes = walk(html());
  const map = nodes.find((node) => node.tagName === 'ol' && attr(node, 'class')?.includes('oq-research-map'));
  assert.ok(map, 'Homepage should have a semantic ordered research map');
  assert.equal(attr(map, 'role'), 'list', 'Preserve list semantics when visual markers are removed');
  const stages = (map.childNodes || []).filter((node) => node.tagName === 'li');
  assert.equal(stages.length, 4);
  for (const [index, name] of ['Prepare', 'Label', 'Validate', 'Evaluate'].entries()) {
    assert.ok(text(stages[index]).includes(name));
    const links = walk(stages[index]).filter((node) => node.tagName === 'a');
    assert.ok(links.length > 0);
    for (const link of links) {
      const href = attr(link, 'href');
      assert.ok(href.startsWith('/openquant/modules/'), href);
      assert.ok(fs.existsSync(path.join(root, 'dist', href.replace('/openquant/', ''), 'index.html')), href);
    }
  }
  assert.ok(nodes.some((node) => node.tagName === 'figcaption' && text(node).includes('not a performance result')));
});

test('diagram illustrations are decorative; their meaning is available in stage text', () => {
  const map = walk(html()).find((node) => attr(node, 'class')?.includes('oq-research-map'));
  assert.ok(map, 'Research map is present');
  const figures = walk(map).filter((node) => node.tagName === 'svg');
  assert.equal(figures.length, 4);
  for (const figure of figures) assert.equal(attr(figure, 'aria-hidden'), 'true');
});

test('logo stays identical across site, favicon and identity source', () => {
  const mark = fs.readFileSync(path.join(root, 'src/assets/openquant-mark.svg'), 'utf8');
  assert.equal(fs.readFileSync(path.join(root, 'public/favicon.svg'), 'utf8'), mark);
  assert.equal(fs.readFileSync(path.join(root, '../docs/design/identity/openquant-mark.svg'), 'utf8'), mark);
  assert.match(mark, /<title id="title">OpenQuant/);
  assert.match(mark, /<desc id="desc">/);
});

test('bold and table type use loaded semibold instead of synthetic bold', () => {
  const css = fs.readFileSync(path.join(root, 'src/styles/starlight.css'), 'utf8');
  assert.match(css, /\.sl-markdown-content :is\(strong, b\)[^{]*\{[^}]*font-weight:\s*600/s);
  assert.match(css, /\.sl-markdown-content table[^}]*font-family:\s*var\(--oq-sans\)/s);
});

test('data cache paths render literally instead of as inline math', () => {
  const page = parse(fs.readFileSync(path.join(root, 'dist/modules/data/index.html'), 'utf8'));
  const row = walk(page).find((node) => node.tagName === 'tr'
    && node.childNodes?.some((cell) => cell.tagName === 'td' && text(cell).trim() === 'cache_dir'));
  assert.ok(row, 'Data guide should document the cache_dir parameter');
  const nodes = walk(row);
  assert.ok(!nodes.some((node) => node.tagName === 'math'
    || attr(node, 'class')?.split(/\s+/).includes('katex')), 'Cache paths must not become math nodes');
  const code = nodes.filter((node) => node.tagName === 'code').map(text);
  for (const literal of ['$OPENQUANT_DATA_CACHE', '$XDG_CACHE_HOME/openquant/data', '~/.cache/openquant/data']) {
    assert.ok(code.includes(literal), `Cache path should remain literal inline code: ${literal}`);
  }
});

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

test('Python docstring sections nest below their module, function, class or method', () => {
  const apiRoot = path.join(root, 'dist/api/python');
  let sections = 0;
  let methodSections = 0;
  for (const entry of fs.readdirSync(apiRoot, { withFileTypes: true })) {
    if (!entry.isDirectory()) continue;
    const nodes = walk(parse(fs.readFileSync(path.join(apiRoot, entry.name, 'index.html'), 'utf8')));
    let parentDepth = 0;
    for (const node of nodes) {
      if (!/^h[1-6]$/.test(node.tagName ?? '')) continue;
      const depth = Number(node.tagName[1]);
      if (attr(node, 'class')?.split(/\s+/).includes('oq-doc-section')) {
        assert.equal(depth, parentDepth + 1, `${entry.name}: ${text(node)}`);
        sections++;
        if (parentDepth === 4) methodSections++;
      } else parentDepth = depth;
    }
  }
  assert.ok(sections > 0, 'Check real generated docstring sections');
  assert.ok(methodSections > 0, 'Check method sections as well as function sections');
});

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


test('Python reference links back to guides and exposes class members without JavaScript', () => {
  const ref = JSON.parse(fs.readFileSync(path.join(root, 'src/data/pythonApiReference.json'), 'utf8'));
  let classes = 0;
  for (const mod of ref.modules) {
    const nodes = walk(parse(fs.readFileSync(path.join(root, 'dist/api/python', mod.slug, 'index.html'), 'utf8')));
    const context = nodes.find((n) => n.tagName === 'nav' && attr(n, 'aria-label') === 'Related documentation');
    assert.ok(context, mod.import);
    assert.ok(walk(context).some((n) => attr(n, 'href') === '/openquant/module-reference/api-task-guide/'
      && text(n).trim() === 'Find an API by task'), mod.import);
    for (const klass of mod.classes.filter((c) => c.methods.length)) {
      const index = nodes.find((n) => n.tagName === 'nav' && attr(n, 'aria-label') === `${klass.name} members`);
      assert.ok(index, `${mod.import}.${klass.name}`);
      for (const method of klass.methods) {
        const id = `${klass.name}.${method.name}`;
        assert.ok(walk(index).some((n) => attr(n, 'href') === `#${id}`), id);
        assert.ok(nodes.some((n) => attr(n, 'id') === id), id);
      }
      classes++;
    }
  }
  assert.ok(classes > 0);
  const evaluation = walk(parse(fs.readFileSync(path.join(root, 'dist/api/python/evaluation/index.html'), 'utf8')));
  assert.ok(evaluation.some((n) => attr(n, 'href') === '/openquant/modules/evaluation/'));
});

test('Python module table has keyboard access and explicit table semantics', () => {
  const nodes = walk(parse(fs.readFileSync(path.join(root, 'dist/api/python/index.html'), 'utf8')));
  const scroll = nodes.find((n) => attr(n, 'class') === 'oq-table-scroll');
  assert.equal(attr(scroll, 'tabindex'), '0');
  assert.equal(attr(scroll, 'role'), 'region');
  assert.ok(attr(scroll, 'aria-label'));
  const table = walk(scroll).find((n) => n.tagName === 'table');
  assert.ok(walk(table).some((n) => n.tagName === 'caption' && text(n).trim()));
  assert.equal(walk(table).filter((n) => attr(n, 'scope') === 'col').length, 4);
  assert.ok(walk(table).some((n) => attr(n, 'scope') === 'row'));
});

test('sidebar disclosure respects reduced motion', () => {
  const css = fs.readFileSync(path.join(root, 'src/styles/starlight.css'), 'utf8');
  assert.match(css, /@media\s*\(prefers-reduced-motion:\s*reduce\)\s*\{\s*\.sidebar-content\s+\.caret\s*\{\s*transition:\s*none/s);
});

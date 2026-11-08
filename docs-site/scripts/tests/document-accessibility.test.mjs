import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { parse } from 'parse5';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const walk = (node) => [node, ...(node.childNodes || []).flatMap(walk)];
const attr = (node, name) => node.attrs?.find((item) => item.name === name)?.value;
const hasClass = (node, name) => attr(node, 'class')?.split(/\s+/).includes(name);
const text = (node) => walk(node).filter((item) => item.nodeName === '#text').map((item) => item.value).join(' ');
const read = (route) => walk(parse(fs.readFileSync(path.join(root, 'dist', route, 'index.html'), 'utf8')));

test('documentation status is a named static landmark, including custom banners', () => {
  for (const route of ['', 'modules/data', 'api/python/data']) {
    const nodes = read(route);
    const banners = nodes.filter((node) => hasClass(node, 'sl-banner'));
    assert.equal(banners.length, 1, route);
    const banner = banners[0];
    assert.equal(banner.tagName, 'aside', route);
    assert.equal(attr(banner, 'aria-label'), 'Documentation status', route);
    assert.equal(attr(banner, 'aria-live'), undefined, 'Static status should not announce as a live update');
    assert.ok(walk(banner).some((node) => hasClass(node, 'doc-status') && text(node).trim()), route);
    assert.ok(text(banner).trim().length > 10, route);
  }
});

test('real Python docstring examples expose named keyboard scroll targets', () => {
  const ref = JSON.parse(fs.readFileSync(path.join(root, 'src/data/pythonApiReference.json'), 'utf8'));
  let examples = 0;
  for (const mod of ref.modules) {
    for (const container of read(`api/python/${mod.slug}`).filter((node) => hasClass(node, 'oq-doc'))) {
      for (const block of walk(container).filter((node) => node.tagName === 'pre')) {
        assert.equal(attr(block, 'tabindex'), '0', mod.import);
        assert.equal(attr(block, 'role'), 'group', mod.import);
        assert.equal(attr(block, 'aria-label'), 'Code example', mod.import);
        assert.ok(walk(block).some((node) => node.tagName === 'code'), mod.import);
        examples++;
      }
    }
  }
  assert.ok(examples > 0, 'Exercise actual generated examples rather than an empty selection');
  const css = fs.readFileSync(path.join(root, 'src/styles/starlight.css'), 'utf8');
  assert.match(css, /\.oq-doc pre\s*\{[^}]*overflow-x:\s*auto/s);
  assert.match(css, /\.oq-doc pre:focus-visible\s*\{[^}]*outline:\s*3px solid var\(--oq-accent\)/s);
});


test('all Python reference entry points link to the task-oriented API guide', () => {
  const ref = JSON.parse(fs.readFileSync(path.join(root, 'src/data/pythonApiReference.json'), 'utf8'));
  for (const route of ['api/python', ...ref.modules.map((mod) => `api/python/${mod.slug}`)]) {
    const main = read(route).find((node) => node.tagName === 'main');
    assert.ok(main, `${route}: main content must exist`);
    // The sidebar also links to this guide; the reference body must offer its own entry point.
    const links = walk(main).filter((node) => node.tagName === 'a'
      && attr(node, 'href') === '/openquant/module-reference/api-task-guide/'
      && text(node).trim() === 'Find an API by task');
    assert.equal(links.length, 1, `${route}: provide one explicit task entry point`);
  }
});


test('human and agent entry pages expose connected workflow choices in main content', () => {
  for (const route of ['', 'agents']) {
    const main = read(route).find((node) => node.tagName === 'main');
    assert.ok(main, `${route || 'overview'}: main content must exist`);
    for (const workflow of ['model-selection-holdout', 'multi-asset-allocation', 'point-in-time-data']) {
      const links = walk(main).filter((node) => node.tagName === 'a'
        && attr(node, 'href') === `/openquant/workflows/${workflow}/`);
      assert.equal(links.length, 1, `${route || 'overview'}: one explicit ${workflow} task choice`);
      assert.ok(text(links[0]).trim(), `${workflow}: visible link text`);
    }
  }
});

test('walkthrough code fences carry deferred scroll names without naming generic pre elements', () => {
  for (const route of ['workflows/model-selection-holdout', 'workflows/multi-asset-allocation', 'workflows/point-in-time-data']) {
    const containers = read(route).filter((node) => hasClass(node, 'expressive-code'));
    const blocks = containers.flatMap((node) => walk(node).filter((child) => child.tagName === 'pre'));
    assert.ok(blocks.length >= 2, `${route}: inspect code and displayed output`);
    for (const block of blocks) {
      assert.ok(attr(block, 'data-oq-scroll-label')?.trim(), route);
      assert.equal(attr(block, 'aria-label'), undefined, 'Runtime assigns a name only after EC creates a region');
      assert.equal(attr(block, 'role'), undefined, 'Overflow detection stays with Expressive Code');
      assert.equal(attr(block, 'tabindex'), undefined, 'Do not force extra keyboard stops before overflow');
    }
  }
});

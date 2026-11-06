import assert from 'node:assert/strict';
import { mkdtempSync, mkdirSync, writeFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import test from 'node:test';

const script = fileURLToPath(new URL('../check-links.mjs', import.meta.url));
function check(t, pages) {
  const cwd = mkdtempSync(path.join(tmpdir(), 'openquant-links-'));
  t.after(() => rmSync(cwd, { recursive: true, force: true }));
  for (const [name, content] of Object.entries(pages)) {
    const file = path.join(cwd, 'dist', name);
    mkdirSync(path.dirname(file), { recursive: true });
    writeFileSync(file, content);
  }
  return spawnSync(process.execPath, [script], { cwd, encoding: 'utf8' });
}

test('accepts site-root, relative, encoded and same-page anchors', (t) => {
  const result = check(t, {
    'index.html': '<a href="/openquant/guide/#caf%C3%A9">Guide</a>',
    'guide/index.html': '<h2 id="café">Title</h2><a href="#café">Here</a><a href="../">Home</a><a href="https://elsewhere.example/missing#anchor">External</a><a href="mailto:hello@example.org">Email</a>',
  });
  assert.equal(result.status, 0, result.stderr);
});

test('rejects an internal link that escapes the deployment base', (t) => {
  const result = check(t, { 'index.html': '<a href="/quickstart/">Quickstart</a>' });
  assert.equal(result.status, 1);
  assert.match(result.stderr, /\/quickstart\//);
});

test('rejects a path that only shares the deployment base prefix', (t) => {
  const result = check(t, { 'index.html': '<a href="/openquant-other/">Other</a>' });
  assert.equal(result.status, 1);
});

test('rejects a missing anchor on an existing page', (t) => {
  const result = check(t, {
    'index.html': '<a href="/openquant/guide/#missing">Guide</a>',
    'guide/index.html': '<h2 id="present">Title</h2>',
  });
  assert.equal(result.status, 1);
  assert.match(result.stderr, /#missing/);
});

test('rejects a missing same-page anchor in single-quoted attributes', (t) => {
  const result = check(t, { 'index.html': "<a href='#missing'>Missing</a>" });
  assert.equal(result.status, 1);
});

test('checks relative paths from the containing page', (t) => {
  const result = check(t, { 'guide/index.html': '<a href="../missing/">Missing</a>' });
  assert.equal(result.status, 1);
});

test('checks absolute links to this site', (t) => {
  const result = check(t, { 'index.html': '<a href="https://open-quant.github.io/openquant/missing/">Missing</a>' });
  assert.equal(result.status, 1);
});

test('checks redirect destinations including fragments', (t) => {
  const result = check(t, {
    'old/index.html': '<meta http-equiv="refresh" content="0;url=/openquant/guide/#missing">',
    'guide/index.html': '<h2 id="present">Title</h2>',
  });
  assert.equal(result.status, 1);
});

test('accepts file downloads, query strings and external protocol-relative links', (t) => {
  const result = check(t, {
    'index.html': '<a href="/openquant/llms.txt?v=1">Index</a><a href="//example.org/file">External</a><a href="?q=hello&amp;page=2#top">Here</a><h1 id="top">Hello</h1>',
    'llms.txt': 'Index',
  });
  assert.equal(result.status, 0, result.stderr);
});

test('does not treat canonical metadata as a navigable link', (t) => {
  const result = check(t, { '404.html': '<link rel="canonical" href="https://open-quant.github.io/openquant/404/">' });
  assert.equal(result.status, 0, result.stderr);
});

test('still checks local stylesheet and icon href targets', (t) => {
  const result = check(t, { 'index.html': '<link rel="stylesheet" href="/openquant/missing.css"><link rel="icon" href="/openquant/missing.svg">' });
  assert.equal(result.status, 1);
  assert.match(result.stderr, /missing.css/);
  assert.match(result.stderr, /missing.svg/);
});

test('parses greater-than characters inside quoted attributes', (t) => {
  const result = check(t, { 'index.html': '<a title="x > y" href="/openquant/missing/">Missing</a>' });
  assert.equal(result.status, 1);
});

test('supports HTML entities, unquoted attributes and special top fragments', (t) => {
  const result = check(t, { 'index.html': '<h2 title="x > y" id="a&amp;b">Title</h2><a href="#a&amp;b">Here</a><a href=#top>Top</a><a href=#TOP>Top</a>' });
  assert.equal(result.status, 0, result.stderr);
});

test('does not scan comments or script strings for HTML links', (t) => {
  const result = check(t, { 'index.html': '<!-- <a href="/openquant/missing/"> --> <script>const sample = \'<a href="/openquant/missing/">\';</script>' });
  assert.equal(result.status, 0, result.stderr);
});

test('supports rustdoc encoded implementation anchors', (t) => {
  const result = check(t, { 'api/rust/openquant/index.html': '<meta name="generator" content="rustdoc"><h2 id="impl-Borrow%3CT%3E-for-T">Impl</h2><a href="#impl-Borrow%3CT%3E-for-T">Impl</a>' });
  assert.equal(result.status, 0, result.stderr);
});

test('checks rustdoc source line ranges against both endpoints', (t) => {
  const result = check(t, {
    'index.html': '<a href="/openquant/api/rust/src/openquant/lib.rs.html#1-2">Source</a>',
    'api/rust/src/openquant/lib.rs.html': '<meta name="generator" content="rustdoc"><a id=1></a><a id=2></a>',
  });
  assert.equal(result.status, 0, result.stderr);
});

test('rejects out-of-range rustdoc source endpoints', (t) => {
  const result = check(t, {
    'index.html': '<a href="/openquant/api/rust/src/openquant/lib.rs.html#1-9">Source</a>',
    'api/rust/src/openquant/lib.rs.html': '<meta name="generator" content="rustdoc"><a id=1></a><a id=2></a>',
  });
  assert.equal(result.status, 1);
});

test('does not invent line range support on ordinary guides', (t) => {
  const result = check(t, { 'index.html': '<a href="#1-2">Range</a><p id=1>One</p><p id=2>Two</p>' });
  assert.equal(result.status, 1);
});

import test from 'node:test';
import assert from 'node:assert/strict';
import vm from 'node:vm';
import { ExpressiveCodeEngine } from '@expressive-code/core';
import { pluginFrames } from '@expressive-code/plugin-frames';
import { toHtml } from '@expressive-code/core/hast';
import { parse } from 'parse5';
import { codeScrollLabels } from '../expressive-code-accessibility.mjs';

const engine = new ExpressiveCodeEngine({ plugins: [pluginFrames(), codeScrollLabels()] });
const modules = await engine.getJsModules();
const overflowScript = modules.find((source) => source.includes('.expressive-code pre > code') && source.includes('ResizeObserver'));
assert.ok(overflowScript, 'Exercise the installed Expressive Code overflow script, not a reimplementation');
const labelScript = codeScrollLabels().jsModules[0];
const walk = (node) => [node, ...(node.children || []).flatMap(walk)];

// Deliberately bounded DOM/observer contract harness: dimensions are supplied, not laid out.
// Actual installed overflow JS and shipped label JS run unchanged in the same VM context.
function harness() {
  const observers = [];
  const resizers = [];
  const timers = new Map();
  const listeners = new Map();
  let timerId = 0;
  const notify = (record) => {
    for (const observer of observers) {
      if (!observer.root?.contains(record.target)) continue;
      if (record.type === 'attributes' && (!observer.options.attributes ||
        !observer.options.attributeFilter.includes(record.attributeName))) continue;
      if (record.type === 'childList' && !observer.options.childList) continue;
      observer.records.push(record);
    }
  };
  class Element {
    constructor(tag, attrs = {}) {
      this.tag = tag; this.attrs = new Map(Object.entries(attrs)); this.children = [];
      this.scrollWidth = 80; this.clientWidth = 100;
    }
    getAttribute(name) { return this.attrs.get(name) ?? null; }
    hasAttribute(name) { return this.attrs.has(name); }
    setAttribute(name, value) {
      this.attrs.set(name, String(value));
      notify({ type: 'attributes', target: this, attributeName: name });
    }
    removeAttribute(name) {
      if (this.attrs.delete(name)) notify({ type: 'attributes', target: this, attributeName: name });
    }
    contains(node) { return node === this || this.children.some((child) => child.contains(node)); }
    inCode() { return this.getAttribute('class') === 'expressive-code' || this.parentElement?.inCode(); }
    matches(selector) {
      if (selector === '.expressive-code pre[data-oq-scroll-label]') {
        return this.tag === 'pre' && this.hasAttribute('data-oq-scroll-label') && this.inCode();
      }
      if (selector === '.expressive-code pre > code') {
        return this.tag === 'code' && this.parentElement?.tag === 'pre' && this.inCode();
      }
      throw new Error(`Unexpected selector: ${selector}`);
    }
    querySelectorAll(selector) { return this.children.flatMap(walk).filter((node) => node.matches(selector)); }
    append(child) {
      child.parentElement = this; this.children.push(child);
      notify({ type: 'childList', target: this, addedNodes: [child] });
      return child;
    }
  }
  class MutationObserver {
    constructor(callback) { this.callback = callback; this.records = []; observers.push(this); }
    observe(root, options) { this.root = root; this.options = options; }
    disconnect() { this.root = undefined; this.records = []; }
  }
  class ResizeObserver {
    constructor(callback) { this.callback = callback; this.nodes = new Set(); resizers.push(this); }
    observe(node) { this.nodes.add(node); }
  }
  const document = {
    body: new Element('body'), readyState: 'complete',
    querySelectorAll(selector) { return this.body.querySelectorAll(selector); },
    addEventListener(name, fn) {
      if (!listeners.has(name)) listeners.set(name, new Set());
      listeners.get(name).add(fn);
    },
  };
  const schedule = (fn) => { timers.set(++timerId, fn); return timerId; };
  const context = vm.createContext({ document, MutationObserver, ResizeObserver,
    window: { requestIdleCallback: schedule, cancelIdleCallback: (id) => timers.delete(id) },
    setTimeout: schedule, clearTimeout: (id) => timers.delete(id),
    console: { error: (...args) => { throw new Error(args.join(' ')); } },
  });
  const flush = () => {
    for (let pass = 0; pass < 100; pass++) {
      let pending = false;
      for (const observer of observers) {
        if (observer.records.length) {
          pending = true; observer.callback(observer.records.splice(0));
        }
      }
      if (timers.size) {
        pending = true;
        for (const [id, fn] of [...timers]) { if (timers.delete(id)) fn(); }
      }
      if (!pending) return;
    }
    throw new Error('Observer/timer callbacks failed to settle');
  };
  const pre = (attrs = {}) => {
    const node = new Element('pre', { 'data-oq-scroll-label': 'Python code', ...attrs });
    node.append(new Element('code')); return node;
  };
  return { Element, document, observers, listeners, pre, flush,
    run: (source) => { vm.runInContext(source, context); flush(); },
    event: (name) => { for (const fn of listeners.get(name) ?? []) fn(); flush(); },
    resize: (node, width) => {
      node.scrollWidth = width;
      for (const observer of resizers) if (observer.nodes.has(node)) observer.callback([{ target: node }]);
      flush();
    },
  };
}

test('supported render hook stores escaped title/language metadata without naming generic pre', async () => {
  for (const [language, title, expected] of [
    ['python', 'fit.py', 'fit.py — Python code'], ['text', '', 'Text block'],
    ['rust', 'a <script> & "quoted"', 'a <script> & "quoted" — Rust code'],
  ]) {
    const result = await engine.render({ code: 'example', language,
      ...(title === 'fit.py' ? { meta: 'title="fit.py"' } : { props: { title } }) });
    const pre = walk(result.renderedGroupAst).find((node) => node.tagName === 'pre');
    assert.equal(pre.properties['data-oq-scroll-label'], expected);
    for (const attr of ['role', 'tabIndex', 'ariaLabel', 'aria-label']) assert.equal(pre.properties[attr], undefined);
    const nodes = (node) => [node, ...(node.childNodes || []).flatMap(nodes)];
    const parsed = nodes(parse(toHtml(result.renderedGroupAst)));
    const renderedPre = parsed.find((node) => node.tagName === 'pre');
    assert.equal(renderedPre.attrs.find((attr) => attr.name === 'data-oq-scroll-label').value, expected);
    assert.ok(!parsed.some((node) => node.tagName === 'script'), 'Title content cannot create markup');
  }
});

test('actual EC overflow script gains and loses a name with its region role, in either script order', () => {
  for (const order of [[overflowScript, labelScript], [labelScript, overflowScript]]) {
    const h = harness();
    const frame = h.document.body.append(new h.Element('div', { class: 'expressive-code' }));
    const pre = frame.append(h.pre());
    order.forEach(h.run);
    assert.equal(pre.getAttribute('aria-label'), null);
    h.resize(pre, 200);
    assert.equal(pre.getAttribute('role'), 'region');
    assert.equal(pre.getAttribute('tabindex'), '0');
    assert.equal(pre.getAttribute('aria-label'), 'Python code');
    h.resize(pre, 80);
    assert.equal(pre.getAttribute('role'), null);
    assert.equal(pre.getAttribute('tabindex'), null);
    assert.equal(pre.getAttribute('aria-label'), null);
  }
});

test('authored names win, and removal/role changes only clear names owned by this plugin', () => {
  const h = harness();
  const frame = h.document.body.append(new h.Element('div', { class: 'expressive-code' }));
  const pre = frame.append(h.pre({ role: 'region' }));
  h.run(labelScript);
  assert.equal(pre.getAttribute('aria-label'), 'Python code');
  pre.setAttribute('aria-label', 'Authored name'); h.flush();
  pre.removeAttribute('role'); h.flush();
  assert.equal(pre.getAttribute('aria-label'), 'Authored name');
  pre.setAttribute('role', 'region'); pre.removeAttribute('aria-label'); h.flush();
  assert.equal(pre.getAttribute('aria-label'), 'Python code');
  pre.setAttribute('aria-labelledby', 'named-heading'); h.flush();
  assert.equal(pre.getAttribute('aria-label'), null);
  pre.removeAttribute('role'); h.flush();
  assert.equal(pre.getAttribute('aria-labelledby'), 'named-heading');
  pre.setAttribute('role', 'region'); pre.removeAttribute('aria-labelledby'); h.flush();
  assert.equal(pre.getAttribute('aria-label'), 'Python code');
  pre.removeAttribute('aria-label'); h.flush();
  assert.equal(pre.getAttribute('aria-label'), 'Python code');
});

test('inserted blocks, replacement bodies and repeated installation retain one naming observer', () => {
  const h = harness();
  const unrelated = h.document.body.append(h.pre({ role: 'region' }));
  h.run(labelScript);
  assert.equal(unrelated.getAttribute('aria-label'), null, 'Do not name blocks outside Expressive Code');
  const frame = h.document.body.append(new h.Element('div', { class: 'expressive-code' }));
  const direct = frame.append(h.pre({ role: 'region' })); h.flush();
  assert.equal(direct.getAttribute('aria-label'), 'Python code');
  const subtree = new h.Element('div', { class: 'expressive-code' });
  const nested = subtree.append(h.pre({ role: 'region' }));
  h.document.body.append(subtree); h.flush();
  assert.equal(nested.getAttribute('aria-label'), 'Python code');
  h.document.body = new h.Element('body');
  const replacement = h.document.body.append(new h.Element('div', { class: 'expressive-code' }));
  const next = replacement.append(h.pre({ role: 'region', 'aria-label': 'Preserved' }));
  h.event('astro:page-load'); h.run(labelScript); h.event('astro:page-load');
  assert.equal(h.observers.length, 1);
  assert.equal(h.listeners.get('astro:page-load').size, 1);
  assert.equal(next.getAttribute('aria-label'), 'Preserved');
  next.removeAttribute('aria-label'); h.flush();
  assert.equal(next.getAttribute('aria-label'), 'Python code');
});

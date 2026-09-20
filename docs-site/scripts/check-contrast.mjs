#!/usr/bin/env node
/**
 * check:contrast — WCAG contrast over the identity's colour pairs.
 *
 * Reads the --oq-* tokens straight out of src/styles/starlight.css (both themes), so the gate
 * measures what ships rather than a copy of it. Every pair that carries text must reach AA
 * (4.5:1); hairlines are not text but must stay visible (1.5:1 — the old theme's 1.34:1 borders
 * were the defect that prompted this). It also fails if the syntax themes in
 * src/styles/code-themes.mjs have drifted from the tokens.
 */
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { monographDark, monographLight } from '../src/styles/code-themes.mjs';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const css = fs.readFileSync(path.join(root, 'src/styles/starlight.css'), 'utf8');

const TEXT_PAIRS = [
  ['text', 'ground'],
  ['text', 'surface'],
  ['muted', 'ground'],
  ['muted', 'surface'],
  ['accent', 'ground'],
  ['accent', 'surface'],
  ['ground', 'accent'], // status pill, primary button
];
// Hairlines are drawn on the page ground; where one borders a code block it still has ground on
// its outer side. tokens.md specifies the pair against ground only.
const RULE_PAIRS = [['rule', 'ground']];

function tokens(selector) {
  const start = css.indexOf(selector);
  if (start < 0) throw new Error(`selector not found: ${selector}`);
  const block = css.slice(css.indexOf('{', start), css.indexOf('}', start));
  const out = {};
  for (const m of block.matchAll(/--oq-(\w+):\s*(#[0-9a-fA-F]{6})/g)) out[m[1]] = m[2].toLowerCase();
  return out;
}

const channel = (v) => (v <= 0.03928 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4);
const luminance = (hex) => {
  const [r, g, b] = [1, 3, 5].map((i) => channel(parseInt(hex.slice(i, i + 2), 16) / 255));
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
};
const contrast = (a, b) => {
  const [hi, lo] = [luminance(a), luminance(b)].sort((x, y) => y - x);
  return (hi + 0.05) / (lo + 0.05);
};

const themes = {
  light: { tokens: tokens(":root[data-theme='light'] {"), code: monographLight },
  dark: { tokens: tokens(":root[data-theme='dark'] {"), code: monographDark },
};

let failed = false;
for (const [name, { tokens: t, code }] of Object.entries(themes)) {
  for (const key of ['ground', 'surface', 'text', 'muted', 'accent', 'rule']) {
    if (!t[key]) throw new Error(`${name}: token --oq-${key} missing from starlight.css`);
  }
  const check = (pairs, min, kind) => {
    for (const [fg, bg] of pairs) {
      const ratio = contrast(t[fg], t[bg]);
      const ok = ratio >= min;
      if (!ok) failed = true;
      console.log(`  ${ok ? 'ok  ' : 'FAIL'} ${name.padEnd(5)} ${kind} ${fg} on ${bg}`.padEnd(44), `${ratio.toFixed(2)}:1 (min ${min})`);
    }
  };
  check(TEXT_PAIRS, 4.5, 'text');
  check(RULE_PAIRS, 1.5, 'rule');

  const used = new Set([code.colors['editor.background'], code.colors['editor.foreground'], ...code.tokenColors.map((c) => c.settings.foreground)]);
  const allowed = new Set([t.surface, t.text, t.muted, t.accent]);
  for (const hex of used) {
    if (!allowed.has(hex.toLowerCase())) {
      failed = true;
      console.log(`  FAIL ${name} code theme uses ${hex}, which is not a token in starlight.css`);
    }
  }
}

if (failed) {
  console.error('\ncheck:contrast: FAILED');
  process.exit(1);
}
console.log('\ncheck:contrast: OK — every text pair meets AA in both themes.');

import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import test from 'node:test';

const checker = new URL('../check-doc-python-examples.mjs', import.meta.url);
const root = fs.mkdtempSync(path.join(os.tmpdir(), 'openquant-doc-python-'));

function run(python) {
  return spawnSync(process.execPath, [checker.pathname], {
    env: { ...process.env, DOC_PYTHON: python }, encoding: 'utf8', timeout: 15000,
  });
}

test('explicit missing interpreter fails without silently choosing another environment', () => {
  const missing = path.join(root, 'missing-python');
  const result = run(missing);
  assert.equal(result.status, 1);
  assert.match(result.stderr, /no interpreter could/);
  assert.equal(result.stderr.split('\n').find(line => line.startsWith('Tried:')), `Tried: ${missing}`);
});

test('explicit interpreter with failed package import does not fall back', () => {
  const executable = path.join(root, 'python with spaces');
  fs.writeFileSync(executable, '#!/bin/sh\nexit 1\n', { mode: 0o755 });
  const result = run(executable);
  assert.equal(result.status, 1);
  assert.equal(result.stderr.split('\n').find(line => line.startsWith('Tried:')), `Tried: ${executable}`);
});

test.after(() => fs.rmSync(root, { recursive: true, force: true }));

#!/usr/bin/env node
/** Execute the ETF program and compare all displayed output, including the roll example.
 * This is separate from check:examples, which deliberately compiles without executing.
 */
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawnSync } from 'node:child_process';

export function assertDocumentedOutput(page, stdout) {
  const normalize = (text) => text.replaceAll('\r\n', '\n').replace(/\n$/, '');
  const blocks = [];
  let fence = null;
  for (const line of page.replaceAll('\r\n', '\n').split('\n')) {
    if (fence) {
      const close = line.match(/^ {0,3}(`{3,}|~{3,})[ \t]*$/);
      if (close && close[1][0] === fence.marker[0] && close[1].length >= fence.marker.length) {
        if (fence.info === 'text') blocks.push(fence.lines.join('\n'));
        fence = null;
      } else {
        fence.lines.push(line);
      }
    } else {
      const open = line.match(/^ {0,3}(`{3,}|~{3,})(.*)$/);
      if (open) fence = { marker: open[1], info: open[2].trim(), lines: [] };
    }
  }
  assert.ok(!fence, 'The ETF page contains an unclosed code fence.');
  assert.ok(blocks.length > 0 && blocks.every((block) => block.trim().length > 0),
    'The ETF page must contain nonempty expected-output blocks.');
  assert.equal(normalize(stdout), blocks.join('\n\n'),
    'Executed ETF output differs from its documentation. Check the source and page together.');
}

function main() {
  const root = fileURLToPath(new URL('../../', import.meta.url));
  const page = fs.readFileSync(path.join(root, 'docs-site/src/content/docs/modules/etf-trick.md'), 'utf8');
  const result = spawnSync('cargo', ['run', '--quiet', '--locked', '-p', 'openquant', '--example', 'docs_etf_trick'],
    { cwd: root, encoding: 'utf8', maxBuffer: 10 * 1024 * 1024 });
  if (result.stderr) process.stderr.write(result.stderr);
  if (result.error || result.status !== 0) {
    console.error(`check:rust-output: ETF execution failed (${result.error?.message ?? result.status ?? result.signal}).`);
    process.exitCode = 1;
    return;
  }
  try {
    assertDocumentedOutput(page, result.stdout);
    console.log('check:rust-output: OK — ETF value and futures-roll output match the page.');
  } catch (error) {
    console.error(error);
    process.exitCode = 1;
  }
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) main();

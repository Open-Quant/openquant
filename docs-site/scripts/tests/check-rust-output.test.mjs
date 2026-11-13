import assert from 'node:assert/strict';
import test from 'node:test';
import { assertDocumentedOutput } from '../check-rust-output.mjs';

const page = '# Two examples\n\n```text\n2024-01-03  K = 1.000000\n```\n\nA second example.\n\n```text\ngap -0.90\n```\n';
const output = '2024-01-03  K = 1.000000\n\ngap -0.90\n';

test('checks every displayed output block in page order', () => {
  assert.doesNotThrow(() => assertDocumentedOutput(page, output));
});

test('accepts platform line endings and a final newline only', () => {
  assert.doesNotThrow(() => assertDocumentedOutput(page.replaceAll('\n', '\r\n'), output.replaceAll('\n', '\r\n')));
});

test('rejects a changed printed value', () => {
  assert.throws(() => assertDocumentedOutput(page, output.replace('1.000000', '1.000001')), /differs/);
});

test('rejects missing, additional or reordered output', () => {
  for (const changed of ['2024-01-03  K = 1.000000\n', `${output}extra\n`, 'gap -0.90\n\n2024-01-03  K = 1.000000\n']) {
    assert.throws(() => assertDocumentedOutput(page, changed), /differs/);
  }
});

test('rejects missing and empty expected-output blocks', () => {
  for (const changed of ['# No output', '```text\n```\n', `${page}\n\`\`\`text\n\`\`\`\n`]) {
    assert.throws(() => assertDocumentedOutput(changed, output), /nonempty/);
  }
});

test('keeps spaces inside displayed lines significant', () => {
  assert.throws(() => assertDocumentedOutput(page, output.replace('  K', ' K')), /differs/);
});

test('rejects additional trailing blank lines in output or documentation', () => {
  assert.throws(() => assertDocumentedOutput(page, `${output}\n`), /differs/);
  assert.throws(() => assertDocumentedOutput(page.replace('gap -0.90\n```', 'gap -0.90\n\n```'), output), /differs/);
});

test('compares tilde and longer Markdown fences instead of ignoring them', () => {
  assert.doesNotThrow(() => assertDocumentedOutput(page.replaceAll('```', '~~~~'), output));
  assert.doesNotThrow(() => assertDocumentedOutput(page.replaceAll('```', '````'), output));
  assert.throws(() => assertDocumentedOutput(`${page}\n~~~text\nextra\n~~~\n`, output), /differs/);
});

test('does not mistake a nested fence example inside a code block for output', () => {
  const nested = `${page}\n\`\`\`\`markdown\n\`\`\`text\nnot output\n\`\`\`\n\`\`\`\`\n`;
  assert.doesNotThrow(() => assertDocumentedOutput(nested, output));
});

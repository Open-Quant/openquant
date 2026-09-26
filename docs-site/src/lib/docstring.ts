/**
 * Render a Python docstring as HTML for the Python API reference.
 *
 * Docstrings here follow the numpy convention (see scripts/pyopenquant_bindings.py): prose
 * paragraphs, then `Parameters` / `Returns` / `Raises` sections, each a title underlined
 * with dashes whose entries are a `name : type` line followed by an indented description.
 * Anything else is rendered as paragraphs, bullet lists and preformatted blocks, so a
 * docstring in another style still reads correctly. Everything is escaped first: a
 * docstring can never inject markup.
 */

const escapeHtml = (s: string): string =>
  s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');

/** Escape, then turn ``code`` and `code` spans into <code>. */
export function inline(s: string): string {
  return escapeHtml(s)
    .replace(/``([^`]+?)``/g, '<code>$1</code>')
    .replace(/`([^`]+?)`/g, '<code>$1</code>');
}

const isRule = (line: string | undefined): boolean => !!line && /^\s*-{3,}\s*$/.test(line);
const indentOf = (line: string): number => line.length - line.trimStart().length;

function renderProse(lines: string[]): string {
  const out: string[] = [];
  let i = 0;
  while (i < lines.length) {
    const line = lines[i];
    if (!line.trim()) {
      i++;
      continue;
    }
    if (line.trim().startsWith('```')) {
      const body: string[] = [];
      i++;
      while (i < lines.length && !lines[i].trim().startsWith('```')) body.push(lines[i++]);
      i++;
      out.push(`<pre><code>${escapeHtml(body.join('\n'))}</code></pre>`);
      continue;
    }
    if (line.trim().startsWith('>>>') || indentOf(line) >= 4) {
      const body: string[] = [];
      while (i < lines.length && lines[i].trim() && (lines[i].trim().startsWith('>>>') || indentOf(lines[i]) >= 4 || lines[i].trim().startsWith('...'))) {
        body.push(lines[i++]);
      }
      out.push(`<pre><code>${escapeHtml(body.join('\n'))}</code></pre>`);
      continue;
    }
    if (/^\s*[-*] /.test(line)) {
      const items: string[] = [];
      while (i < lines.length && lines[i].trim()) {
        if (/^\s*[-*] /.test(lines[i])) items.push(lines[i].replace(/^\s*[-*] /, ''));
        else items[items.length - 1] += ' ' + lines[i].trim();
        i++;
      }
      out.push(`<ul>${items.map((it) => `<li>${inline(it)}</li>`).join('')}</ul>`);
      continue;
    }
    const para: string[] = [];
    while (i < lines.length && lines[i].trim() && !/^\s*[-*] /.test(lines[i]) && !lines[i].trim().startsWith('```')) {
      para.push(lines[i++].trim());
    }
    out.push(`<p>${inline(para.join(' '))}</p>`);
  }
  return out.join('\n');
}

/** A numpy section body: `term` lines at the base indent, descriptions indented below them. */
function renderEntries(lines: string[]): string {
  const nonEmpty = lines.filter((l) => l.trim());
  if (!nonEmpty.length) return '';
  const base = Math.min(...nonEmpty.map(indentOf));
  const entries: { term: string; desc: string[] }[] = [];
  for (const line of lines) {
    if (line.trim() && indentOf(line) === base) entries.push({ term: line.trim(), desc: [] });
    else if (entries.length) entries[entries.length - 1].desc.push(line.slice(base));
  }
  const dt = (term: string) => {
    const m = term.match(/^([A-Za-z_][\w, ]*?)\s+:\s+(.*)$/);
    return m
      ? `<code class="oq-param">${escapeHtml(m[1])}</code> : <span class="oq-type">${inline(m[2])}</span>`
      : `<span class="oq-type">${inline(term)}</span>`;
  };
  return `<dl class="oq-doc-entries">${entries
    .map((e) => `<dt>${dt(e.term)}</dt><dd>${renderProse(dedent(e.desc))}</dd>`)
    .join('')}</dl>`;
}

function dedent(lines: string[]): string[] {
  const nonEmpty = lines.filter((l) => l.trim());
  if (!nonEmpty.length) return lines;
  const n = Math.min(...nonEmpty.map(indentOf));
  return lines.map((l) => l.slice(n));
}

const ENTRY_SECTIONS = new Set([
  'Parameters',
  'Returns',
  'Yields',
  'Raises',
  'Warns',
  'Attributes',
  'Other Parameters',
]);

export function renderDocstring(doc: string): string {
  if (!doc.trim()) return '<p class="oq-doc-missing">No docstring yet.</p>';
  const lines = doc.replace(/\r\n/g, '\n').split('\n');
  const parts: string[] = [];
  let current: { title: string | null; body: string[] } = { title: null, body: [] };
  const flush = () => {
    if (current.title === null) {
      parts.push(renderProse(current.body));
    } else {
      const body = ENTRY_SECTIONS.has(current.title)
        ? renderEntries(current.body)
        : renderProse(dedent(current.body));
      parts.push(`<h4 class="oq-doc-section">${escapeHtml(current.title)}</h4>${body}`);
    }
  };
  for (let i = 0; i < lines.length; i++) {
    if (lines[i].trim() && isRule(lines[i + 1])) {
      flush();
      current = { title: lines[i].trim(), body: [] };
      i++;
      continue;
    }
    current.body.push(lines[i]);
  }
  flush();
  return parts.filter(Boolean).join('\n');
}

/** First paragraph of a docstring, as one line of plain text. */
export function summary(doc: string): string {
  return (doc || '').split(/\n\s*\n/)[0].replace(/\s+/g, ' ').trim();
}

/** Machine-readable navigation, generated from the same collection as the site. */
import type { APIRoute } from 'astro';
import { getCollection } from 'astro:content';
import { execFileSync } from 'node:child_process';
import reference from '../data/pythonApiReference.json';

const repository = 'https://github.com/Open-Quant/openquant';
const oneLine = (value: string) => value.replace(/\s+/g, ' ').trim();
const label = (value: string) => oneLine(value).replace(/[\\[\]]/g, '\\$&');

export const GET: APIRoute = async ({ site }) => {
  if (!site) throw new Error('The agent index needs the configured canonical site URL.');
  const base = import.meta.env.BASE_URL.replace(/\/$/, '');
  const url = (slug: string) => new URL(`${base}/${slug ? `${slug}/` : ''}`, site).href;
  let revision: string | null = null;
  let workingTree = 'unavailable';
  try {
    const head = execFileSync('git', ['rev-parse', 'HEAD'], { encoding: 'utf8', stdio: ['ignore', 'pipe', 'ignore'] }).trim();
    if (/^[a-f0-9]{40,64}$/.test(head)) revision = head;
    const status = execFileSync('git', ['status', '--porcelain', '--untracked-files=normal'], { encoding: 'utf8', stdio: ['ignore', 'pipe', 'ignore'] });
    workingTree = status.trim() ? 'modified' : 'clean';
  } catch {
    // Source archives may not contain Git metadata. Never claim an invented revision.
  }

  const pages = (await getCollection('docs')).map((entry) => ({
    slug: entry.id.replace(/(^|\/)index$/, ''),
    title: entry.data.title,
    description: entry.data.description ?? '',
    status: entry.data.status,
  })).sort((a, b) => a.slug.localeCompare(b.slug));
  const starts = ['', 'quickstart', 'agents', 'modules', 'runbooks'];
  const line = (page: (typeof pages)[number]) =>
    `- [${label(page.title)}](${url(page.slug)}): [${page.status}] ${oneLine(page.description)}`;

  const text = [
    '# OpenQuant', '',
    '> Rust implementations of financial machine-learning methods, with Python bindings.', '',
    `Source revision: ${revision ?? 'unavailable (source archive without Git metadata)'}`,
    `Working tree: ${workingTree}`,
    ...(workingTree === 'modified' ? ['Local edits are included. The revision identifies the base commit, not the uncommitted content.'] : []), '',
    'This index is generated from the documentation collection and Python reference snapshot.',
    'Read exact signatures, return shapes and limitations in the linked API reference; this file does not copy them.',
    'The Python distribution is pyopenquant; import it as openquant. Do not install the unrelated openquant PyPI project.',
    'Page status describes documentation review, not numerical correctness. Authored does not mean human-reviewed.', '',
    '## Start here', '',
    ...starts.flatMap((slug) => pages.filter((page) => page.slug === slug).map(line)), '',
    '## API reference', '',
    `- [Python API](${url('api/python')}): Generated signatures and docstrings. Prefer public wrappers to compiled internals.`,
    `- [Rust API](${url('api/rust/openquant')}): Generated rustdoc for the numerical core.`,
    ...reference.modules.filter((module) => !module.import.startsWith('openquant._core.'))
      .sort((a, b) => a.import.localeCompare(b.import))
      .map((module) => `- [${label(module.import)}](${url(`api/python/${module.slug}`)}): [generated] ${module.kind} module.`), '',
    '## Guides and module explanations', '',
    ...pages.filter((page) => !starts.includes(page.slug)).map(line), '',
    '## Repository sources', '',
    `- [Source checkout](${revision ? `${repository}/tree/${revision}` : repository}): Match the implementation and tests to this documentation revision.`,
    `- [Contribution workflow](${repository}/blob/${revision ?? 'main'}/CONTRIBUTING.md): Setup, required checks and generated-source ownership.`, '',
  ].join('\n');
  return new Response(text, { headers: { 'Content-Type': 'text/plain; charset=utf-8' } });
};

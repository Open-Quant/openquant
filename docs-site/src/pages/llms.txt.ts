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
  const routes = [
    { slug: 'module-reference/api-task-guide', purpose: 'Choose the Python wrapper, native binding or Rust-only entry point for a concrete task before copying an API call.' },
    { slug: 'setup/python-bindings', purpose: 'Install or rebuild Python bindings; verify the imported extension before testing.' },
    { slug: 'workflows/rust-core-workflow', purpose: 'Follow a Rust component walkthrough and inspect alignment, numerical conventions and its synthetic-input limits.' },
    { slug: 'workflows/notebook-research-workflow', purpose: 'Run or reproduce notebooks with the correct tier, environment and evidence.' },
    { slug: 'workflows/point-in-time-data', purpose: 'Select revisions available at a research cutoff; distinguish synthetic selection checks and content/policy identity from provider-history evidence.' },
    { slug: 'modules/provenance', purpose: 'Select supplied revisions before cleaning and bind exact typed values through stable-ID rejoining, without claiming provider authentication.' },
    { slug: 'modules/feature-assembly', purpose: 'Require complete declared source fields, retain ordered lineage and missing-input exclusions, and inspect coverage without assuming representativeness.' },
    { slug: 'modules/availability', purpose: 'Intersect original training indices with explicit source-availability and label-maturity boundaries; keep purging and learned preprocessing separate.' },
    { slug: 'modules/data', purpose: 'Load data and establish cleaning, cache and provenance contracts before analysis.' },
    { slug: 'modules/evaluation', purpose: 'Define explicit trial identity, registry scope and shared-read limitations before comparing searches.' },
    { slug: 'workflows/research-journey', purpose: 'Follow fold-local fitting and read back the saved JSON event, fold and prediction handoffs.' },
    { slug: 'workflows/model-selection-holdout', purpose: 'Separate development-only parameter selection and fold-local preprocessing from refitting and final chronological holdout classification.' },
    { slug: 'workflows/multi-asset-allocation', purpose: 'Align long-format observations, matrix axes and named weights; preserve moment units and fitting boundaries without treating allocation as a realized backtest.' },
    { slug: 'examples/catalog', purpose: 'Choose runnable entry points and audit saved grid artifacts without confusing flags with validation.' },
    { slug: 'modules/cross-validation', purpose: 'Construct label-span-aware evaluation and preserve row alignment.' },
    { slug: 'setup/local-build', purpose: 'Build documentation and distinguish preview, generation and aggregate checks.' },
    { slug: 'setup/troubleshooting', purpose: 'Diagnose command, interpreter and extension failures.' },
  ];
  const routedSlugs = new Set(routes.map((route) => route.slug));
  const line = (page: (typeof pages)[number]) =>
    `- [${label(page.title)}](${url(page.slug)}): [${page.status}] ${oneLine(page.description)}`;

  const text = [
    '# OpenQuant', '',
    '> Rust implementations of financial machine-learning methods, with Python bindings.', '',
    `Source revision: ${revision ?? 'unavailable (source archive without Git metadata)'}`,
    `Working tree: ${workingTree}`,
    ...(workingTree === 'modified' ? ['Local edits are included. The revision identifies the base commit, not the uncommitted content.'] : []), '',
    'This index is generated from the documentation collection and Python reference snapshot.',
    'The source revision identifies the site build, not a separately verified installed package. Match your interpreter and imported package paths to the intended checkout before testing changes.',
    'Before resuming work, preserve local edits and identify the selected interpreter, imported extension, dataset, registry and output paths; inspect saved artifacts before rerunning.',
    'Read exact signatures, return shapes and limitations in the linked API reference; this file does not copy them.',
    'The Python distribution is pyopenquant; import it as openquant. Do not install the unrelated openquant PyPI project.',
    'Page status describes documentation review, not numerical correctness. Authored does not mean human-reviewed.', '',
    '## Start here', '',
    ...starts.flatMap((slug) => pages.filter((page) => page.slug === slug).map(line)), '',
    '## Task routes', '',
    ...routes.flatMap((route) => pages.filter((page) => page.slug === route.slug)
      .map((page) => `- [${label(page.title)}](${url(page.slug)}): [${page.status}] ${route.purpose}`)), '',
    '## API reference', '',
    `- [Python API](${url('api/python')}): Generated signatures and docstrings. Prefer public wrappers to compiled internals.`,
    `- [Rust API](${url('api/rust/openquant')}): Generated rustdoc for the numerical core.`,
    ...reference.modules.filter((module) => !module.import.startsWith('openquant._core.'))
      .sort((a, b) => a.import.localeCompare(b.import))
      .map((module) => `- [${label(module.import)}](${url(`api/python/${module.slug}`)}): [generated] ${module.kind} module.`), '',
    '## Guides and module explanations', '',
    ...pages.filter((page) => !starts.includes(page.slug) && !routedSlugs.has(page.slug)).map(line), '',
    '## Repository sources', '',
    `- [Source checkout](${revision ? `${repository}/tree/${revision}` : repository}): Match the implementation and tests to this documentation revision.`,
    `- [Repository workflow](${repository}/blob/${revision ?? 'main'}/AI-DLC.md): Read repository instructions before editing; use applicable instructions from your checkout.`,
    `- [Contribution workflow](${repository}/blob/${revision ?? 'main'}/CONTRIBUTING.md): Setup, required checks and generated-source ownership.`,
    `- [Documentation ownership](${repository}/blob/${revision ?? 'main'}/docs/documentation-guide.md): Canonical sources, documentation review and evidence requirements.`, '',
  ].join('\n');
  return new Response(text, { headers: { 'Content-Type': 'text/plain; charset=utf-8' } });
};

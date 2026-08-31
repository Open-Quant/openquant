import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { moduleDocs } from '../src/data/moduleDocs.ts';

// Anchored to this file, not to process.cwd(). Running the generator from the
// repo root used to silently create a second, stray src/content/docs/modules/
// tree there instead of writing the real pages.
const docsSite = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const outDir = path.join(docsSite, 'src/content/docs/modules');
fs.mkdirSync(outDir, { recursive: true });

// These pages are machine-emitted from src/data/moduleDocs.ts and have not been read
// by a human, so they may only ever claim 'generated' — never 'reviewed' or 'validated'.
// Those stronger statuses are reachable only by a person deliberately editing a page.
// Date-only, so repeated runs on the same day are byte-identical (the generator must
// stay idempotent) and so the stamp is comparable with the schema gate's mtime rule.
const generatedOn = new Date().toISOString().slice(0, 10);

const q = (value) => JSON.stringify(String(value));
const toYamlList = (values) => values.map((v) => `  - ${q(v)}`).join('\n');

// conceptOverview, whenToUse and relatedModules are load-bearing: without them a
// page is a heading skeleton. They were optional once and 27 of 39 modules left
// them out, which shipped as thin pages rather than as an error. Fail loudly now.
const REQUIRED_TEXT = ['conceptOverview', 'whenToUse'];
for (const doc of moduleDocs) {
  for (const field of REQUIRED_TEXT) {
    if (typeof doc[field] !== 'string' || !doc[field].trim()) {
      throw new Error(
        `moduleDocs.ts: ${doc.slug} is missing a non-empty ${field}. ` +
          'Write it — a module page without one is a heading skeleton.'
      );
    }
  }
  if (!Array.isArray(doc.relatedModules) || doc.relatedModules.length === 0) {
    throw new Error(
      `moduleDocs.ts: ${doc.slug} is missing relatedModules. ` +
        'Name at least one related module; an isolated page is a dead end.'
    );
  }
  const slugs = new Set(moduleDocs.map((d) => d.slug));
  for (const slug of doc.relatedModules) {
    if (!slugs.has(slug)) {
      throw new Error(
        `moduleDocs.ts: ${doc.slug} lists relatedModules entry "${slug}", ` +
          'which is not a module slug — the generated link would 404.'
      );
    }
  }
}

for (const doc of moduleDocs) {
  const sections = [];

  // --- Concept Overview ---
  if (doc.conceptOverview) {
    sections.push(`## Concept Overview\n\n${doc.conceptOverview}`);
  }

  // --- When to Use ---
  if (doc.whenToUse) {
    sections.push(`## When to Use\n\n${doc.whenToUse}`);
  }

  // `## Subject` and `## Why This Module Exists` used to render here for modules
  // without a conceptOverview. `subject` is a one-word taxonomy string — metadata
  // formatted as a section — and `whyItExists` is a single sentence that
  // conceptOverview now covers properly. Both branches are gone with the stubs.

  // --- Mathematical Foundations ---
  const formulas = doc.formulas.length
    ? doc.formulas
        .map(
          (f) =>
            `### ${f.label}\n\n$$${f.latex}$$` +
            (f.where ? `\n\nwhere ${f.where}` : '')
        )
        .join('\n\n')
    : null;

  if (formulas) {
    sections.push(`## Mathematical Foundations\n\n${formulas}`);
  }

  // --- Key Parameters ---
  if (doc.keyParameters && doc.keyParameters.length) {
    const header = '| Parameter | Type | Description | Default |\n|-----------|------|-------------|---------|\n';
    const rows = doc.keyParameters
      .map(
        (p) =>
          `| \`${p.name}\` | \`${p.type}\` | ${p.description} | ${p.default ?? '—'} |`
      )
      .join('\n');
    sections.push(`## Key Parameters\n\n${header}${rows}`);
  }

  // --- Usage Examples ---
  if (doc.examples.length) {
    const pythonExamples = doc.examples.filter((ex) => ex.language === 'python');
    const rustExamples = doc.examples.filter((ex) => ex.language === 'rust');
    const otherExamples = doc.examples.filter(
      (ex) => ex.language !== 'python' && ex.language !== 'rust'
    );

    const exSections = [];
    if (pythonExamples.length) {
      exSections.push(
        `### Python\n\n${pythonExamples
          .map((ex) => `#### ${ex.title}\n\n\`\`\`python\n${ex.code}\n\`\`\``)
          .join('\n\n')}`
      );
    }
    if (rustExamples.length) {
      exSections.push(
        `### Rust\n\n${rustExamples
          .map((ex) => `#### ${ex.title}\n\n\`\`\`rust\n${ex.code}\n\`\`\``)
          .join('\n\n')}`
      );
    }
    if (otherExamples.length) {
      exSections.push(
        otherExamples
          .map(
            (ex) =>
              `### ${ex.title}\n\n\`\`\`${ex.language}\n${ex.code}\n\`\`\``
          )
          .join('\n\n')
      );
    }

    sections.push(`## Usage Examples\n\n${exSections.join('\n\n')}`);
  }

  // --- Common Pitfalls ---
  if (doc.commonPitfalls && doc.commonPitfalls.length) {
    sections.push(
      `## Common Pitfalls\n\n${doc.commonPitfalls.map((p) => `- ${p}`).join('\n')}`
    );
  }

  // --- API Reference ---
  const apiParts = [];
  if (doc.pythonApis && doc.pythonApis.length) {
    apiParts.push(
      `### Python API\n\n${doc.pythonApis.map((api) => `- \`${api}\``).join('\n')}`
    );
  }
  if (doc.keyApis.length) {
    const label = doc.apiSurface === 'python-only' ? 'Key Functions' : 'Rust API';
    apiParts.push(
      `### ${label}\n\n${doc.keyApis.map((api) => `- \`${api}\``).join('\n')}`
    );
  }
  if (apiParts.length) {
    sections.push(`## API Reference\n\n${apiParts.join('\n\n')}`);
  } else {
    sections.push(
      `## Key Public APIs\n\n${doc.keyApis.map((api) => `- \`${api}\``).join('\n')}`
    );
  }

  // --- Risk Notes and Caveats ---
  // These were emitted twice on every page: once as the `risk_notes` frontmatter
  // key and again, verbatim, as a body section titled 'Implementation Notes'.
  // Nothing reads the frontmatter key (it is optional in src/content/config.ts
  // and referenced by no component or script), and the content is caveats rather
  // than implementation detail — so it is now rendered once, under its real name.
  if (doc.notes.length) {
    sections.push(
      `## Risk Notes and Caveats\n\n${doc.notes.map((n) => `- ${n}`).join('\n')}`
    );
  }

  // --- Related Modules ---
  if (doc.relatedModules && doc.relatedModules.length) {
    sections.push(
      `## Related Modules\n\n${doc.relatedModules
        .map((slug) => `- [\`${slug}\`](/modules/${slug}/)`)
        .join('\n')}`
    );
  }

  // --- AFML Chapter References ---
  const chapterNote =
    doc.afmlChapters && doc.afmlChapters.length
      ? `afml_chapters:\n${doc.afmlChapters.map((c) => `  - ${c}`).join('\n')}\n`
      : '';

  const content = `---
title: ${q(doc.module)}
description: ${q(doc.summary)}
status: generated
generated_from: src/data/moduleDocs.ts
last_generated: '${generatedOn}'
audience:
  - quant-dev
  - platform-engineering
module: ${q(doc.module)}
${doc.apiSurface ? `api_surface: ${q(doc.apiSurface)}\n` : ''}${chapterNote}rust_api:
${toYamlList(doc.keyApis)}
sidebar:
  badge: Module
---

${sections.join('\n\n')}
`;

  fs.writeFileSync(path.join(outDir, `${doc.slug}.md`), content, 'utf8');
}

// --- Index: the canonical module index -------------------------------------
// This page is the one index of the 39 modules. It absorbed
// module-reference/api-surfaces.md (the 'By language surface' section below),
// which was a fourth hand-maintained listing of the same modules and had
// already drifted from the data; generating it here means it cannot drift
// again.

// --- Index: grouped by subject ---
const bySubject = new Map();
for (const doc of moduleDocs) {
  const group = bySubject.get(doc.subject) || [];
  group.push(doc);
  bySubject.set(doc.subject, group);
}

const groupedIndex = [...bySubject.entries()]
  .sort(([a], [b]) => a.localeCompare(b))
  .map(
    ([subject, docs]) =>
      `### ${subject}\n\n${docs
        .sort((a, b) => a.module.localeCompare(b.module))
        .map(
          (doc) =>
            `- [\`${doc.module}\`](/modules/${doc.slug}/) — ${doc.summary}`
        )
        .join('\n')}`
  )
  .join('\n\n');

// --- Index: grouped by language surface ---
// `apiSurface` says which bindings a module is reachable through; `pythonApis`
// entries are namespace-qualified (`bet_sizing.get_signal`), so the Python half
// is grouped by the namespace a reader actually imports.
const rustModules = moduleDocs
  .filter((doc) => doc.apiSurface !== 'python-only')
  .sort((a, b) => a.module.localeCompare(b.module));

const byNamespace = new Map();
for (const doc of moduleDocs) {
  for (const api of doc.pythonApis ?? []) {
    const dot = api.lastIndexOf('.');
    const namespace = dot === -1 ? doc.module : api.slice(0, dot);
    const entry = byNamespace.get(namespace) || { slug: doc.slug, fns: [] };
    if (entry.slug !== doc.slug) {
      throw new Error(
        `moduleDocs.ts: python namespace "${namespace}" is claimed by both ` +
          `${entry.slug} and ${doc.slug}. The index can only link it to one page — ` +
          'split the namespace or move the APIs onto one module.'
      );
    }
    entry.fns.push(api.slice(dot + 1));
    byNamespace.set(namespace, entry);
  }
}

const languageIndex = [
  '### Rust core',
  '',
  rustModules
    .map((doc) => `- [\`${doc.module}\`](/modules/${doc.slug}/) — ${doc.summary}`)
    .join('\n'),
  '',
  '### Python namespaces',
  '',
  [...byNamespace.entries()]
    .sort(([a], [b]) => a.localeCompare(b))
    .map(
      ([namespace, { slug, fns }]) =>
        `- [\`${namespace}\`](/modules/${slug}/) — ${[...new Set(fns)]
          .map((fn) => `\`${fn}\``)
          .join(', ')}`
    )
    .join('\n'),
].join('\n');

const indexContent = `---
title: "Module Reference Index"
description: "Full OpenQuant module documentation index with AFML-aligned summaries."
status: generated
generated_from: src/data/moduleDocs.ts
last_generated: '${generatedOn}'
audience:
  - quant-dev
  - platform-engineering
sidebar:
  order: 1
---

This is the canonical index of every OpenQuant module: one page each, with
purpose, APIs, formulas, examples, and implementation notes. It lists the same
39 modules twice over — by subject, and by the language surface they are
reachable through — because those are the two questions readers arrive with.
For the AFML chapter each module implements, see
[By AFML Chapter](/module-reference/by-afml-chapter/).

## By subject

${groupedIndex}

## By language surface

${languageIndex}
`;

fs.writeFileSync(path.join(outDir, 'index.md'), indexContent, 'utf8');
console.log(`Generated ${moduleDocs.length} module pages.`);

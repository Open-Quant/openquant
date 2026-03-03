import fs from 'node:fs';
import path from 'node:path';
import { moduleDocs } from '../src/data/moduleDocs.ts';

const outDir = path.resolve(process.cwd(), 'src/content/docs/modules');
fs.mkdirSync(outDir, { recursive: true });

const q = (value) => JSON.stringify(String(value));
const toYamlList = (values) => values.map((v) => `  - ${q(v)}`).join('\n');

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

  // --- Subject (kept for non-enriched modules) ---
  if (!doc.conceptOverview) {
    sections.push(`## Subject\n\n**${doc.subject}**`);
  }

  // --- Why This Module Exists (kept for non-enriched modules) ---
  if (!doc.conceptOverview) {
    sections.push(`## Why This Module Exists\n\n${doc.whyItExists}`);
  }

  // --- Mathematical Foundations ---
  const formulas = doc.formulas.length
    ? doc.formulas
        .map((f) => `### ${f.label}\n\n$$${f.latex}$$`)
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

  // --- Implementation Notes ---
  if (doc.notes.length) {
    sections.push(
      `## Implementation Notes\n\n${doc.notes.map((n) => `- ${n}`).join('\n')}`
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

  // Build frontmatter — risk_notes stay for schema queries, body uses Implementation Notes
  const content = `---
title: ${q(doc.module)}
description: ${q(doc.summary)}
status: validated
last_validated: '2026-03-02'
audience:
  - quant-dev
  - platform-engineering
module: ${q(doc.module)}
${doc.apiSurface ? `api_surface: ${q(doc.apiSurface)}\n` : ''}${chapterNote}risk_notes:
${toYamlList(doc.notes)}
rust_api:
${toYamlList(doc.keyApis)}
sidebar:
  badge: Module
---

${sections.join('\n\n')}
`;

  fs.writeFileSync(path.join(outDir, `${doc.slug}.md`), content, 'utf8');
}

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

const indexContent = `---
title: "Module Reference Index"
description: "Full OpenQuant module documentation index with AFML-aligned summaries."
status: validated
last_validated: '2026-03-02'
audience:
  - quant-dev
  - platform-engineering
sidebar:
  order: 1
---

This index contains one page per OpenQuant module with purpose, APIs, formulas, examples, and implementation notes.

${groupedIndex}
`;

fs.writeFileSync(path.join(outDir, 'index.md'), indexContent, 'utf8');
console.log(`Generated ${moduleDocs.length} module pages.`);

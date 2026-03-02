import fs from 'node:fs';
import path from 'node:path';
import { moduleDocs } from '../src/data/moduleDocs.ts';

const outDir = path.resolve(process.cwd(), 'src/content/docs/modules');
fs.mkdirSync(outDir, { recursive: true });

const q = (value) => JSON.stringify(String(value));
const toYamlList = (values) => values.map((v) => `  - ${q(v)}`).join('\n');

for (const doc of moduleDocs) {
  const formulas = doc.formulas.length
    ? doc.formulas
        .map((f) => `### ${f.label}\n\n$$${f.latex}$$`)
        .join('\n\n')
    : 'No formal equations documented yet for this module.';

  const examples = doc.examples.length
    ? doc.examples
        .map(
          (ex) =>
            `### ${ex.title}\n\n\`\`\`${ex.language}\n${ex.code}\n\`\`\``
        )
        .join('\n\n')
    : 'No code examples documented yet for this module.';

  const content = `---
title: ${q(doc.module)}
description: ${q(doc.summary)}
status: validated
last_validated: '2026-03-02'
audience:
  - quant-dev
  - platform-engineering
module: ${q(doc.module)}
risk_notes:
${toYamlList(doc.notes)}
rust_api:
${toYamlList(doc.keyApis)}
sidebar:
  badge: Module
---

## Subject

**${doc.subject}**

## Why This Module Exists

${doc.whyItExists}

## Key Public APIs

${doc.keyApis.map((api) => `- \`${api}\``).join('\n')}

## Mathematical Definitions

${formulas}

## Implementation Examples

${examples}

## Implementation Notes

${doc.notes.map((n) => `- ${n}`).join('\n')}
`;

  fs.writeFileSync(path.join(outDir, `${doc.slug}.md`), content, 'utf8');
}

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

${moduleDocs
  .slice()
  .sort((a, b) => a.module.localeCompare(b.module))
  .map((doc) => `- [\`${doc.module}\`](/modules/${doc.slug}/) - ${doc.summary}`)
  .join('\n')}
`;

fs.writeFileSync(path.join(outDir, 'index.md'), indexContent, 'utf8');
console.log(`Generated ${moduleDocs.length} module pages.`);

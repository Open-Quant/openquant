# Module Doc Template (AFML-aligned)

Use this structure for each module entry in `docs-site/src/data/moduleDocs.ts`.

## Required fields

1. `summary`
2. `whyItExists`
3. `keyApis` (public APIs only)
4. `formulas` (2+ where applicable)
5. `examples` (2+ realistic executable snippets)
6. `notes` (production caveats)

## Writing checklist

1. Tie each claim to AFML chapter context and concrete API behavior.
2. Prefer realistic examples with data shape assumptions and error handling notes.
3. Include at least one failure-mode or misuse warning.
4. Keep naming and units consistent with module implementation.
5. Cross-link conceptually adjacent modules in notes.

## Consistency rules

1. Keep tense and voice consistent across modules.
2. Avoid marketing phrases and vague adjectives.
3. Use exact API names from Rust modules.
4. Keep formulas readable and connected to implementation.
5. Keep examples concise but runnable.

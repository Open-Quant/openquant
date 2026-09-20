import { defineCollection, z } from 'astro:content';
import { docsSchema } from '@astrojs/starlight/schema';

const docs = defineCollection({
  schema: docsSchema({
    extend: z.object({
      audience: z.array(z.string()).optional(),
      // Kept in step with STATUS_DATE_FIELD in scripts/check-content-schema.mjs,
      // which additionally enforces that the date backing a status is real and is
      // not older than the page's last change. Required, not optional: a page with
      // no status is a page making an unexamined claim by omission.
      status: z.enum(['generated', 'draft', 'authored', 'reviewed', 'validated']),
      last_generated: z.string().optional(),
      generated_from: z.string().optional(),
      last_validated: z.string().optional(),
      last_authored: z.string().optional(),
      // Where the method comes from: AFML chapter/section/snippet and the original paper.
      // Required on every hand-written module page by scripts/check-content-schema.mjs.
      citation: z.array(z.string()).optional(),
      module: z.string().optional(),
      afml_chapter: z.array(z.string()).optional(),
      rust_api: z.array(z.string()).optional(),
      python_api: z.array(z.string()).optional(),
      examples: z.array(z.string()).optional(),
      risk_notes: z.array(z.string()).optional(),
    }),
  }),
});

export const collections = { docs };

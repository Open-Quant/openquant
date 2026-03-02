import { defineCollection, z } from 'astro:content';
import { docsSchema } from '@astrojs/starlight/schema';

const docs = defineCollection({
  schema: docsSchema({
    extend: z.object({
      audience: z.array(z.string()).optional(),
      status: z.enum(['draft', 'in_review', 'validated']).optional(),
      last_validated: z.string().optional(),
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

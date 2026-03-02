import { defineCollection, z } from 'astro:content';
import { docsSchema } from '@astrojs/starlight/schema';
import { docsLoader } from '@astrojs/starlight/loaders';

const docs = defineCollection({
  loader: docsLoader(),
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

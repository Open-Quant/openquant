import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';

export default defineConfig({
  base: '/openquant',
  output: 'static',
  integrations: [
    starlight({
      title: 'OpenQuant Documentation',
      description: 'Institutional-grade quantitative research and production docs for OpenQuant.',
      logo: {
        src: './src/assets/openquant-icon.svg',
        alt: 'OpenQuant',
      },
      customCss: ['./src/styles/starlight.css'],
      social: [
        { icon: 'github', label: 'GitHub', href: 'https://github.com/Open-Quant/openquant' },
      ],
      sidebar: [
        {
          label: 'Getting Started',
          items: [
            { label: 'Overview', link: '/' },
            { label: 'Quickstart', link: '/quickstart/' },
          ],
        },
        {
          label: 'Setup',
          autogenerate: { directory: 'setup' },
        },
        {
          label: 'Core Workflows',
          autogenerate: { directory: 'workflows' },
        },
        {
          label: 'Module Reference',
          autogenerate: { directory: 'module-reference' },
        },
        {
          label: 'Examples',
          autogenerate: { directory: 'examples' },
        },
        {
          label: 'Governance & Release',
          autogenerate: { directory: 'governance' },
        },
        { label: 'Coverage Dashboard', link: '/coverage/' },
        { label: 'Module Pages', link: '/module/' },
      ],
    }),
  ],
  redirects: {
    '/getting-started': '/setup/local-build/',
    '/guides': '/workflows/rust-core-workflow/',
    '/tutorials': '/workflows/python-core-workflow/',
    '/notebook-research-workflow': '/workflows/notebook-research-workflow/',
    '/api-reference': '/module-reference/api-surfaces/',
    '/examples': '/examples/catalog/',
    '/modules': '/module-reference/by-afml-chapter/',
    '/search': '/module-reference/indexing-and-discovery/',
    '/publishing': '/governance/versioning-and-release-policy/',
    '/performance': '/governance/benchmark-policy/',
    '/contributing': '/governance/support-and-escalation/',
    '/faq': '/governance/methodology-and-leakage-controls/',
  },
});

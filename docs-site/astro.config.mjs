import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import { remarkBaseUrl } from './scripts/remark-base-url.mjs';

export default defineConfig({
  site: 'https://open-quant.github.io',
  base: '/openquant',
  output: 'static',
  markdown: {
    remarkPlugins: [remarkMath, remarkBaseUrl({ base: '/openquant' })],
    rehypePlugins: [rehypeKatex],
  },
  integrations: [
    starlight({
      title: 'OpenQuant Documentation',
      description: 'Institutional-grade quantitative research and production docs for OpenQuant.',
      logo: {
        src: './src/assets/openquant-icon.svg',
        alt: 'OpenQuant',
      },
      head: [
        {
          tag: 'link',
          attrs: {
            rel: 'preconnect',
            href: 'https://fonts.googleapis.com',
          },
        },
        {
          tag: 'link',
          attrs: {
            rel: 'preconnect',
            href: 'https://fonts.gstatic.com',
            crossorigin: true,
          },
        },
        {
          tag: 'link',
          attrs: {
            rel: 'stylesheet',
            href: 'https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&family=IBM+Plex+Mono:wght@400;500&display=swap',
          },
        },
      ],
      customCss: [
        'katex/dist/katex.min.css',
        './src/styles/starlight.css',
      ],
      components: {
        // Derives each page's status pill from its `status` frontmatter rather
        // than from hand-written markup in `banner.content`.
        Banner: './src/components/DocStatusBanner.astro',
      },
      social: [
        { icon: 'github', label: 'GitHub', href: 'https://github.com/Open-Quant/openquant' },
      ],
      // Seven top-level groups. It was twenty, thirteen of them one AFML
      // chapter each and five holding a single page, so a reader scrolled past
      // a chapter menu to reach Reference, Governance and Coverage. The AFML
      // chapters are still here — nested one level inside Modules, where a
      // one-page chapter costs a line rather than a heading.
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
          label: 'Workflows',
          autogenerate: { directory: 'workflows' },
        },
        {
          label: 'Modules',
          items: [
            { label: 'All Modules', link: '/modules/' },
            {
              label: 'Ch 2: Data Structures',
              collapsed: true,
              items: [
                { slug: 'modules/data-structures' },
                { slug: 'modules/filters' },
                { slug: 'modules/etf-trick' },
              ],
            },
            {
              label: 'Ch 3: Labeling',
              collapsed: true,
              items: [
                { slug: 'modules/labeling' },
                { slug: 'modules/bet-sizing' },
              ],
            },
            {
              label: 'Ch 4: Sample Weights',
              collapsed: true,
              items: [
                { slug: 'modules/sampling' },
                { slug: 'modules/sample-weights' },
                { slug: 'modules/sb-bagging' },
              ],
            },
            {
              label: 'Ch 5–7: Features & Validation',
              collapsed: true,
              items: [
                { slug: 'modules/fracdiff' },
                { slug: 'modules/ensemble-methods' },
                { slug: 'modules/cross-validation' },
              ],
            },
            {
              label: 'Ch 8–9: Importance & Tuning',
              collapsed: true,
              items: [
                { slug: 'modules/feature-importance' },
                { slug: 'modules/fingerprint' },
                { slug: 'modules/hyperparameter-tuning' },
              ],
            },
            {
              label: 'Ch 10–12: Backtesting',
              collapsed: true,
              items: [
                { slug: 'modules/backtesting-engine' },
                { slug: 'modules/synthetic-backtesting' },
              ],
            },
            {
              label: 'Ch 14–15: Diagnostics & Risk',
              collapsed: true,
              items: [
                { slug: 'modules/backtest-statistics' },
                { slug: 'modules/risk-metrics' },
                { slug: 'modules/strategy-risk' },
              ],
            },
            {
              label: 'Ch 16: Portfolio Construction',
              collapsed: true,
              items: [
                { slug: 'modules/hrp' },
                { slug: 'modules/hcaa' },
                { slug: 'modules/onc' },
                { slug: 'modules/cla' },
                { slug: 'modules/portfolio-optimization' },
              ],
            },
            {
              label: 'Ch 17–19: Microstructure & Regimes',
              collapsed: true,
              items: [
                { slug: 'modules/structural-breaks' },
                { slug: 'modules/microstructural-features' },
                { slug: 'modules/codependence' },
              ],
            },
            {
              label: 'Ch 20–22: HPC & Advanced',
              collapsed: true,
              items: [
                { slug: 'modules/hpc-parallel' },
                { slug: 'modules/combinatorial-optimization' },
                { slug: 'modules/streaming-hpc' },
              ],
            },
            {
              label: 'Shared Utilities',
              collapsed: true,
              items: [
                { slug: 'modules/ef3m' },
                { slug: 'modules/util-fast-ewma' },
                { slug: 'modules/util-volatility' },
              ],
            },
            {
              label: 'Python Modules',
              collapsed: true,
              items: [
                { slug: 'modules/data' },
                { slug: 'modules/feature-diagnostics' },
                { slug: 'modules/pipeline' },
                { slug: 'modules/research' },
                { slug: 'modules/adapters' },
                { slug: 'modules/viz' },
              ],
            },
          ],
        },
        {
          label: 'Reference',
          items: [
            { slug: 'module-reference/by-afml-chapter' },
            { slug: 'examples/catalog' },
          ],
        },
        {
          label: 'Governance',
          autogenerate: { directory: 'governance' },
        },
        { label: 'Coverage', link: '/coverage/' },
      ],
    }),
  ],
  redirects: {
    '/getting-started': '/openquant/setup/local-build/',
    '/guides': '/openquant/workflows/rust-core-workflow/',
    '/tutorials': '/openquant/workflows/python-core-workflow/',
    '/notebook-research-workflow': '/openquant/workflows/notebook-research-workflow/',
    // api-surfaces.md was merged into the module index as 'By language surface'.
    '/api-reference': '/openquant/modules/',
    '/examples': '/openquant/examples/catalog/',
    // '/modules' intentionally absent: it shadowed the real modules/index.md page
    // and broke the homepage "Browse Modules" CTA.
    // indexing-and-discovery.md described the sidebar; its discovery paths now
    // live on the home page.
    '/search': '/openquant/',
    '/publishing': '/openquant/governance/versioning-and-release-policy/',
    '/performance': '/openquant/governance/benchmark-policy/',
    '/contributing': '/openquant/governance/support-and-escalation/',
    '/faq': '/openquant/governance/methodology-and-leakage-controls/',
    '/module': '/openquant/modules/',
  },
});

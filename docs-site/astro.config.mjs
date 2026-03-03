import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import { remarkBaseUrl } from './scripts/remark-base-url.mjs';

export default defineConfig({
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
        // ── AFML Chapter-based module groups ──
        {
          label: 'Ch 2: Data Structures',
          items: [
            { slug: 'modules/data-structures' },
            { slug: 'modules/filters' },
            { slug: 'modules/etf-trick' },
          ],
        },
        {
          label: 'Ch 3: Labeling',
          items: [
            { slug: 'modules/labeling' },
            { slug: 'modules/bet-sizing' },
          ],
        },
        {
          label: 'Ch 4: Sample Weights',
          items: [
            { slug: 'modules/sampling' },
            { slug: 'modules/sample-weights' },
            { slug: 'modules/sb-bagging' },
          ],
        },
        {
          label: 'Ch 5: Fractional Differentiation',
          items: [
            { slug: 'modules/fracdiff' },
          ],
        },
        {
          label: 'Ch 6: Ensemble Methods',
          items: [
            { slug: 'modules/ensemble-methods' },
          ],
        },
        {
          label: 'Ch 7: Cross-Validation',
          items: [
            { slug: 'modules/cross-validation' },
          ],
        },
        {
          label: 'Ch 8: Feature Importance',
          items: [
            { slug: 'modules/feature-importance' },
            { slug: 'modules/fingerprint' },
          ],
        },
        {
          label: 'Ch 9: Hyperparameter Tuning',
          items: [
            { slug: 'modules/hyperparameter-tuning' },
          ],
        },
        {
          label: 'Ch 10–12: Backtesting',
          items: [
            { slug: 'modules/backtesting-engine' },
            { slug: 'modules/synthetic-backtesting' },
          ],
        },
        {
          label: 'Ch 14–15: Diagnostics & Risk',
          items: [
            { slug: 'modules/backtest-statistics' },
            { slug: 'modules/risk-metrics' },
            { slug: 'modules/strategy-risk' },
          ],
        },
        {
          label: 'Ch 16: Portfolio Construction',
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
          items: [
            { slug: 'modules/structural-breaks' },
            { slug: 'modules/microstructural-features' },
            { slug: 'modules/codependence' },
          ],
        },
        {
          label: 'Ch 20–22: HPC & Advanced',
          items: [
            { slug: 'modules/hpc-parallel' },
            { slug: 'modules/combinatorial-optimization' },
            { slug: 'modules/streaming-hpc' },
          ],
        },
        {
          label: 'Shared Utilities',
          items: [
            { slug: 'modules/ef3m' },
            { slug: 'modules/util-fast-ewma' },
            { slug: 'modules/util-volatility' },
          ],
        },
        // ── Python-only modules ──
        {
          label: 'Python Modules',
          items: [
            { slug: 'modules/data' },
            { slug: 'modules/feature-diagnostics' },
            { slug: 'modules/pipeline' },
            { slug: 'modules/research' },
            { slug: 'modules/adapters' },
            { slug: 'modules/viz' },
          ],
        },
        // ── Reference ──
        {
          label: 'Reference',
          items: [
            { slug: 'module-reference/api-surfaces' },
            { slug: 'module-reference/by-afml-chapter' },
            { slug: 'module-reference/indexing-and-discovery' },
            { label: 'All Modules', link: '/modules/' },
          ],
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
    '/module': '/modules/',
  },
});

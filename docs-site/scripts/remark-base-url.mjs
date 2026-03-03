/**
 * Remark plugin that prepends the Astro `base` path to absolute markdown links
 * so they work correctly when the site is deployed under a subpath (e.g. /openquant).
 *
 * Only rewrites links that start with "/" and don't already start with the base.
 * Does not touch external URLs, anchors, or relative paths.
 */
import { visit } from 'unist-util-visit';

export function remarkBaseUrl({ base = '/' } = {}) {
  const prefix = base.replace(/\/+$/, '');
  if (!prefix) return () => {};

  return () => (tree) => {
    visit(tree, 'link', (node) => {
      if (
        typeof node.url === 'string' &&
        node.url.startsWith('/') &&
        !node.url.startsWith(prefix + '/')
      ) {
        node.url = prefix + node.url;
      }
    });
  };
}

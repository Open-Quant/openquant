/** Name only the scroll regions created by Expressive Code's overflow observer. */
export function installCodeScrollLabels(document = globalThis.document, Observer = globalThis.MutationObserver) {
  const key = Symbol.for('openquant.code-scroll-labels');
  if (document[key]) {
    document[key]();
    return;
  }
  const selector = '.expressive-code pre[data-oq-scroll-label]';
  const owned = new WeakMap();
  const sync = (node) => {
    if (!node.matches?.(selector)) return;
    const previous = owned.get(node);
    const authored = node.hasAttribute('aria-labelledby') ||
      (node.hasAttribute('aria-label') && node.getAttribute('aria-label') !== previous);
    if (node.getAttribute('role') !== 'region' || authored) {
      owned.delete(node);
      if (previous !== undefined && node.getAttribute('aria-label') === previous) {
        node.removeAttribute('aria-label');
      }
      return;
    }
    if (previous !== undefined && node.getAttribute('aria-label') === previous) return;
    const label = node.getAttribute('data-oq-scroll-label')?.trim();
    if (label) {
      owned.set(node, label);
      node.setAttribute('aria-label', label);
    }
  };
  const scan = (root) => {
    sync(root);
    root.querySelectorAll?.(selector).forEach(sync);
  };
  const observer = new Observer((records) => {
    for (const record of records) {
      if (record.type === 'attributes') sync(record.target);
      else record.addedNodes.forEach(scan);
    }
  });
  let body;
  const refresh = () => {
    if (document.body !== body) {
      observer.disconnect();
      body = document.body;
      if (body) observer.observe(body, {
        subtree: true, childList: true, attributes: true,
        attributeFilter: ['role', 'aria-label', 'aria-labelledby'],
      });
    }
    if (body) scan(body);
  };
  document[key] = refresh;
  document.addEventListener('astro:page-load', refresh);
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', refresh, { once: true });
  }
  refresh();
}

export function codeScrollLabels() {
  return {
    name: 'Named code scroll regions',
    hooks: {
      postprocessRenderedBlock: ({ codeBlock, renderData }) => {
        const language = codeBlock.language;
        const kinds = { python: 'Python code', rust: 'Rust code', bash: 'Shell commands',
          sh: 'Shell commands', text: 'Text block', plaintext: 'Text block',
          json: 'JSON block', toml: 'TOML configuration' };
        const kind = kinds[language] ?? (language ? `${language} code` : 'Code block');
        const title = String(codeBlock.props.title ?? '').replace(/\s+/g, ' ').trim();
        const label = title ? `${title} — ${kind}` : kind;
        const visit = (node) => {
          if (node.type === 'element' && node.tagName === 'pre') {
            node.properties['data-oq-scroll-label'] = label;
          }
          node.children?.forEach(visit);
        };
        visit(renderData.blockAst);
      },
    },
    jsModules: [`(${installCodeScrollLabels.toString()})();`],
  };
}

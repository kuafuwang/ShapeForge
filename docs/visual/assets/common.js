/* Shared data and path helpers. No HTTP requests: file:// must work offline. */
(() => {
  'use strict';
  const payload = window.SF_DOC_VISUAL || { manifest: {}, docs: [] };
  const categories = [
    ['overview', '项目入口'],
    ['tutorials', '新手与专业教程'],
    ['architecture', '架构与协议'],
    ['harness', '可信 harness'],
    ['research', '论文与研究'],
    ['operations', '运维与安全'],
    ['archive', '归档计划'],
  ];
  const labels = Object.fromEntries(categories);

  function normalize(path) {
    const parts = [];
    for (const part of String(path).split('/')) {
      if (!part || part === '.') continue;
      if (part === '..') parts.pop();
      else parts.push(part);
    }
    return parts.join('/');
  }

  function dirname(path) {
    const at = path.lastIndexOf('/');
    return at < 0 ? '' : path.slice(0, at);
  }

  function resolveDocPath(docPath, target) {
    const clean = target.replace(/^<|>$/g, '');
    if (clean.startsWith('/')) return normalize(clean.slice(1));
    return normalize(`${dirname(docPath)}/${clean}`);
  }

  function relativePath(fromDir, toPath) {
    const from = fromDir ? fromDir.split('/') : [];
    const to = toPath.split('/');
    let common = 0;
    while (common < from.length && common < to.length && from[common] === to[common]) common += 1;
    return [...from.slice(common).map(() => '..'), ...to.slice(common)].join('/') || '.';
  }

  function sourceHref(docPath, target) {
    const resolved = resolveDocPath(docPath, target);
    return relativePath('docs/visual', resolved);
  }

  function docHref(docPath, anchor = '') {
    return `reader.html?doc=${encodeURIComponent(docPath)}${anchor ? `#${anchor}` : ''}`;
  }

  function escapeHTML(value) {
    return String(value).replace(/[&<>"']/g, (ch) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[ch]));
  }

  function formatBytes(bytes) {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KiB`;
    return `${(bytes / 1024 / 1024).toFixed(2)} MiB`;
  }

  window.SFVisual = { payload, docs: payload.docs, manifest: payload.manifest, categories, labels, normalize, dirname, resolveDocPath, relativePath, sourceHref, docHref, escapeHTML, formatBytes };
})();

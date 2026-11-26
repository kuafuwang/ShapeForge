(() => {
  'use strict';
  const { docs, labels, docHref, sourceHref, resolveDocPath, relativePath, formatBytes } = window.SFVisual;
  const content = document.querySelector('#content');
  const errorBox = document.querySelector('#render-error');
  const docId = new URLSearchParams(location.search).get('doc');
  const doc = docs.find((item) => item.id === docId);

  function fail(message) {
    errorBox.hidden = false;
    errorBox.textContent = message;
  }

  if (!doc) {
    document.querySelector('#doc-title').textContent = '找不到这篇文档';
    content.textContent = `未知文档：${docId || '(没有给 doc 参数)'}。请回文档库重新选择。`;
    return;
  }

  document.title = `${doc.title} · ShapeForge`;
  document.querySelector('#doc-category').textContent = labels[doc.category] || doc.category;
  document.querySelector('#doc-title').textContent = doc.title;
  const path = document.querySelector('#doc-path');
  path.textContent = doc.path;
  path.title = doc.path;
  const meta = document.querySelector('#doc-meta');
  for (const text of [`${doc.lines} 行`, formatBytes(doc.bytes), `${doc.headings.length} 个标题`, doc.mermaid ? `${doc.mermaid} 张 Mermaid 图` : '无 Mermaid 图']) {
    const span = document.createElement('span');
    span.textContent = text;
    meta.append(span);
  }
  document.querySelector('#source-link').href = relativePath('docs/visual', doc.path);

  function extractMermaid(text) {
    const blocks = [];
    const output = [];
    let current = null;
    for (const line of text.split('\n')) {
      if (current === null && line.trim().startsWith('```mermaid')) {
        current = [];
        output.push(`SF-MERMAID-${blocks.length}-TOKEN`);
      } else if (current !== null && line.trim().startsWith('```')) {
        blocks.push(current.join('\n'));
        current = null;
      } else if (current !== null) {
        current.push(line);
      } else {
        output.push(line);
      }
    }
    if (current !== null) {
      output.push(...current);
    }
    return { text: output.join('\n'), blocks };
  }

  function withoutFrontmatter(text) {
    if (!text.startsWith('---\n')) return text;
    const end = text.indexOf('\n---\n', 4);
    return end < 0 ? text : text.slice(end + 5);
  }

  function renderToc() {
    const toc = document.querySelector('#toc');
    const entries = doc.headings.filter((heading) => heading.depth > 1);
    if (!entries.length) {
      toc.textContent = '本文只有一个标题。';
      return;
    }
    for (const heading of entries) {
      const link = document.createElement('a');
      link.href = `#${heading.anchor}`;
      link.className = `d${heading.depth}`;
      link.textContent = heading.title;
      toc.append(link);
    }
  }

  function resolveLinks() {
    for (const anchor of content.querySelectorAll('a[href]')) {
      const raw = anchor.getAttribute('href');
      if (/^(https?:|mailto:)/i.test(raw)) {
        anchor.target = '_blank';
        anchor.rel = 'noopener noreferrer';
        continue;
      }
      if (raw.startsWith('#') || raw === '') continue;
      const [target, anchorPart] = raw.split('#');
      if (!target) continue;
      if (/\.md(?:$|\?)/i.test(target)) {
        const resolved = resolveDocPath(doc.path, target);
        anchor.href = docs.some((item) => item.id === resolved) ? docHref(resolved, anchorPart || '') : sourceHref(doc.path, target);
      } else if (!/^[a-z][a-z0-9+.-]*:/i.test(raw)) {
        anchor.href = sourceHref(doc.path, raw);
      }
    }
    for (const image of content.querySelectorAll('img[src]')) {
      const raw = image.getAttribute('src');
      if (!/^(https?:|data:)/i.test(raw)) image.src = sourceHref(doc.path, raw);
    }
  }

  function renderMermaidSources(blocks) {
    for (const [index, source] of blocks.entries()) {
      const token = `SF-MERMAID-${index}-TOKEN`;
      const paragraph = [...content.querySelectorAll('p')].find((node) => node.textContent.trim() === token);
      if (!paragraph) continue;
      const holder = document.createElement('div');
      holder.className = 'mermaid';
      holder.textContent = source;
      paragraph.replaceWith(holder);
    }
  }

  async function render() {
    const extracted = extractMermaid(withoutFrontmatter(doc.content));
    const html = window.marked.parse(extracted.text, { gfm: true, breaks: false });
    content.innerHTML = window.DOMPurify.sanitize(html, { USE_PROFILES: { html: true } });
    const headings = content.querySelectorAll('h1, h2, h3, h4, h5, h6');
    headings.forEach((node, index) => {
      if (doc.headings[index]) node.id = doc.headings[index].anchor;
    });
    renderToc();
    resolveLinks();
    renderMermaidSources(extracted.blocks);

    if (extracted.blocks.length) {
      window.mermaid.initialize({ startOnLoad: false, securityLevel: 'strict', theme: 'dark', fontFamily: '-apple-system, BlinkMacSystemFont, Segoe UI, PingFang SC, sans-serif', themeVariables: { background: '#0b1728', primaryColor: '#16243a', primaryTextColor: '#e7eef9', primaryBorderColor: '#4dd8ff', lineColor: '#a4b4ce', secondaryColor: '#16243a', tertiaryColor: '#16243a' } });
      try {
        await window.mermaid.run({ nodes: content.querySelectorAll('.mermaid') });
      } catch (error) {
        fail(`Mermaid 渲染失败：${error.message || error}`);
      }
    }
    if (location.hash) {
      requestAnimationFrame(() => document.getElementById(location.hash.slice(1))?.scrollIntoView());
    }
  }

  function clearHighlights() {
    for (const mark of content.querySelectorAll('mark[data-search]')) {
      mark.replaceWith(document.createTextNode(mark.textContent));
    }
    content.normalize();
  }

  function highlight(query) {
    clearHighlights();
    const result = document.querySelector('#within-result');
    if (!query) {
      result.textContent = '';
      return;
    }
    const needle = query.toLocaleLowerCase();
    const walker = document.createTreeWalker(content, NodeFilter.SHOW_TEXT, {
      acceptNode(node) {
        if (!node.data.toLocaleLowerCase().includes(needle)) return NodeFilter.FILTER_REJECT;
        if (node.parentElement.closest('script, style, .mermaid, mark')) return NodeFilter.FILTER_REJECT;
        return NodeFilter.FILTER_ACCEPT;
      },
    });
    const nodes = [];
    while (walker.nextNode()) nodes.push(walker.currentNode);
    let count = 0;
    let first = null;
    for (const node of nodes) {
      const text = node.data;
      const lower = text.toLocaleLowerCase();
      const fragment = document.createDocumentFragment();
      let start = 0;
      let at = lower.indexOf(needle);
      while (at >= 0 && count < 200) {
        fragment.append(document.createTextNode(text.slice(start, at)));
        const mark = document.createElement('mark');
        mark.dataset.search = '1';
        mark.textContent = text.slice(at, at + needle.length);
        fragment.append(mark);
        first ||= mark;
        count += 1;
        start = at + needle.length;
        at = lower.indexOf(needle, start);
      }
      fragment.append(document.createTextNode(text.slice(start)));
      node.replaceWith(fragment);
      if (count >= 200) break;
    }
    result.textContent = `${count}${count >= 200 ? '+' : ''} 处匹配`;
    first?.scrollIntoView({ block: 'center' });
  }

  const layout = document.querySelector('.reader-layout');
  const sidebarButtons = {
    left: document.querySelector('#toggle-left'),
    right: document.querySelector('#toggle-right'),
  };
  const sidebarState = { left: true, right: true };
  try {
    const saved = JSON.parse(localStorage.getItem('sf-doc-reader-sidebars') || '{}');
    if (typeof saved.left === 'boolean') sidebarState.left = saved.left;
    if (typeof saved.right === 'boolean') sidebarState.right = saved.right;
  } catch (_error) {
    // file:// or privacy mode may disable storage; toggles still work for this page.
  }

  function applySidebarState() {
    for (const side of ['left', 'right']) {
      const expanded = sidebarState[side];
      layout.classList.toggle(`hide-${side}`, !expanded);
      const button = sidebarButtons[side];
      button.textContent = expanded ? `隐藏${side === 'left' ? '左' : '右'}栏` : `展开${side === 'left' ? '左' : '右'}栏`;
      button.setAttribute('aria-expanded', String(expanded));
      button.title = expanded ? `收起${side === 'left' ? '目录' : '信息'}栏` : `展开${side === 'left' ? '目录' : '信息'}栏`;
    }
    try {
      localStorage.setItem('sf-doc-reader-sidebars', JSON.stringify(sidebarState));
    } catch (_error) {
      // Storage is optional.
    }
  }

  for (const side of ['left', 'right']) {
    sidebarButtons[side].addEventListener('click', () => {
      sidebarState[side] = !sidebarState[side];
      applySidebarState();
    });
  }
  applySidebarState();

  let searchTimer = null;
  document.querySelector('#within-doc').addEventListener('input', (event) => {
    clearTimeout(searchTimer);
    searchTimer = setTimeout(() => highlight(event.target.value.trim()), 180);
  });
  document.querySelector('#toggle-wide').addEventListener('click', () => {
    document.body.classList.toggle('wide');
  });
  document.querySelector('#print-doc').addEventListener('click', () => window.print());

  render().catch((error) => {
    fail(`文档渲染失败：${error.message || error}`);
    content.textContent = doc.content;
  });
})();

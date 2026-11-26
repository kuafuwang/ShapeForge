(() => {
  'use strict';
  const { docs, manifest, categories, labels, docHref, formatBytes } = window.SFVisual;
  const search = document.querySelector('#search');
  const results = document.querySelector('#results');
  const empty = document.querySelector('#empty');
  const count = document.querySelector('#result-count');
  const activeFilter = document.querySelector('#active-filter');
  let category = 'all';

  document.querySelector('#doc-count').textContent = manifest.document_count;
  document.querySelector('#mermaid-count').textContent = manifest.mermaid_count;
  document.querySelector('#digest').textContent = manifest.source_digest.slice(0, 12);

  const categoryRow = document.querySelector('#categories');
  for (const [id, label] of [['all', '全部'], ...categories]) {
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'category-pill';
    button.dataset.category = id;
    button.textContent = id === 'all' ? `${label} ${docs.length}` : `${label} ${manifest.categories[id] || 0}`;
    button.addEventListener('click', () => {
      category = id;
      for (const node of categoryRow.children) node.classList.toggle('active', node.dataset.category === id);
      render();
    });
    categoryRow.append(button);
  }
  categoryRow.firstElementChild.classList.add('active');

  function card(doc) {
    const link = document.createElement('a');
    link.className = 'doc-card';
    link.href = docHref(doc.id);
    const cat = document.createElement('span');
    cat.className = 'cat';
    cat.textContent = labels[doc.category] || doc.category;
    const title = document.createElement('h3');
    title.textContent = doc.title;
    const excerpt = document.createElement('p');
    excerpt.textContent = doc.excerpt || doc.path;
    const footer = document.createElement('footer');
    const path = document.createElement('span');
    path.textContent = doc.path;
    path.title = doc.path;
    const meta = document.createElement('span');
    meta.textContent = `${doc.lines} 行 · ${formatBytes(doc.bytes)}${doc.mermaid ? ` · ${doc.mermaid} 图` : ''}`;
    footer.append(path, meta);
    link.append(cat, title, excerpt, footer);
    return link;
  }

  function render() {
    const query = search.value.trim().toLocaleLowerCase();
    const filtered = docs.filter((doc) => {
      if (category !== 'all' && doc.category !== category) return false;
      if (!query) return true;
      return `${doc.title}\n${doc.path}\n${doc.content}`.toLocaleLowerCase().includes(query);
    });
    results.replaceChildren(...filtered.map(card));
    count.textContent = `${filtered.length} / ${docs.length} 篇`;
    activeFilter.textContent = query ? `正文包含：${search.value.trim()}` : '';
    empty.hidden = filtered.length > 0;
  }

  search.addEventListener('input', render);
  document.querySelector('#clear-search').addEventListener('click', () => {
    search.value = '';
    category = 'all';
    for (const node of categoryRow.children) node.classList.toggle('active', node.dataset.category === 'all');
    render();
    search.focus();
  });
  document.addEventListener('keydown', (event) => {
    if (event.key === '/' && document.activeElement !== search) {
      event.preventDefault();
      search.focus();
    }
    if (event.key === 'Escape' && document.activeElement === search) {
      search.value = '';
      render();
      search.blur();
    }
  });
  render();
})();

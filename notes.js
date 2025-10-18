const POSTS_PER_PAGE = 5;

function loadPosts() {
  const source = Array.isArray(window.POSTS) ? window.POSTS : [];
  const posts = source
    .map((p) => ({ ...p, _ts: Date.parse(p.date) || 0 }))
    .sort((a, b) => b._ts - a._ts);
  return posts;
}

function renderPosts(posts, page = 1) {
  const list = document.getElementById('posts');
  const start = (page - 1) * POSTS_PER_PAGE;
  const pageItems = posts.slice(start, start + POSTS_PER_PAGE);

  if (pageItems.length === 0) {
    list.innerHTML = '<p class="muted">No notes yet.</p>';
    renderPagination(posts.length, page);
    return;
  }

  list.innerHTML = pageItems
    .map((p) => {
      const date = p._ts ? new Date(p._ts) : null;
      const dateLabel = date ? date.toISOString().slice(0, 10) : p.date;
      return `
      <a class="post-card" href="post.html?slug=${encodeURIComponent(p.slug)}" aria-label="${p.title}">
        <div class="post-card__head">
          <span class="chip">${p.category}</span>
          <time class="muted" datetime="${date ? date.toISOString() : p.date}">${dateLabel}</time>
        </div>
        <h3 class="post-card__title">${p.title}</h3>
        <p class="post-card__summary">${p.summary}</p>
      </a>
    `;
    })
    .join('');

  renderPagination(posts.length, page);
}

function renderPagination(total, page) {
  const totalPages = Math.max(1, Math.ceil(total / POSTS_PER_PAGE));
  const el = document.getElementById('pagination');
  if (totalPages <= 1) {
    el.innerHTML = '';
    return;
  }
  const nums = Array.from({ length: totalPages }, (_, i) => i + 1)
    .map((n) => `<button class="page-btn ${n === page ? 'active' : ''}" data-page="${n}">${n}</button>`) 
    .join('');
  el.innerHTML = `
    <button class="page-btn" data-page="prev" ${page === 1 ? 'disabled' : ''}>Prev</button>
    ${nums}
    <button class="page-btn" data-page="next" ${page === totalPages ? 'disabled' : ''}>Next</button>
  `;
}

(function initNotes() {
  const tabs = document.querySelectorAll('.tab');
  const all = loadPosts();
  let currentCategory = 'all';
  let currentPage = 1;

  function filtered() {
    return currentCategory === 'all' ? all : all.filter((p) => p.category === currentCategory);
  }

  renderPosts(filtered(), currentPage);

  tabs.forEach((btn) => {
    btn.addEventListener('click', () => {
      tabs.forEach((b) => b.classList.remove('active'));
      btn.classList.add('active');
      currentCategory = btn.dataset.category;
      currentPage = 1;
      renderPosts(filtered(), currentPage);
    });
  });

  document.getElementById('pagination').addEventListener('click', (e) => {
    const target = e.target.closest('button.page-btn');
    if (!target) return;
    const totalPages = Math.max(1, Math.ceil(filtered().length / POSTS_PER_PAGE));
    const val = target.dataset.page;
    if (val === 'prev') currentPage = Math.max(1, currentPage - 1);
    else if (val === 'next') currentPage = Math.min(totalPages, currentPage + 1);
    else currentPage = parseInt(val, 10);
    renderPosts(filtered(), currentPage);
    window.scrollTo({ top: 0, behavior: 'smooth' });
  });
})();

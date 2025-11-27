const POSTS_PER_PAGE = 5;
const thumbnailCache = new Map();
const thumbnailsSupported = typeof fetch === 'function' && typeof window !== 'undefined';

function loadPosts() {
  const source = Array.isArray(window.POSTS) ? window.POSTS : [];
  const posts = source
    .map((p) => ({ ...p, _ts: Date.parse(p.date) || 0 }))
    .sort((a, b) => b._ts - a._ts);
  return posts;
}

function resolveImageSrc(value, slug) {
  if (!value) return null;
  const raw = value.trim();
  if (!raw || raw.startsWith('#') || raw.startsWith('blob:') || raw.startsWith('data:')) {
    return null;
  }
  if (/^(?:[a-z]+:)?\/\//i.test(raw) || raw.startsWith('/')) {
    return raw;
  }

  const noteBase = new URL(`notes/${slug}.md`, window.location.href);
  const assetBase = new URL(`notes-assets/${slug}/`, window.location.href);
  const siteBase = new URL('./', window.location.href);

  try {
    let resolved;
    if (raw.startsWith('@assets/')) {
      resolved = new URL(raw.slice(8), assetBase);
    } else if (raw.startsWith('notes-assets/')) {
      resolved = new URL(raw, siteBase);
    } else if (raw.startsWith('./') || raw.startsWith('../')) {
      resolved = new URL(raw, noteBase);
    } else if (raw.includes('/')) {
      resolved = new URL(raw, noteBase);
    } else {
      resolved = new URL(raw, assetBase);
    }
    return resolved.href;
  } catch (err) {
    console.warn('Unable to resolve thumbnail src', raw, err);
    return null;
  }
}

async function fetchFirstImage(slug) {
  if (!thumbnailsSupported) return null;
  if (thumbnailCache.has(slug)) return thumbnailCache.get(slug);
  let thumbnail = null;
  try {
    const res = await fetch(`notes/${slug}.md`, { cache: 'no-store' });
    if (!res.ok) throw new Error(`missing markdown for ${slug}`);
    const markdown = await res.text();
    const match = markdown.match(/!\[[^\]]*]\(([^)\s]+)[^)]*\)/);
    if (match && match[1]) {
      thumbnail = resolveImageSrc(match[1], slug);
    }
  } catch (err) {
    console.warn('Thumbnail fetch failed', slug, err);
  }
  thumbnailCache.set(slug, thumbnail);
  return thumbnail;
}

function applyThumbnails(items) {
  if (!thumbnailsSupported) return;
  items.forEach((p) => {
    fetchFirstImage(p.slug)
      .then((src) => {
        if (!src) return;
        const selectorSlug = (p.slug || '').replace(/"/g, '\\"');
        const card = document.querySelector(`.post-card[data-slug="${selectorSlug}"]`);
        if (!card) return;
        const thumb = card.querySelector('.post-card__thumb');
        if (!thumb) return;
        thumb.innerHTML = `<img src="${src}" alt="" loading="lazy" decoding="async" />`;
        card.classList.add('has-thumb');
      })
      .catch((err) => console.warn('Thumbnail apply failed', p.slug, err));
  });
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
      <a class="post-card" data-slug="${p.slug}" href="post.html?slug=${encodeURIComponent(p.slug)}" aria-label="${p.title}">
        <div class="post-card__thumb" aria-hidden="true"></div>
        <div class="post-card__body">
          <div class="post-card__head">
            <span class="chip">${p.category}</span>
            <time class="muted" datetime="${date ? date.toISOString() : p.date}">${dateLabel}</time>
          </div>
          <h3 class="post-card__title">${p.title}</h3>
          <p class="post-card__summary">${p.summary}</p>
        </div>
      </a>
    `;
    })
    .join('');

  renderPagination(posts.length, page);
  applyThumbnails(pageItems);
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

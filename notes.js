const POSTS_PER_PAGE = 5;
const mediaCache = new Map();
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

function extractImages(markdown, slug) {
  const regex = /!\[[^\]]*]\(([^)\s]+)(?:\s+"[^"]*")?\)/g;
  const out = [];
  let m;
  while (out.length < 3 && (m = regex.exec(markdown))) {
    const resolved = resolveImageSrc(m[1], slug);
    if (resolved) out.push(resolved);
  }
  return out;
}

function loadImageMeta(src) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => {
      const ratio = img.naturalWidth && img.naturalHeight ? img.naturalWidth / img.naturalHeight : 1;
      resolve({ src, ratio });
    };
    img.onerror = reject;
    img.src = src;
  });
}

async function fetchImages(slug) {
  if (!thumbnailsSupported) return [];
  if (mediaCache.has(slug)) return mediaCache.get(slug);
  let metas = [];
  try {
    const res = await fetch(`notes/${slug}.md`, { cache: 'no-store' });
    if (!res.ok) throw new Error(`missing markdown for ${slug}`);
    const markdown = await res.text();
    const sources = extractImages(markdown, slug);
    const loaded = await Promise.all(
      sources.map((src) =>
        loadImageMeta(src).catch(() => null)
      )
    );
    metas = loaded.filter(Boolean).slice(0, 3);
  } catch (err) {
    console.warn('Media fetch failed', slug, err);
  }
  mediaCache.set(slug, metas);
  return metas;
}

function renderMedia(card, metas) {
  const mediaSlot = card.querySelector('.post-card__media');
  if (!mediaSlot) return;
  card.classList.remove('media-banner', 'media-inline', 'media-collage');
  mediaSlot.innerHTML = '';
  if (!metas || metas.length === 0) return;

  if (metas.length === 1) {
    const { src, ratio } = metas[0];
    if (ratio > 1.2) {
      card.classList.add('media-banner');
      mediaSlot.innerHTML = `<div class="thumb thumb-banner"><img src="${src}" alt="" loading="lazy" decoding="async" /></div>`;
    } else {
      card.classList.add('media-inline');
      mediaSlot.innerHTML = `<div class="thumb thumb-inline"><img src="${src}" alt="" loading="lazy" decoding="async" /></div>`;
    }
    return;
  }

  const items = metas.slice(0, 3);
  card.classList.add('media-collage');
  mediaSlot.innerHTML = `
    <div class="thumb thumb-collage">
      <div class="collage-grid">
        ${items
          .map(
            (m) =>
              `<div class="collage-cell"><img src="${m.src}" alt="" loading="lazy" decoding="async" /></div>`
          )
          .join('')}
      </div>
    </div>
  `;
}

function applyMedia(items) {
  if (!thumbnailsSupported) return;
  items.forEach((p) => {
    const selectorSlug = (p.slug || '').replace(/"/g, '\\"');
    const card = document.querySelector(`.post-card[data-slug="${selectorSlug}"]`);
    if (!card) return;
    fetchImages(p.slug)
      .then((metas) => renderMedia(card, metas))
      .catch((err) => console.warn('Media apply failed', p.slug, err));
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
        <div class="post-card__body">
          <div class="post-card__head">
            <span class="chip">${p.category}</span>
            <time class="muted" datetime="${date ? date.toISOString() : p.date}">${dateLabel}</time>
          </div>
          <h3 class="post-card__title">${p.title}</h3>
          <p class="post-card__summary">${p.summary}</p>
        </div>
        <div class="post-card__media" aria-hidden="true"></div>
      </a>
    `;
    })
    .join('');

  renderPagination(posts.length, page);
  applyMedia(pageItems);
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
  const scopedCategory = document.body.dataset.categoryScope;
  const tabCategories = Array.from(tabs).map((tab) => tab.dataset.category);
  const defaultCategory = document.body.dataset.defaultCategory;
  let currentCategory = 'all';
  if (scopedCategory) {
    currentCategory = scopedCategory;
  } else if (defaultCategory && tabCategories.includes(defaultCategory)) {
    currentCategory = defaultCategory;
  } else if (!tabCategories.includes(currentCategory) && tabCategories.length) {
    currentCategory = tabCategories[0];
  }
  let currentPage = 1;

  function filtered() {
    if (scopedCategory) {
      return all.filter((p) => p.category === scopedCategory);
    }
    return currentCategory === 'all' ? all : all.filter((p) => p.category === currentCategory);
  }

  function syncTabs(category) {
    tabs.forEach((btn) => {
      const isActive = btn.dataset.category === category;
      btn.classList.toggle('active', isActive);
      btn.setAttribute('aria-selected', isActive ? 'true' : 'false');
    });
  }

  syncTabs(currentCategory);
  renderPosts(filtered(), currentPage);

  tabs.forEach((btn) => {
    btn.addEventListener('click', () => {
      if (scopedCategory) {
        return;
      }
      currentCategory = btn.dataset.category;
      currentPage = 1;
      syncTabs(currentCategory);
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

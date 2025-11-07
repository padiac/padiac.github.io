function loadPost(slug) {
  const posts = Array.isArray(window.POSTS) ? window.POSTS : [];
  return posts.find((p) => p.slug === slug);
}

function createMarkdownRenderer() {
  if (!window.markdownit) {
    return null;
  }
  return window.markdownit({
    html: true,
    linkify: true,
    breaks: false
  });
}

function rewriteRelativeImageSources(container, slug) {
  if (!container || !slug) return;
  const noteBase = new URL(`notes/${slug}.md`, window.location.href);
  const assetBase = new URL(`notes-assets/${slug}/`, window.location.href);
  const siteBase = new URL('./', window.location.href);

  container.querySelectorAll('img').forEach((img) => {
    const raw = img.getAttribute('src');
    if (!raw) return;
    const value = raw.trim();
    if (
      !value ||
      value.startsWith('#') ||
      value.startsWith('data:') ||
      value.startsWith('blob:') ||
      /^(?:[a-z]+:)?\/\//i.test(value) ||
      value.startsWith('/')
    ) {
      return;
    }
    try {
      let resolved;
      if (value.startsWith('@assets/')) {
        resolved = new URL(value.slice(8), assetBase);
      } else if (value.startsWith('notes-assets/')) {
        resolved = new URL(value, siteBase);
      } else if (value.startsWith('./') || value.startsWith('../')) {
        resolved = new URL(value, noteBase);
      } else if (value.includes('/')) {
        resolved = new URL(value, noteBase);
      } else {
        resolved = new URL(value, assetBase);
      }
      img.setAttribute('src', resolved.href);
    } catch (err) {
      console.warn('Unable to resolve image path', value, err);
    }
  });
}

function enhanceImages(container) {
  if (!container) return;
  container.querySelectorAll('img').forEach((img) => {
    if (!img.getAttribute('loading')) {
      img.setAttribute('loading', 'lazy');
    }
    img.setAttribute('decoding', 'async');
  });
}

async function renderPostContent(post, container) {
  const renderer = createMarkdownRenderer();
  if (!renderer) {
    container.innerHTML = '<p class="muted">Markdown renderer unavailable.</p>';
    return;
  }

  container.innerHTML = '<p class="muted">Loading...</p>';
  try {
    const res = await fetch(`notes/${post.slug}.md`, { cache: 'no-store' });
    if (!res.ok) throw new Error(`Missing markdown for slug ${post.slug}`);
    const markdown = await res.text();
    container.innerHTML = renderer.render(markdown);
    rewriteRelativeImageSources(container, post.slug);
    enhanceImages(container);
    if (window.queueMathJax) {
      window.queueMathJax(container);
    }
  } catch (err) {
    console.error(err);
    container.innerHTML = '<p class="muted">Content is not available right now.</p>';
  }
}

(async function initPost() {
  const params = new URLSearchParams(location.search);
  const slug = params.get('slug');
  if (!slug) return;

  const post = loadPost(slug);
  const titleEl = document.getElementById('post-title');
  const dateEl = document.getElementById('post-date');
  const catEl = document.getElementById('post-category');
  const contentEl = document.getElementById('post-content');

  if (!post) {
    titleEl.textContent = 'Not found';
    contentEl.innerHTML = '<p>The post you are looking for does not exist.</p>';
    return;
  }

  document.title = `${post.title} - Notes`;
  titleEl.textContent = post.title;
  const timestamp = Date.parse(post.date);
  const dateLabel = Number.isFinite(timestamp) ? new Date(timestamp).toISOString().slice(0, 10) : post.date;
  dateEl.textContent = dateLabel;
  dateEl.setAttribute('datetime', Number.isFinite(timestamp) ? new Date(timestamp).toISOString() : post.date);
  catEl.textContent = post.category;

  await renderPostContent(post, contentEl);
})();

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
  dateEl.textContent = post.date;
  dateEl.setAttribute('datetime', post.date);
  catEl.textContent = post.category;

  await renderPostContent(post, contentEl);
})();

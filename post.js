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

async function renderMarkdown(post, el) {
  const renderer = createMarkdownRenderer();
  if (!renderer) {
    el.innerHTML = '<p class="muted">Markdown renderer unavailable.</p>';
    return;
  }

  el.innerHTML = '<p class="muted">Loading...</p>';
  try {
    const response = await fetch(`notes/${post.slug}.md`);
    if (!response.ok) {
      throw new Error(`Missing markdown for slug ${post.slug}`);
    }
    const markdown = await response.text();
    el.innerHTML = renderer.render(markdown);
    if (window.renderMathInElement) {
      window.renderMathInElement(el, {
        delimiters: [
          { left: '$$', right: '$$', display: true },
          { left: '\\[', right: '\\]', display: true },
          { left: '$', right: '$', display: false },
          { left: '\\(', right: '\\)', display: false }
        ],
        throwOnError: false
      });
    }
  } catch (error) {
    console.error(error);
    el.innerHTML = '<p class="muted">Content is not available right now.</p>';
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

  await renderMarkdown(post, contentEl);
})();

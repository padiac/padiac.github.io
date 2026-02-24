# AGENTS.md

## Cursor Cloud specific instructions

### Overview

This is a **zero-dependency static website** (personal portfolio + technical blog). There is no `package.json`, no build system, no test framework, and no backend. All content is vanilla HTML/CSS/JS served as static files.

### Running the dev server

```bash
python3 -m http.server 8080
```

A local HTTP server is **required** because `post.js` and `notes.js` use `fetch()` to load Markdown content from `notes/`. Opening files via `file://` will fail due to CORS restrictions.

### Key architecture notes

- **Content**: Markdown files in `notes/`, metadata in `posts-data.js` (JS array assigned to `window.POSTS`).
- **Rendering**: Client-side Markdown rendering via CDN-hosted `markdown-it`; math via CDN-hosted MathJax v3.
- **Pages**: `index.html` (homepage), `ml-notes.html` (ML notes listing), `coding-notes.html` (coding notes listing), `post.html` (individual post viewer).
- **No lint/test/build**: There are no linters, test frameworks, or build steps configured in this repository.
- **Publishing workflow**: See `codex_rules.md` for the Codex-automated note publishing process (format, classify, commit, push).

# PersonWeb - Static Blog

## Overview

Zero-dependency static website (personal portfolio + technical blog). No package.json, no build system, no tests, no backend. All content is vanilla HTML/CSS/JS served as static files.

## Dev Server

```bash
python3 -m http.server 8080
```

Required because `post.js` and `notes.js` use `fetch()` to load Markdown from `notes/`. Opening via `file://` fails due to CORS.

## Architecture

- **Content**: Markdown files in `notes/`, metadata in `posts-data.js` (`window.POSTS` array).
- **Rendering**: Client-side markdown-it + MathJax v3 (both CDN).
- **Pages**: `index.html` (home), `ml-notes.html` (ML notes), `coding-notes.html` (coding notes), `post.html` (post viewer).
- **Git repo root**: `E:\Repo\PersonWeb\PersonWeb\` (not the outer `PersonWeb\`).

---

## Note Publishing Workflow

When the user pastes raw Markdown to publish, execute all steps automatically:

1. Normalize Markdown per the math and formatting rules below.
2. Detect or reuse category from `posts-data.js`.
3. Derive `title`, `slug`, `summary` (category-prefixed slug and title).
4. Remove the top-level H1 from saved Markdown (site renders title from metadata).
5. Write to `notes/<slug>.md`.
6. Update `posts-data.js` (prepend new entry or update existing).
7. `git add`, `git commit -m "add: <slug>"`, `git push`.
8. Print completion summary.

### Title, Slug, Summary

- **Title**: First `# ` line is the base title. Final title in posts-data.js: `<Category> - <BaseTitle>`.
- **Slug**: `<category-lower-hyphen>-<title-slug>` (e.g., `statistics-normal-approximation`).
- **Summary**: First 1-2 sentences of body, no formulas, max 160 chars.
- **Date**: Current UTC time in ISO 8601, generated at publish time.
- Inline overrides (`@title:`, `@slug:`, `@category:`, `@summary:`) are stripped before saving.

### Category Detection

1. Collect distinct categories from `posts-data.js`.
2. Use `@category:` if provided.
3. Otherwise classify by keyword counts:
   - **Statistics**: variance, estimator, MVUE, MLE, likelihood, hypothesis, distribution, probability, CLT, Bayes, posterior, prior.
   - **Machine Learning**: regression, SVM, CNN, RNN, YOLO, training, gradient, loss, overfitting, cross-validation, dataset, feature.
   - **LLM**: LLM, transformer, attention, tokenizer, prompt, embedding, RAG.
   - **Flow Matching**: flow matching, rectified flow, probability flow ODE, CNF, neural ODE.
4. Highest match count wins. No match -> `Misc`.

### posts-data.js Entry Format

```js
{
  slug: '<slug>',
  title: '<Category> - <BaseTitle>',
  date: 'YYYY-MM-DDTHH:MM:SSZ',
  category: '<Category>',
  summary: '<1-2 sentence summary, no formulas>'
}
```

Prepend new entries (newest first). Preserve `window.POSTS = [ ... ];` wrapper.

---

## Site Markdown Rules

- **Language**: Preserve input language (Chinese stays Chinese, English stays English).
- **Headings**: `##` and `###` in body. H1 is removed (title comes from metadata).
- **Spacing**: One blank line before/after headings and between paragraphs.
- **Lists**: `-` for unordered, `1.` for ordered.
- **Punctuation**: Half-width (English) punctuation only, no CJK punctuation mixing.
- **No HTML** or unusual syntax.

### Math Rules (Critical)

- **Inline**: `$...$`
- **Block**: `$$ ... $$` on a **single logical line**, no internal newlines.
- **Escape ALL underscores** in math as `\_`. Never use raw `_` or `_{...}`.
  - Example: `$\sum\_{i=1}^{n} x\_i$` not `$\sum_{i=1}^{n} x_i$`.
- **No `\|`** — use `\lvert`, `\rvert`, or `\Vert` instead.
- **No spacing commands**: no `\,`, `\!`, `\;`, `\:` anywhere in math.
- **No `\mathbb`** — use plain letters (e.g., `E\_X` not `\mathbb{E}\_X`).
- **No `\underbrace`** — replace with `\text{(...)}` annotations.
- **No `\boxed{...}`** — keep content, remove wrapper.
- **Replace `*` with `\ast`** inside math.
- **Barred Greek**: `\bar\alpha\_t` not `\bar{\alpha}\_t` (no braces).
- **`\left`/`\right`**: prefer `\bigl`/`\bigr` to avoid rendering issues.
- **One blank line** before and after block equations.
- **Never put equations inside fenced code blocks**.
- Multi-step derivations: separate `$$` blocks with prose between them, not stacked inside one `$$`.
- `\begin{cases}` is OK but keep on single line: `$$ f(x) = \begin{cases} a, & x > 0 \\ b, & \text{otherwise} \end{cases} $$`

### Math Normalization Example

Input:
```
$$ L_{reg} = \frac{1}{M} \sum_{i=1}^{M} (x_{i} - \hat{x}_{i})^{2} $$
```

Correct output:
```
$$ L\_{reg} = \frac{1}{M} \sum\_{i=1}^{M} (x\_{i} - \hat{x}\_{i})^{2} $$
```

---

## Completion Summary Format

```
Published
Title: <Category> - <BaseTitle>
Slug: <slug>
Category: <category>
Summary: <summary>
Date (UTC): <ISO8601>
Path: notes/<slug>.md
posts-data.js: updated
Git: committed and pushed
```

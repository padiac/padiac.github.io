# Codex Publishing Rules (Auto Execute: Format -> Classify -> Commit & Push)

> Zero-typing workflow: invoke with `@file:.codex/rules.md` and paste a raw Markdown block. Codex must execute all steps without confirmation.

---

## Purpose
When this rules file is referenced and a raw Markdown note is pasted, Codex must:
1. Normalize the Markdown to match the site's authoring rules.
2. Detect (or reuse) the category from existing `posts-data.js` categories; fall back to `Misc`.
3. Derive `title`, `slug`, and `summary` (**with category-prefixed slug and title**).
4. **Remove the top-level H1 from the saved Markdown** to avoid duplicate titles on the site.
5. Write the final note to `notes/<slug>.md`.
6. Update `posts-data.js` (add or refresh entry).
7. Run Git commands to commit and push.
8. Print a detailed completion summary.

The user will not provide any parameters beyond the pasted Markdown block.

---

## Site Markdown Rules (Must Follow)
- **Language**: Preserve the language of the input Markdown. If the pasted note is in Chinese, keep all normalization and additions in Chinese; if it is in English, keep everything in English.
- **File path**: `notes/<slug>.md` (slug must be lowercase, hyphen-joined, identical to `posts-data.js`).
- **Headings**: allow `#`, `##`, `###` in raw input, but **the first `#` H1 is only used to extract the title and must be removed from the saved Markdown**.
- **Spacing**: one blank line before/after headings and between paragraphs.
- **Lists**: `-` for unordered; `1.` for ordered.
- **Tables**: standard Markdown table syntax.
- **Math**:
- Inline: `$...$`
- Block:
  ```
  $$
  ...
  $$
  ```
  - Keep every display formula on a single logical line inside `$$ ... $$` with no internal newlines, blank lines, or indentation; collapse multi-line equations into one line so GitHub Pages renders them.
  - Escape every underscore inside math as `\_`; never use raw `_` or `_{...}`.
  - Convert `\(...\)` -> `$...$`, `\[...\]` -> `$$...$$`.
  - Remove `\boxed{...}` but keep its content.
  - Do not use spacing commands such as `\,`, `\!`, `\;`, or `\:` anywhere in math (inline or block); remove them entirely, especially inside block formulas, or the GitHub Pages renderer will fail the equation.
  - Keep multi-step derivations as separate block equations: close each `$$` after a single formula and add prose (or list items) between steps instead of stacking lines inside one pair of `$$`.
  - When writing expectations or similar operators, stick to forms like `E\_X[...]` without any spacing macros so no stray exclamation marks appear in the rendered output.
  - In the GitHub Pages kramdown + MathJax lite environment, avoid extended macros such as `\mathbb`, `\|`, and `\!`; use plain `E\_X` instead of `\mathbb{E}`, prefer `\Vert` for norms, and skip manual spacing commands so formulas render consistently across pages.
  - The GitHub Pages math renderer does not support `\underbrace` or other AMS-style extensible symbols. Replace `\underbrace{...}_{...}` with inline annotations such as `\text{(...)}`
    placed after the term to ensure the formula renders correctly.
  - Inside inline math (`$...$`), never use `\,` or `\!`. Inline math on GitHub must contain only a single uninterrupted LaTeX expression without manual spacing macros or newlines.
  - For barred Greek letters, avoid braces: use `\bar\alpha\_t` instead of `\bar{\alpha}\_t` (braces can break rendering on GitHub Pages MathJax/Katex).
  - Always replace literal `*` symbols inside math expressions with `\ast`, and escape every underscore in math as `\_`.
  - If Markdown still swallows a subscript, ensure the underscore is escaped and keep the subscript literal (e.g., `$\hat{\theta}\_E$` or `$\hat{\theta}\_ME$`). Do not use `_` or `_{...}` and do not add spacing macros.
  - Keep one blank line before and after block equations.
  - Never leave equations inside fenced code blocks; convert them into the inline or block math formats above.
- **Text style**: **bold** as `**text**`; _italic_ as `_text_`; code as `` `code` ``.
- **Punctuation**: use half-width (English) punctuation; do not mix CJK and English punctuation.
- **Links/Images**: `[Title](URL)` and `![Alt](URL)`.
- **No HTML** or unusual syntax.

## Math Normalization (GitHub Pages Stability)
Normalize the following math for GitHub Pages stability.

Rules to apply:
- Keep display math in `$$ ... $$` on a single line.
- Escape ALL subscripts by replacing `_` with `\_` inside math.
- Do NOT revert `\_` back to `_{...}`.
- Do NOT introduce line breaks or formatting changes.
- Output the math exactly as it should appear in the final Markdown.

Input math:
$$ L_{reg} = \frac{1}{M} \sum_{i=1}^{M} \left[(x_{i} - \hat{x}_{i})^{2} + (y_{i} - \hat{y}_{i})^{2} + (w_{i} - \hat{w}_{i})^{2} + (h_{i} - \hat{h}_{i})^{2}\right] $$

Normalized output:
$$ L\_{reg} = \frac{1}{M} \sum\_{i=1}^{M} \left[(x\_{i} - \hat{x}\_{i})^{2} + (y\_{i} - \hat{y}\_{i})^{2} + (w\_{i} - \hat{w}\_{i})^{2} + (h\_{i} - \hat{h}\_{i})^{2}\right] $$

---

## Title, Slug, Summary
- **Title**:
  - The first line starting with `# ` is the base title.
  - Final title stored in `posts-data.js` must be **prefixed with the category**, formatted as:  
    `<Category> - <BaseTitle>`  
    Example: `"Statistics - Normal Approximation"`.
  - This H1 must NOT be kept in the final saved Markdown (delete that line and the following blank line if present).
- **Category**:
  - If `@category:` override exists, use it.
  - Else auto-detect per "Category Detection" below.
- **Slug** (category-prefixed rule):
  - If an inline override `@slug:` exists, use it verbatim.
  - Otherwise, build as:  
    `slug = <category-lower-hyphen> + '-' + <title-slug>`  
    Example: `statistics-normal-approximation`.
- **Summary**:
  - If `@summary:` exists, use it, trimmed to <=160 characters and with formulas removed.
  - Otherwise, take the first 1-2 sentences of the first paragraph after the removed title (<=160 chars, formulas removed).
- **Strip any inline overrides** (see **Inline Overrides**) from the final Markdown prior to saving.

---

## Category Detection (Auto + Reuse Existing)
1. Read `posts-data.js` and collect distinct existing `"category"` values (e.g., `Statistics`, `Machine Learning`, `LLM`, etc.).
2. If `@category:` exists and matches an existing category, use it.
3. Otherwise, classify by **case-insensitive keyword counts** in the pasted content:
   - **Statistics**: variance, estimator, MVUE, MLE, likelihood, hypothesis, distribution, t-distribution, chi-square, F distribution, probability, CLT, Bayes, posterior, prior.
   - **Machine Learning**: regression, SVM, CNN, RNN, YOLO, training, gradient, loss, overfitting, cross-validation, dataset, feature, sklearn, PyTorch, TensorFlow, metric.
   - **LLM**: LLM, transformer, attention, tokenizer, prompt, embedding, RAG, vector DB, LangChain, agent, OpenAI, inference.
   - **Flow Matching**: flow matching, rectified flow, probability flow ODE, continuous normalizing flow, CNF, neural ODE, continuity equation, Liouville, instantaneous change of variables, ICOV.
4. Choose the category with the **highest match count among existing categories**.
5. If no keywords match any existing category, use `"Misc"`. If `"Misc"` does not exist, add it when updating `posts-data.js`.
6. If `@category:` is present but not in existing categories, still accept it and add that category to `posts-data.js`.

---

## Update `posts-data.js`
Every note must have an entry:
```js
{
  slug: "<slug>",
  title: "<Category> - <BaseTitle>",
  date: "YYYY-MM-DDTHH:MM:SSZ", // ISO 8601 UTC
  category: "<Category>",
  summary: "<1-2 sentence summary, no formulas>"
}
```
**Rules:**
- **date is the current UTC time in ISO 8601**, e.g., `"2025-10-18T12:29:01Z"`.
- Always generate the timestamp at publish time via the system clock (e.g., `Get-Date` or `date`) and convert it to UTC; never reuse a cached or pre-filled value.
- **If an entry with the same slug exists, update its fields (keep array order and surrounding module syntax intact).**
- **If none exists, append a new object to the exported array.**
- **Preserve the module wrapper**: `export const posts = [ ... ];`
- **Preserve existing indentation and commas.**

---

## Write Target Files
- Output path: `notes/<slug>.md` (overwrite if exists).
- Remove any inline overrides from the final Markdown.
- Ensure the final document complies with all "Site Markdown Rules".
- **Ensure the top-level H1 title line is removed** from the saved Markdown (only body remains).

---

## Git Commands (Always Attempt)
After writing the note and updating `posts-data.js`, execute:

> **Always show details**
```bash
git add notes/<slug>.md posts-data.js
git commit -m "add: <slug>"
git push
```

If there are no changes, skip the commit but still report "no changes".

---

## Output Summary (Print in Chat)
After all steps, print:

> **Always show details**
```
? Published
Title: <Category> - <BaseTitle>
Slug: <slug>
Category: <category>
Summary: <summary>
Date (UTC): <ISO8601>
Path: notes/<slug>.md
posts-data.js: updated
Git: committed and pushed (or no changes)
```

---

## Inline Overrides (Optional, in the Pasted Markdown; Remove Before Save)
> **Always show details**
```
@title: Custom Title
@slug: custom-slug
@category: Statistics | Machine Learning | LLM | <any>
@summary: One-two sentence description without formulas
```

---

## Zero-Typing Usage
User action in Codex:

> **Always show details**
```
@file:.codex/rules.md
<<<
# Your Note Title
(paste your raw Markdown here; optionally include @overrides above)
>>>
```

Codex must automatically: **Format -> Classify -> Build category-prefixed slug and title -> Remove H1 -> Write -> Update posts-data.js -> Commit & Push -> Print Summary**.

# Codex Publishing Rules (Auto Execute: Format → Classify → Commit & Push)

> This file defines a zero-typing workflow for publishing notes.  
> When invoked as `@file:.codex/rules.md` followed by a raw Markdown block, Codex must complete all steps without asking for confirmation.

---

## Purpose

When this rules file is referenced and a raw Markdown note is pasted, Codex must:

1. Normalize the Markdown to match the site’s authoring rules.
2. Determine the category automatically (reusing existing categories in `posts-data.js`).
3. Derive `title`, `slug`, and `summary`.
4. Write the final note to `notes/<slug>.md`.
5. Update `posts-data.js` (add or refresh entry).
6. Run Git commands to commit and push.
7. Print a brief completion summary.

The user will not provide any parameters beyond the pasted Markdown block.

---

## Site Markdown Rules (Must Follow)

- **File path**: `notes/<slug>.md` (slug must be lowercase, hyphen-joined, identical to `posts-data.js`).
- **Headings**: use `#`, `##`, `###` only.
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
  - Convert `\(...\)` → `$...$`, `\[...\]` → `$$...$$`.
  - Remove `\boxed{...}` but keep its content.
  - Keep one blank line before and after block equations.
- **Text style**: **bold** as `**text**`; _italic_ as `_text_`; code as `` `code` ``.
- **Punctuation**: use half-width (English) punctuation; do not mix CJK and English punctuations.
- **Links/Images**: `[Title](URL)` and `![Alt](URL)`.
- **No HTML** or unusual syntax.

---

## Title, Slug, Summary

- **Title**: the first line starting with `# ` is the title. If missing, synthesize a concise title from content.
- **Slug**:
  - If an inline override `@slug:` exists, use it.
  - Otherwise:  
    1) lowercase the title;  
    2) replace spaces with `-`;  
    3) drop non `[a-z0-9-]`;  
    4) compress consecutive `-`.
- **Summary**:
  - If `@summary:` exists, use it, trimmed to ≤160 characters and with formulas removed.
  - Otherwise, take the first 1–2 sentences of the first paragraph after the title (≤160 chars, formulas removed).

Strip any inline overrides from the final Markdown (see “Inline Overrides” below).

---

## Category Detection (Auto + Reuse Existing)

1. Read `posts-data.js` and collect distinct existing `"category"` values (e.g., `Statistics`, `Machine Learning`, `LLM`, etc.).
2. If `@category:` exists and matches an existing category, use it.
3. Otherwise, classify by **case-insensitive keyword counts** in the pasted content:

   - **Statistics**: `variance`, `estimator`, `MVUE`, `MLE`, `likelihood`, `hypothesis`, `distribution`, `t-distribution`, `chi-square`, `F distribution`, `probability`, `CLT`, `Bayes`, `posterior`, `prior`.
   - **Machine Learning**: `regression`, `SVM`, `CNN`, `RNN`, `YOLO`, `training`, `gradient`, `loss`, `overfitting`, `cross-validation`, `dataset`, `feature`, `sklearn`, `PyTorch`, `TensorFlow`, `metric`.
   - **LLM**: `LLM`, `transformer`, `attention`, `tokenizer`, `prompt`, `embedding`, `RAG`, `vector DB`, `LangChain`, `agent`, `OpenAI`, `inference`.

4. Choose the category with the **highest match count among existing categories**.
5. If no keywords match any existing category, use `"Misc"`. If `"Misc"` does not exist, add it when updating `posts-data.js`.

> If `@category:` is present but not in existing categories, accept it and add that category to `posts-data.js`.

---

## Write Target Files

- Output path: `notes/<slug>.md`.
- Overwrite if the file already exists.
- Remove any inline overrides (`@title:`, `@slug:`, `@summary:`, `@category:`) from the final Markdown.
- Ensure the final document complies with all “Site Markdown Rules”.

---

## Update `posts-data.js`

Every note must have an entry:

```js
{
  slug: "<slug>",
  title: "<Title>",
  date: "YYYY-MM-DDTHH:MM:SSZ", // ISO 8601 UTC
  category: "<Category>",
  summary: "<1–2 sentence summary, no formulas>"
}
```

Rules:

- `date` is the **current UTC time** in ISO 8601, e.g., `"2025-10-18T12:29:01Z"`.
- If an entry with the same `slug` exists, update its fields (keep array order and surrounding module syntax intact).
- If none exists, append a new object to the exported array.
- Preserve the module wrapper: `export const posts = [ ... ];`
- Preserve existing indentation and commas.

---

## Git Commands (Always Attempt)

After writing the note and updating `posts-data.js`, execute:

```bash
git add notes/<slug>.md posts-data.js
git commit -m "add: <slug>"
git push
```

If there are no changes, skip the commit but still report “no changes”.

---

## Output Summary (Print in Chat)

After all steps, print:

```
✅ Published
Title: <title>
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

```
@title: Custom Title
@slug: custom-slug
@category: Statistics | Machine Learning | LLM | <any>
@summary: One–two sentence description without formulas
```

---

## Zero-Typing Usage

User action in Codex:

```
@file:.codex/rules.md
<<<
# Your Note Title
(paste your raw Markdown here; optionally include @overrides above)
>>>
```

Codex must automatically: **Format → Classify → Write → Update posts-data.js → Commit & Push → Print Summary**.

Notes on YouTube-style video search covering query-video pairs, lexical retrieval, embeddings, indexing pipelines, and fusion/re-ranking.

## 1) What does "10 million Video-Text Query Pairs" mean?

When an ML system says it has **N video-text query pairs** for training, it usually means the training data is organized as **pairs of**:

- a **text query** (what a user typed)
- a **video** (a candidate result)
- plus some **supervision signal** that indicates how relevant that video is to that query

### 1.1 Common forms of supervision

1. **Binary relevance**
   - $y = 1$ if the video is relevant to the query
   - $y = 0$ otherwise

2. **Graded relevance** (more common in ranking)
   - e.g. $y \in \{0,1,2,3\}$ where 3 means "very relevant"

3. **Implicit feedback derived from logs**
   - clicks, watch time, long-press, "not interested", dwell time
   - often converted into a label or a training weight

So a "pair" does **not** necessarily mean _one query maps to one video only_ in the overall dataset. It means each **training record** is (query, video, label).

### 1.2 But in reality, one query corresponds to many videos (and vice versa)

- A single query like "dogs playing indoor" will be associated with **many** videos (clicked / impressed / labeled).
- A single video can be relevant to **many** queries.

The dataset is stored as **many individual pair records**, even if the same query appears many times with different videos.

### 1.3 Pairwise / listwise ranking vs "pair records"

Even if the model is trained with a ranking objective, the data is often still represented as "pairs" or "lists of pairs":

- **Pointwise**: learn $s(q, v)$ directly from labeled pairs
- **Pairwise**: learn that $v\_{pos}$ should rank above $v\_{neg}$ for the same query
  - training samples can look like $(q, v\_{pos}, v\_{neg})$
- **Listwise**: learn from an entire list of candidates per query

Here $q$ is the query, $v$ is the video, and $s$ is the scoring function.

In practice, "10 million pairs" often means **10 million supervised examples**, regardless of whether the final loss is pointwise/pairwise/listwise.

## 2) Text search, "textual data", and inverted index - where does "similarity" come from?

You noticed something important: an **inverted index** is not "ML", yet it still produces a ranked list. The ranking comes from **classical IR (Information Retrieval) scoring**, not from neural embeddings.

### 2.1 What is the "video's textual data"?

In a video platform, "textual data" typically includes:

- Title
- Description
- Tags (manual tags)
- Auto-generated tags (from an auto-tagger model)
- Captions / transcripts (ASR output)
- Comments (sometimes, but often not used directly in search ranking)

Text search uses this textual data as the document content to index.

### 2.2 What is an inverted index?

An inverted index maps:

- **term -> list of documents that contain the term**

Example:

- `dog -> [video_1, video_6, video_42, ...]`
- `indoor -> [video_1, video_9, ...]`

At query time, you retrieve posting lists for each query term and compute a ranking score.

### 2.3 How does inverted index produce "similarity" / ranking?

The "similarity" is usually a **lexical relevance score**, such as:

- **BM25** (very common in Elasticsearch)
- TF-IDF variants
- Boolean matching + field boosting + other heuristics

BM25-like scoring favors:

- documents containing query terms
- query terms that are **rare** across the corpus
- reasonable term frequency saturation (not linear growth)

So even though it's not neural, it's still a scoring-based retrieval system:
**similarity = lexical relevance score**.

### 2.4 Why text search is still useful next to embeddings

- Text search is strong at **exact matches**
  - names, product IDs, specific phrases, spellings
- Embedding search is strong at **semantic matches**
  - paraphrases, synonyms, "same meaning different words"

This is exactly why many modern systems do **both**, then fuse.

## 3) NFKD - what is it, and why is it part of text normalization?

**NFKD** is a Unicode normalization form:

- **Normalization Form KD** = **Compatibility Decomposition**
- It converts characters into a more canonical / comparable representation by:
  - decomposing combined forms into simpler components
  - applying compatibility mappings (e.g., "ﬁ" ligature -> "fi")

### 3.1 Why it matters in search / NLP pipelines

Text can contain visually similar characters that are not byte-wise identical:

- Accents: `Noël` vs `Noel`
- Full-width vs half-width forms (common in some languages): `Ａ` vs `A`
- Ligatures: `ﬁ` vs `fi`
- Combined diacritics vs precomposed characters

NFKD helps normalize these so tokenization and matching behave consistently.

### 3.2 Typical pipeline usage

Often you will see:

1. Unicode normalize (NFKD or NFKC)
2. Case fold / lowercase
3. Strip accents (optional; sometimes done after NFKD)
4. Remove punctuation / trim whitespace (depending on task)

For search, this reduces "surface-form noise" and improves recall.

## 4) Token -> ID: Lookup Table (LUT) vs Hashing - what is the _real_ difference?

You asked: "If collisions happen, doesn't hashing become unusable?"
This is the right skepticism. The answer is: hashing is used when you accept a controlled amount of information loss in exchange for scalability.

### 4.1 Lookup Table (Vocabulary / LUT)

**Idea:** build a vocabulary of all tokens you care about, assign each token a unique integer ID.

- Pros:
  - No collisions: each token maps to a unique ID
  - Can invert: ID -> token is possible (useful for debugging/interpretability)
- Cons:
  - Vocabulary building and storage cost
  - Out-of-vocabulary (OOV) tokens need special handling (`[UNK]`)
  - Updating vocab in production can be painful

### 4.2 Feature hashing (Hashing trick)

**Idea:** map token to an integer bucket via a hash function:

- $id = \mathrm{hash}(token) \bmod M$
  where $M$ is the number of buckets.

- Pros:
  - No explicit vocabulary storage
  - Handles unseen tokens "for free"
  - Extremely scalable for very large / dynamic vocabularies
- Cons:
  - Collisions: different tokens can land in the same bucket
  - ID -> token inversion is not possible
  - Collisions introduce noise in the representation

### 4.3 If collisions exist, why is hashing still useful?

Because collisions do **not necessarily** destroy performance if:

- M is large enough relative to the number of active tokens
- The task is robust to some feature noise
- The model is regularized / trained with that noise

A useful mental model:

> Hashing makes your feature space a "compressed sketch" of the true sparse space.

If M is too small, collisions become frequent and performance degrades.
If M is reasonably large, the collision rate can be low enough to be acceptable.

### 4.4 When hashing is reasonable (and when it's not)

Hashing is reasonable when:

- you have a huge or fast-changing token universe
- you need memory-efficient sparse features
- the task can tolerate mild noise (e.g., some retrieval or classification settings)

Hashing is risky when:

- you need interpretability (ID -> token)
- collisions would confuse critical tokens (e.g., safety keywords)
- M must be small due to resource constraints

In modern deep NLP, hashing is less common for text _tokens_ (because subword tokenizers + vocab are standard), but hashing is still common for:

- high-cardinality categorical features (user IDs, item IDs) in recommendation systems
- extremely large sparse feature spaces

## 5) Text representations: BoW, TF-IDF, Embeddings, Transformers

This section connects the "token to ID" step to what you actually feed into a model.

### 5.1 BoW (Bag of Words)

- Represent a document by a vector of word counts.
- Fast, simple, but:
  - ignores word order
  - sparse high-dimensional vectors
  - poor at semantics

### 5.2 TF-IDF

- Like BoW but reweights words by their importance.
- Still sparse and orderless, but often better than raw counts.

### 5.3 Embedding lookup layer

- Token IDs index into a trainable matrix $E \in R^{V \times d}$.
- Each token gets a dense vector in $R^d$.
- Advantages:
  - dense, low-dimensional
  - learns similarity structure through training
- Limitation:
  - "static" word embeddings: the same token always maps to the same vector

### 5.4 Word2Vec (CBOW / Skip-gram)

- Learns embeddings by predicting word contexts.
- Still mostly static embeddings (token-based), not context-dependent.

### 5.5 Transformer-based models (contextual embeddings)

- Embedding depends on surrounding words.
- Handles polysemy better ("bank" in different contexts).
- More expensive but much stronger for semantic tasks.

## 6) Video embeddings: Video-level vs Frame-level encoders

The goal is to represent each video as a vector in an embedding space for retrieval.

### 6.1 Video-level model (whole video in, one vector out)

- Input: entire video sequence
- Model: 3D CNN, video transformer, etc.
- Pro:
  - can model temporal dynamics directly
- Con:
  - expensive (compute/memory), hard to scale for large search indices

### 6.2 Frame-level model (sample frames, embed each frame, aggregate)

Steps:

1. preprocess + sample frames
2. run an image/frame encoder per frame -> frame embeddings
3. aggregate frame embeddings -> video embedding
   - average, max, attention pooling, temporal model, etc.

Why it dominates large-scale retrieval:

- scalable and cache-friendly
- easy to parallelize
- can reuse strong image encoders

Trade-off:

- temporal information may be simplified unless you use stronger aggregation

## 7) Fusing layer - what it is and why it exists

A "fusing layer" in this context is **not** a deep neural layer by necessity. It is a **system component** that merges multiple ranked candidate lists into one final ranked list.

### 7.1 What are the lists being fused?

Typically at least two:

1. **Text search results**
   - from Elasticsearch / inverted index
   - good at exact lexical matching

2. **Embedding (vector) search results**
   - from nearest neighbor search in embedding space (ANN)
   - good at semantic similarity

Each produces a list of videos with its own scoring scheme.

### 7.2 The simplest fusion: weighted sum scoring

For a video $v$, define:

- $\mathrm{score}\_{text}(v)$ from lexical search
- $\mathrm{score}\_{vec}(v)$ from embedding similarity

Compute:

$$ \mathrm{score}\_{final}(v) = w\_{text} \cdot \mathrm{score}\_{text}(v) + w\_{vec} \cdot \mathrm{score}\_{vec}(v) $$

Then sort by $\mathrm{score}\_{final}(v)$.

Important practical detail:

- Scores often need normalization (z-score, min-max, rank-based) because text and vector scores live on different scales.

### 7.3 A more powerful fusion: re-ranking model

Instead of a fixed weighted sum, train a model that takes multiple signals and outputs a final ranking score.

Features might include:

- text score
- vector score
- freshness / upload time
- popularity
- watch time prediction
- policy features (safety, dedup, etc.)

Pros:

- better quality and adaptability

Cons:

- requires training data and ongoing maintenance
- increases serving latency and complexity

### 7.4 Why fusion is "system glue," not "core ML"

Because large-scale search quality depends on:

- recall sources (how you get candidates)
- ranking signals
- policy/business constraints

Fusion is where you balance:

- precision vs recall
- exact match vs semantic match
- quality vs latency

## 8) Indexing pipelines: why there are two (text vs video)

### 8.1 Video indexing pipeline (vector index)

When a video is uploaded (or periodically reprocessed):

1. preprocess video
2. compute video embedding using video encoder
3. store embedding in a vector index (ANN index)
4. store metadata in a database

At query time:

- encode the query (text embedding)
- nearest neighbor search in vector index -> candidate videos

### 8.2 Text indexing pipeline (inverted index)

When a video is uploaded or metadata changes:

1. gather textual fields (title, tags, transcript)
2. normalize and tokenize
3. update inverted index (Elasticsearch)

At query time:

- tokenize query
- retrieve posting lists and rank via BM25/TF-IDF-like scoring

### 8.3 Auto-tagging (why it matters)

If many creators do not provide tags:

- an auto-tagger model can generate tags from video content
- these tags improve text search recall
- but tags can be noisy -> still useful as weak signals

## 9) Putting it all together: end-to-end flow (high level)

**Offline / indexing**

1. Video upload
2. Video pipeline produces a video embedding -> vector index
3. Text pipeline indexes title/tags/transcript -> inverted index

**Online / serving**

1. User query arrives
2. Text search retrieves lexical candidates
3. Vector search retrieves semantic candidates
4. Fusing layer merges + scores candidates
5. Re-ranking service applies business/policy logic (optional)
6. Final ranked list returned to user

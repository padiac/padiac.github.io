This document summarizes key conceptual points discussed while studying a recommendation system chapter, with an emphasis on modeling assumptions and geometry rather than implementation details.

---

## 1. Content-Based Filtering (CBF)

Content-based filtering constructs recommendations directly from item features.

- Each item has an explicit feature vector (genres, tags, text embeddings, etc.).
- A user profile is typically an aggregation (often an average) of features of consumed items.
- Recommendations are made by finding items close to the user profile in feature space.

Key property:
CBF can naturally handle niche interests because it does not rely on population-level interaction patterns.

---

## 2. Collaborative Filtering (CF)

Collaborative filtering learns preferences purely from user–item interactions.

- Users and items are embedded into a shared latent space.
- Similarity is inferred from interaction patterns, not explicit semantics.

Limitation:
Latent factors are shaped by collective behavior. Rare interests are often underrepresented and pulled toward dominant patterns.

---

## 3. Matrix Factorization

Matrix factorization models approximate the interaction matrix A as:

A ≈ U · Vᵀ

where U and V are learned embeddings.

### Observed vs Unobserved Pairs

- Observed pairs correspond to explicit interactions.
- Unobserved pairs are ambiguous: they may indicate dislike or simply lack of exposure.

A common loss is a weighted combination:

L = Σ_observed (A_ij − U_i·V_j)²  
  + w Σ_unobserved (0 − U_i·V_j)²

The zeros remain zeros; the weight w only controls how strongly unobserved pairs influence training.

---

## 4. Inference and Scoring

At inference time:
- User and item embeddings are fixed.
- Relevance is computed via dot product.
- Scores are used for ranking, not absolute evaluation.

The dot product represents a learned relevance score, not a literal rating.

---

## 5. Two-Tower Neural Networks

Two-tower models consist of:
- A user encoder mapping user features to embeddings.
- An item encoder mapping item features to embeddings.

Encoders are shared across all users and items.

Key distinction from MF:
Two-tower models can generalize to unseen users or items because embeddings are produced by functions, not lookup tables.

---

## 6. Similarity Measures

- Dot product mixes direction and magnitude.
- Cosine similarity compares direction only and requires normalization.
- L2 distance measures absolute proximity in space.

If embeddings are L2-normalized, dot product, cosine similarity, and L2 distance become equivalent up to monotonic transformations.

---

## 7. Offline Metrics

Common offline metrics include:
- Precision@k
- Mean Average Precision (MAP)
- Diversity (measured via average pairwise similarity)

Diversity must be interpreted jointly with relevance.

---

## 8. Summary Insight

Different recommendation models encode different assumptions about:
- Data sparsity
- Geometry of embedding space
- Generalization behavior

Understanding these assumptions is more important than treating models as interchangeable tools.

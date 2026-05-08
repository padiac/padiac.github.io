## Overview

The goal of word embedding is to represent each word as a dense, low-dimensional vector that captures semantic relationships. Three landmark methods define the trajectory from neural prediction to matrix factorization to subword-aware training.

---

## Taxonomy

```
Word Embeddings (goal: obtain word vectors)
│
├── Word2Vec (2013)
│   ├── Architecture: Skip-gram or CBOW
│   └── Training speedup: Hierarchical Softmax or Negative Sampling
│
├── GloVe (2014)
│   └── Co-occurrence matrix factorization
│
└── FastText (2016)
    └── Word2Vec + subword information
```

---

## 1. Word2Vec

Word2Vec learns word vectors by training a shallow neural network on a self-supervised task derived from raw text.

### Skip-gram

Given a center word $w\_c$, predict each context word $w\_o$ within a fixed window. The objective maximizes

$$ \sum\_{t=1}^{T} \sum\_{-m \le j \le m, j \ne 0} \log P(w\_{t+j} \mid w\_t) $$

where $T$ is the corpus length and $m$ is the window size. The conditional probability is modeled with a softmax over the entire vocabulary $V$:

$$ P(w\_o \mid w\_c) = \frac{\exp(\mathbf{u}\_{w\_o}^\top \mathbf{v}\_{w\_c})}{\sum\_{i \in V} \exp(\mathbf{u}\_i^\top \mathbf{v}\_{w\_c})} $$

Each word maintains two vectors: $\mathbf{v}$ (as center) and $\mathbf{u}$ (as context).

### CBOW (Continuous Bag of Words)

Reverses the direction: given all context words in the window, predict the center word. The context representation is the average of the context vectors.

### Training Speedups

Computing the full softmax denominator over $\lvert V \rvert$ is expensive.

- **Hierarchical Softmax.** Replaces the flat softmax with a binary tree (typically a Huffman tree). Each word sits at a leaf; predicting a word requires $O(\log \lvert V \rvert)$ binary classifier decisions along the path from root to leaf.

- **Negative Sampling.** Approximates the softmax by contrasting the true (center, context) pair against $k$ randomly sampled negative pairs. The objective for a single positive pair $(w\_c, w\_o)$ becomes

$$ \log \sigma(\mathbf{u}\_{w\_o}^\top \mathbf{v}\_{w\_c}) + \sum\_{i=1}^{k} \mathbb{E}\_{w\_i \sim P\_n(w)} [\log \sigma(-\mathbf{u}\_{w\_i}^\top \mathbf{v}\_{w\_c})] $$

where $P\_n(w) \propto f(w)^{3/4}$ is the noise distribution based on word frequency $f(w)$.

---

## 2. GloVe (Global Vectors)

GloVe bridges the gap between count-based and prediction-based methods. It constructs a global word-word co-occurrence matrix $X$, where $X\_{ij}$ counts how often word $j$ appears in the context of word $i$.

The loss minimizes the weighted least-squares objective

$$ J = \sum\_{i,j=1}^{\lvert V \rvert} f(X\_{ij}) (\mathbf{w}\_i^\top \tilde{\mathbf{w}}\_j + b\_i + \tilde{b}\_j - \log X\_{ij})^2 $$

where $f$ is a weighting function that caps the influence of very frequent co-occurrences:

$$ f(x) = \begin{cases} (x / x\_{\max})^\alpha, & x < x\_{\max} \\ 1, & \text{otherwise} \end{cases} $$

The key insight is that ratios of co-occurrence probabilities encode semantic relationships more reliably than raw counts.

---

## 3. FastText

FastText extends Skip-gram by representing each word as a bag of character n-grams. For example, the word "where" with $n = 3$ produces the n-grams: `<wh`, `whe`, `her`, `ere`, `re>`, plus the special whole-word token `<where>`.

The word vector is the sum of all its n-gram vectors:

$$ \mathbf{v}\_w = \sum\_{g \in \mathcal{G}(w)} \mathbf{z}\_g $$

This gives FastText two advantages over vanilla Word2Vec:

1. **Morphological awareness.** Words sharing substrings (e.g., "learn", "learning", "learned") share n-gram vectors, capturing morphological regularity.
2. **Out-of-vocabulary handling.** Unseen words can still be represented by summing their constituent n-gram vectors.

---

## Comparison

| Aspect | Word2Vec | GloVe | FastText |
|--------|----------|-------|----------|
| Core idea | Predict context from center (or vice versa) | Factorize co-occurrence matrix | Skip-gram over character n-grams |
| Training signal | Local windows | Global co-occurrence statistics | Local windows + subword info |
| OOV handling | None | None | Sum of n-gram vectors |
| Semantic analogies | Strong | Strong | Strong |
| Morphology | Weak | Weak | Strong |

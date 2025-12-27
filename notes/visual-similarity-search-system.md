This note summarizes a visual similarity search system (Pinterest-style "find similar images"), covering embeddings, training objectives, evaluation metrics, and ANN retrieval.

## 1) Problem framing: visual similarity search

A visual search system takes a query image (or crop) and returns a ranked list of images that look visually or semantically similar.

Typical pipeline:

1. Encode each image into an embedding vector.
2. Compare the query embedding against candidate embeddings using a distance or similarity function.
3. Rank candidates by similarity (most similar first).
4. Use an ANN index to scale retrieval to millions or billions of vectors.
5. Evaluate ranking quality with offline metrics (MRR, Recall@k, Precision@k, mAP, nDCG).

## 2) Why ResNet or ViT for embeddings (vs VGG, MobileNet, AlexNet, LeNet, U-Net)

### 2.1 Core idea: representation learning, not classification output

Even if the architecture was designed for classification, we often use it as an encoder:

- Input image -> deep network -> embedding vector
- Then do nearest neighbor search in embedding space

So yes, it feels like latent representation or compression (as in VAE), but:

- VAE is a generative model with a decoder and probabilistic latent variables.
- ResNet or ViT here is typically a discriminative encoder trained with contrastive or metric learning objectives to produce good embeddings for similarity search.

### 2.2 Why ResNet

ResNet became a default backbone because:

- Strong, stable baseline for images
- Residual connections make deep training practical
- A massive ecosystem: pretrained weights, known behavior, predictable tuning

### 2.3 Why ViT

Vision Transformers (ViT) are popular because:

- They model global context well (useful for semantics)
- With large data or pretraining, they often produce strong embeddings
- Widely available pretrained variants

### 2.4 Are ResNet and ViT always optimal?

No. They are common defaults because they are robust and widely proven, not because they dominate every setting.

Other backbones still matter:

- MobileNet or EfficientNet: when compute or latency is tight (mobile or edge).
- VGG, AlexNet, LeNet: historically important, but not SOTA for modern embedding quality.
- U-Net: primarily for segmentation or denoising, not a standard choice for retrieval embeddings (unless the task is special).

## 3) Training for similarity: contrastive learning + softmax + cross-entropy

### 3.1 What the model outputs during training

The model maps each image to an embedding:

- Query embedding: $q$
- Candidate embeddings: $k_{1}, k_{2}, ..., k_{n}$

Then we compute similarity scores (dot product or cosine similarity):

- $s_{i} = \mathrm{sim}(q, k_{i})$

Common similarity or distance choices:

$$
\mathrm{cos}(q, k) = \frac{q \cdot k}{\Vert q \Vert \Vert k \Vert}
$$

$$
d(q, k) = \Vert q - k \Vert_{2}
$$

### 3.2 Why softmax then cross-entropy appears

This is a common form of contrastive or InfoNCE loss:

- Softmax converts similarity scores into a probability distribution over candidates.
- Cross-entropy trains the model to assign high probability to the true positive (and low to negatives).

$$
p_{i} = \frac{\exp(s_{i} / \tau)}{\sum_{j=1}^{n} \exp(s_{j} / \tau)}
$$

$$
L = -\log \frac{\exp(s_{+} / \tau)}{\sum_{j=1}^{n} \exp(s_{j} / \tau)}
$$

Here $\tau$ is a temperature scalar.

Important:
The softmax probabilities are training-time normalization over the candidate set for that training example.
They are not the same as the evaluation ground truth similarity score scale.

### 3.3 Key clarification: "softmax gives normalized values" does not mean scores become 0-5

Softmax outputs probabilities in [0, 1] summing to 1 within that batch or candidate set.

That does not mean:

- The system has a 0-5 similarity score
- Or the similarity function outputs are calibrated to 0-5

Softmax is just a convenient way to formulate a classification-style objective over "which candidate is the positive."

## 4) The 0-5 ground truth similarity score vs model similarities

You may see a mismatch:

- The model computes similarities (dot, cosine, L2), then uses softmax + cross-entropy during training.
- Offline evaluation may show ground truth similarity labels on a 0-5 scale.

How it is reconciled:

1. Model output is a ranking signal, not necessarily a human-readable score.
2. Offline evaluation often uses graded relevance labels (0-5) to judge ranking quality (nDCG is built for that).
3. If you truly need a calibrated score in 0-5, add a separate calibration or regression layer (not required for ranking).

So: ranking is what matters; absolute score scale usually does not.

## 5) Ranking metrics: what each metric measures

In ranking, we evaluate lists like:

- One query image -> one ranked list of candidates

Ground truth provides either:

- Binary relevance: relevant or not relevant (for Precision, Recall, mAP)
- Graded relevance: 0-5 (for DCG and nDCG)

## 6) MRR (Mean Reciprocal Rank): what the "M" means, and why models can tie

### 6.1 Definition (plain language)

For each query:

1. Find the rank position of the first relevant item.
2. Compute reciprocal rank: RR = 1 / rank.
3. Average over queries: MRR = mean(RR).

$$
\mathrm{MRR} = \frac{1}{M} \sum_{q=1}^{M} \frac{1}{r_{q}}
$$

### 6.2 What is "M" (or "m") in the formula?

It is the number of queries or ranked lists:

- In visual search, that is typically the number of query images in the evaluation set.

### 6.3 Why two models can have the same MRR

MRR only cares about where the first relevant result appears.
It ignores whether the rest of the list is good or terrible.

So two models can tie in MRR even if their overall ranking differs.

## 7) Recall@k vs Precision@k

They both count "how many relevant items are in the top k," but their denominators differ:

- Precision@k: among the top k returned, what fraction is relevant? Denominator is k.
- Recall@k: among all relevant items, what fraction was retrieved in top k? Denominator is the number of all relevant items.

$$
\frac{1}{k} 
$$

$$
\mathrm{Precision}_{k} = \frac{1}{k} \sum_{i=1}^{k} r_{i}
$$

$$
\mathrm{Recall}_{k} = \frac{1}{R} \sum_{i=1}^{k} r_{i}
$$

Here $r_{i} \in \{0, 1\}$ and $R$ is the total number of relevant items.

Memory hook:

- Precision fears junk (irrelevant items included).
- Recall fears missing (relevant items omitted).

## 8) AP and mAP: average precision is not averaging over k

### 8.1 AP (Average Precision) for a single query

AP is computed by:

1. Scan the ranked list.
2. Every time you encounter a relevant item at rank r, compute Precision@r.
3. Average those precision values over all relevant items.

$$
P_{i} = \frac{1}{i} \sum_{j=1}^{i} r_{j}
$$

$$
\mathrm{AP} = \frac{1}{R} \sum_{i=1}^{N} P_{i} \ast r_{i}
$$

Here $N$ is the list length.

In words:
"Whenever the system hits a relevant result, ask: how precise has it been up to now?"

### 8.2 mAP (mean AP) across many queries

mAP is simply the average AP over all queries.

### 8.3 Limitation of mAP

mAP aligns with binary relevance (relevant or not).
When relevance is graded (0-5), nDCG is usually more appropriate.

## 9) DCG and nDCG: the formula and what each term does

### 9.1 DCG (Discounted Cumulative Gain)

Common formula:

$$
DCG_{p} = \sum_{i=1}^{p} \frac{{rel}_{i}}{\log_{2}(i+1)}
$$

- $rel_{i}$: ground truth relevance (often 0-5) of the item at rank $i$
- $\log_{2}(i+1)$: discount factor, lower ranks are worth less

Interpretation:

- High relevance at top ranks contributes a lot.
- High relevance at deep ranks contributes less due to discount.

### 9.2 Why DCG needs normalization

Raw DCG depends on how many relevant items exist and their label magnitudes.
Different queries can have different maximum possible DCG.

### 9.3 nDCG (normalized DCG)

$$
{nDCG}_{p} = \frac{{DCG}_{p}}{{IDCG}_{p}}
$$

- IDCG is the DCG of the ideal ranking (items sorted by true relevance).
- nDCG is then in [0, 1] (typically), making it comparable across queries.

Key interpretation:
nDCG measures how close your ranking is to the ideal ranking, while respecting graded relevance and position bias.

## 10) ANN (Approximate Nearest Neighbor): scaling similarity search

### 10.1 Why ANN is needed

Exact nearest neighbor search over huge databases is too slow:

- For N vectors, naive search is O(N) per query.
- At million or billion scale, that is not workable.

ANN methods aim to:

- Search only a small candidate subset.
- Return "good enough" neighbors quickly.

### 10.2 Categories

#### A) Tree-based ANN

Idea: partition space recursively (tree) and search only relevant regions.

Examples:

- KD-trees, R-trees
- Annoy (Spotify)

Works best for lower dimension, moderate data sizes.
Struggles with high-dimensional embeddings (curse of dimensionality).

#### B) LSH (Locality-Sensitive Hashing)

Idea: hash so that similar points collide with higher probability, then search within same bucket(s).

Pros: conceptually simple and can be fast.
Cons: can require heavy parameter tuning; quality may lag modern graph or cluster approaches.

#### C) Clustering-based ANN

Idea: cluster vectors and search only nearby clusters (a common industrial strategy).

Often implemented in systems like inverted file indexes (IVF-style) and combined with compression for memory efficiency.

### 10.3 Industrial libraries

- Faiss (Meta)
- ScaNN (Google)

## 11) Practical mental model tying everything together

1. Backbone encoder (ResNet, ViT, MobileNet, and so on) produces embeddings.
2. Training (contrastive learning) makes embeddings meaningful for similarity.
3. Serving uses ANN indexing to retrieve candidates fast.
4. Ranking metrics evaluate whether the top results match human judgments:
   - Binary metrics: Precision, Recall, mAP
   - Graded metrics: nDCG

## 12) Common pitfalls

- Softmax normalization during training does not mean output scores match a 0-5 human label scale.
- MRR cannot distinguish models that differ after the first relevant item (it only "sees" the first hit).
- Precision@k vs Recall@k look similar but differ by denominator: quality of shown items vs coverage of all relevant items.
- AP is not averaging over k; it averages Precision at ranks where relevant items occur.
- DCG is not comparable across queries until normalized into nDCG.

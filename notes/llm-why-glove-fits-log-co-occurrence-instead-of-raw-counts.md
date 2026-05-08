When learning GloVe, the loss function looks like this:

$$ \text{minimize} \sum\_{i,j} f(X\_{ij}) \bigl(\theta\_i^T e\_j - \log X\_{ij}\bigr)^2 $$

where $X\_{ij}$ is the co-occurrence count of word $i$ and word $j$. The first time I saw this formula, my reaction was: why $\log X\_{ij}$ instead of $X\_{ij}$? Is it just to make the numbers smaller?

No. The log is a mathematical necessity, not decoration.

## The Surface Reason: Numerical Range

Fitting $X\_{ij}$ directly is obviously problematic. "the" and "of" co-occur a million times, while "orange" and "king" co-occur twice. Asking a 300-dimensional dot product to equal one million is unrealistic. Moreover, high-frequency word pairs would completely dominate the loss, forcing the model to spend all its capacity fitting semantically vacuous combinations like the/of/a.

After taking the log, $\log 1{,}000{,}000 \approx 14$ and $\log 2 \approx 0.7$, compressing all targets into a range that dot products can reasonably reach. This explanation is correct but superficial.

## The Deeper Reason: Language Is Multiplicative, Vector Spaces Are Additive

The real reason is hidden in one observation: **the associations between words are fundamentally multiplicative**.

The probability chain rule tells us:

$$ P(\text{orange}, \text{juice}) = P(\text{orange}) \cdot P(\text{juice} \mid \text{orange}) $$

But vector spaces excel at addition. Properties like "king - man + woman = queen" can only hold if semantic relationships are expressible through vector addition and subtraction.

**Log is the bridge that translates multiplication into addition**:

$$ \log P(\text{orange}, \text{juice}) = \log P(\text{orange}) + \log P(\text{juice} \mid \text{orange}) $$

In log space, the multiplicative structure of probabilities becomes the additive structure of vectors. This is not a coincidence; it is the fundamental reason why word analogies work.

## Information Theory Gives the Same Answer

In information theory, the standard measure of association strength between word pairs is **PMI (Pointwise Mutual Information)**:

$$ \text{PMI}(i, j) = \log \frac{P(i, j)}{P(i) \cdot P(j)} $$

PMI inherently contains a log. This is not coincidental: the natural tool for measuring "surprise" and "association" in information theory is the logarithm, because information content $= -\log P$.

Levy and Goldberg proved an elegant result in 2014: **the optimal solution of Word2Vec's Skip-gram with Negative Sampling is equivalent to implicitly factorizing a PMI matrix**. In other words, Word2Vec and GloVe use different methods on the surface but fit the same thing mathematically: **a log-form association measure**.

## A Broader Pattern

Looking back, nearly every objective in deep learning that involves probabilities or counts takes a log:

- Cross-entropy loss: $-\log P$
- Log-likelihood: $\log P$
- PMI / GloVe: $\log P(i,j) / (P(i)P(j))$
- KL divergence: $\sum P \log(P/Q)$

Why? Because **probabilities live in a multiplicative world, while optimizers and vector spaces live in an additive world**. Log is the only bridge between these two worlds.

## One-Line Summary

GloVe fits log co-occurrence not to make the numbers look nicer, but because lexical associations are inherently multiplicative (probabilities multiply) while vector spaces express relationships through addition. Log converts multiplication into addition, making magical linear properties like "king - man + woman = queen" possible.

Next time you see a log appear in a loss function, think one level deeper: **which multiplicative world is it translating into which additive world?**

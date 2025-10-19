This note summarizes variance identities from Sections 2.1–2.3 and documents standard discrete examples (Poisson and Binomial) where the formulas take particularly simple forms.

---

## 1. Definition of Variance

For a random variable $X$ with mean $\mathbb{E}[X]$, the variance measures the average squared deviation from the mean:

$$
\operatorname{Var}(X) = \mathbb{E}\left[(X - \mathbb{E}[X])^2\right] = \mathbb{E}[X^2] - \mathbb{E}[X]^2.
$$

The computational form follows from expanding $(X - \mathbb{E}[X])^2$ and applying linearity of expectation.

---

## 2. Properties of Variance (Theorem 2.1)

Let $C$ be a constant and $X$ a random variable. Then:

1. $\operatorname{Var}(X + C) = \operatorname{Var}(X)$.
2. $\operatorname{Var}(CX) = C^2 \operatorname{Var}(X)$.

**Sketch.** Adding $C$ shifts realizations uniformly, so deviations from the mean are unchanged. Scaling by $C$ multiplies every deviation by $C$, so squared deviations scale by $C^2$.

---

## 3. Variance of Independent Sums (Theorem 2.2)

If $X_1, \dots, X_n$ are independent, then

$$
\operatorname{Var}\left(\sum_{i=1}^{n} X_i\right) = \sum_{i=1}^{n} \operatorname{Var}(X_i).
$$

Independence implies $\mathbb{E}[X_i X_j] = \mathbb{E}[X_i]\,\mathbb{E}[X_j]$ for $i \neq j$, so all cross-terms disappear when expanding the variance of the sum.

---

## 4. Example 2.1 – Standardization

If $\mathbb{E}[X] = a$ and $\operatorname{Var}(X) = \sigma^2$, define

$$
Y = \frac{X - a}{\sigma}.
$$

Then $\mathbb{E}[Y] = 0$ and $\operatorname{Var}(Y) = 1$. Standardizing isolates pure shape information by putting the variable on a zero-mean, unit-variance scale.

---

## 5. Example 2.2 – Poisson Distribution

Let $X \sim \operatorname{Pois}(\lambda)$. The mean and variance coincide:

$$
\mathbb{E}[X] = \lambda, \qquad \operatorname{Var}(X) = \lambda.
$$

**Sketch.** Decompose $i^2 = i(i-1) + i$ inside the second moment

$$
\mathbb{E}[X^2] = \sum_{i=0}^{\infty} i^2 \frac{e^{-\lambda} \lambda^i}{i!} = \lambda^2 + \lambda,
$$

and subtract $\mathbb{E}[X]^2$.

---

## 6. Example 2.3 – Binomial Distribution

Let $X \sim \operatorname{Bin}(n, p)$ and write $X = \sum_{i=1}^{n} X_i$ with i.i.d. Bernoulli variables $X_i \sim \operatorname{Bern}(p)$. Then

$$
\mathbb{E}[X] = np, \qquad \operatorname{Var}(X) = np(1-p).
$$

The variance follows from Theorem 2.2 because $\operatorname{Var}(X_i) = p(1-p)$ and the Bernoulli trials are independent.

---

## Summary Table

| Distribution | Mean $\mathbb{E}[X]$ | Variance $\operatorname{Var}(X)$ |
|:-------------|:---------------------|:--------------------------------|
| General | $\mathbb{E}[X]$ | $\mathbb{E}[X^2] - \mathbb{E}[X]^2$ |
| $\operatorname{Pois}(\lambda)$ | $\lambda$ | $\lambda$ |
| $\operatorname{Bin}(n,p)$ | $np$ | $np(1-p)$ |

---

## Conceptual Summary

- Variance is translation-invariant and scales quadratically with deterministic multipliers.
- Independence makes variance additive; cross-products vanish because mixed expectations factor.
- Standardizing to zero mean and unit variance removes units, easing comparisons across variables.
- Poisson and Binomial families illustrate how discrete models inherit mean–variance structure directly from their parameters.

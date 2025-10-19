This note summarizes the expectation and variance properties of three foundational distributions derived from the normal family: chi-square, Student's t, and F.

---

## 1. Chi-Square Distribution

Let

$$
X_i \sim \mathcal{N}(0,1), \qquad i = 1,\dots,n, \qquad \chi_n^2 = \sum_{i=1}^{n} X_i^2.
$$

Key moments:

$$
\mathbb{E}[X_i^2] = 1, \qquad \mathbb{E}[\chi_n^2] = n, \qquad \operatorname{Var}(\chi_n^2) = 2n.
$$

> The mean of a chi-square variable equals its degrees of freedom.

---

## 2. Student's t Distribution

Define

$$
T = \frac{Z}{\sqrt{V / n}}
$$

with

$$
Z \sim \mathcal{N}(0,1), \qquad V \sim \chi_n^2, \qquad Z \perp V.
$$

- When $n = 1$, the expectation does not exist (the integral diverges).
- When $n > 1$, symmetry yields $\mathbb{E}[T] = 0$.
- The variance exists only if $n > 2$ and equals $\operatorname{Var}(T) = \dfrac{n}{n-2}$.

---

## 3. F Distribution

Let

$$
F = \frac{(X_1^2 / m)}{(X_2^2 / n)},
$$

where $X_1^2 \sim \chi_m^2$ and $X_2^2 \sim \chi_n^2$ are independent. Using

$$
\mathbb{E}[X_1^2] = m, \qquad \mathbb{E}\!\left[\frac{1}{X_2^2}\right] = \frac{1}{n-2},
$$

(valid for $n > 2$), we get

$$
\mathbb{E}[F] = \frac{n}{n-2}.
$$

---

## Summary Table

| Distribution | Definition | Expected value | Existence condition |
|:-------------|:-----------|:---------------|:--------------------|
| **$\chi_n^2$** | $\sum X_i^2$ | $n$ | Always |
| **$t_n$** | $Z / \sqrt{V / n}$ | $0$ | $n > 1$ |
| **$F_{m,n}$** | $(X_1^2 / m) / (X_2^2 / n)$ | $n / (n - 2)$ | $n > 2$ |

---

**Takeaway:** chi-square means track degrees of freedom, the t distribution stays centered at zero once $n$ exceeds one, and the F distribution's mean exceeds one because the denominator is itself random.

## Overview

In statistical inference, a **Minimum Variance Unbiased Estimator (MVUE)** is an unbiased estimator that has the smallest variance among all unbiased estimators for a parameter. A related concept is the **Cramer-Rao Lower Bound (CRLB)**, which provides a theoretical lower limit on the variance of any unbiased estimator.

If an unbiased estimator achieves this lower bound, it is said to be **efficient**.

---

## 1. Cramer-Rao Lower Bound (CRLB)

Let $X_1, X_2, \ldots, X_n$ be independent samples with probability density function $f(x; \theta)$. Suppose we wish to estimate a function $g(\theta)$. Then for any unbiased estimator $\hat{g}(X)$ of $g(\theta)$, the following inequality holds:

$$
\mathrm{Var}_\theta(\hat{g}) \ge \frac{[g'(\theta)]^2}{n I(\theta)}
$$

where $I(\theta)$ is the **Fisher Information**, defined by

$$
I(\theta) = \mathbb{E}_\theta \left[ \left(\frac{\partial}{\partial \theta} \log f(X; \theta) \right)^2 \right].
$$

---

## 2. Examples of MVUE and CRLB

### (a) Normal Distribution - Mean Known or Unknown

For $X_i \sim \mathcal{N}(\mu, \sigma^2)$:

- If $\sigma^2$ is known, the sample mean $\bar{X}$ is unbiased for $\mu$ and achieves the CRLB with $\mathrm{Var}(\bar{X}) = \sigma^2 / n$, so $\bar{X}$ is efficient.
- If both $\mu$ and $\sigma^2$ are unknown, the sample mean remains unbiased, but the CRLB cannot be directly achieved because of the multi-parameter dependency.

### (b) Normal Distribution - Variance Estimation

For the same setup, the sample variance is

$$
S^2 = \frac{1}{n-1} \sum (X_i - \bar{X})^2.
$$

- $S^2$ is unbiased for $\sigma^2$, but it does not achieve the CRLB (which corresponds to division by $n$, not $n-1$). Therefore, $S^2$ is the MVUE but not efficient.

### (c) Exponential Distribution

For $X_i \sim \mathrm{Exp}(\lambda)$ with density $f(x; \lambda) = \lambda e^{-\lambda x}$:

- The MLE is $\hat{\lambda}_{\mathrm{mle}} = 1 / \bar{X}$.
- The Fisher information is $I(\lambda) = 1 / \lambda^2$.
- The CRLB for estimating $\lambda$ is $\lambda^2 / n$.

However, $\hat{\lambda}_{\mathrm{mle}}$ is biased. An unbiased estimator is $\hat{\lambda}_{\mathrm{unbiased}} = (n-1)/(n \bar{X})$, which has variance slightly above the CRLB and is therefore not efficient.

### (d) Poisson Distribution

For $X_i \sim \mathrm{Pois}(\lambda)$:

- $\bar{X}$ is unbiased for $\lambda$.
- The Fisher information is $I(\lambda) = 1 / \lambda$.
- The CRLB is $\lambda / n$.

Since $\mathrm{Var}(\bar{X}) = \lambda / n$, the sample mean $\bar{X}$ achieves the CRLB and is an efficient MVUE.

### (e) Binomial Distribution

For $X_i \sim \mathrm{Bin}(m, p)$, the unbiased estimator of $p$ is $\hat{p} = \bar{X} / m$.

- The Fisher information is $I(p) = m / [p(1-p)]$.
- The CRLB is $p(1-p)/(m n)$.

The variance of $\hat{p}$ equals $p(1-p)/(m n)$, so $\hat{p}$ achieves the CRLB and is efficient.

---

## 3. When the CRLB Is Not Achievable

The CRLB assumes:

1. Regularity conditions hold (differentiation under the integral sign is valid).
2. The parameter space is one-dimensional and continuous.
3. The estimator is unbiased.

If these fail, such as in non-regular cases (uniform distribution, discrete parameters, or boundary points), the CRLB may not be tight or even defined.

---

## 4. Discussion and Takeaways

- The CRLB is a lower bound only under certain assumptions; the actual minimum variance might be higher.
- An MVUE can exist even when the CRLB is not achievable.
- Efficiency requires unbiasedness and variance equal to the CRLB.
- Common examples achieving the CRLB include the normal mean, Poisson mean, and binomial proportion.
- Common examples not achieving the CRLB include the normal variance, exponential rate, and uniform maximum.

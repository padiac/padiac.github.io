## 1. From Variance Estimation to Bayesian Thinking

In classical statistics, the sample variance is often defined as

$$
S^2 = \frac{1}{n - 1} \sum_{i=1}^n (X_i - \bar{X})^2,
$$

where the division by $n - 1$ ensures unbiasedness when estimating the population variance $\sigma^2$. In a Bayesian framework, the focus shifts from unbiased point estimators to posterior inference informed by both the likelihood and prior beliefs.

---

## 2. Empirical Bayes and the Moment Method

### 2.1 The Idea

Empirical Bayes methods use the data to estimate hyperparameters of the prior distribution. Suppose the prior is $\pi(\theta \mid \lambda)$, where $\lambda$ represents hyperparameters. We estimate $\lambda$ with:

- The method of moments, matching theoretical expectations to sample moments.
- Type-II maximum likelihood (ML-II), maximizing the marginal likelihood with respect to $\lambda$.

After obtaining $\hat{\lambda}$ we substitute it into the prior to produce a data-informed prior.

### 2.2 The Circularity Concern

A common concern is the apparent circularity of using the data twice. The workflow is sequential rather than circular:

1. Choose the prior family before observing the data.
2. Tune only the hyperparameters using the data as a pragmatic compromise between subjective priors and frequentist estimation.
3. Combine the empirical prior with the likelihood to obtain the posterior.

When viewed as an outer optimization over hyperparameters, the procedure remains coherent.

---

## 3. Location and Scale Parameters

### 3.1 The Role of Invariance

A location parameter, such as $\mu$ in $X \sim f(x - \mu)$, respects translation invariance: shifting all observations by a constant leaves inference about $\mu$ unchanged. This invariance motivates a uniform prior, $\pi(\mu) \propto 1$.

A scale parameter, such as $\sigma$ in $X \sim f(x / \sigma)/\sigma$, respects multiplicative scaling. The prior should be uniform in $\ln \sigma$, which yields $\pi(\sigma) \propto 1/\sigma$. Although not flat in $\sigma$, it treats each order of magnitude equally.

---

## 4. The Jeffreys Prior: The Unified Rule

Jeffreys (1961) generalized invariance ideas into a single principle. Given a likelihood $f(x \mid \theta)$, define the Fisher information matrix

$$
I(\theta) = \mathbb{E}_\theta \left[ \left( \frac{\partial \ln f(X \mid \theta)}{\partial \theta} \right) \left( \frac{\partial \ln f(X \mid \theta)}{\partial \theta} \right)^{\top} \right].
$$

The Jeffreys prior is $\pi(\theta) \propto \sqrt{\det I(\theta)}$.

### 4.1 Motivation

This prior is invariant under reparameterization. If $\phi = g(\theta)$, then $\pi_\phi(\phi)  d\phi = \pi_\theta(\theta)  d\theta$. No parameterization encodes more or less prior information than another.

---

## 5. Examples

### 5.1 Normal Distribution $\mathcal{N}(\mu, \sigma^2)$

#### Case 1: Both $\mu$ and $\sigma$ unknown

The log-likelihood is

$$
\ell(\mu, \sigma) = -\frac{n}{2} \ln (2\pi) - n \ln \sigma - \frac{1}{2\sigma^2} \sum_{i=1}^n (x_i - \mu)^2.
$$

The Fisher information matrix is

$$
I(\mu, \sigma) =
\begin{pmatrix}
\frac{n}{\sigma^2} & 0 \\
0 & \frac{2n}{\sigma^2}
\end{pmatrix}.
$$

Its determinant is $\det I(\mu, \sigma) = 2n^2 / \sigma^4$, which yields $\pi(\mu, \sigma) \propto 1/\sigma^2$. If we factorize the prior, we recover $\pi(\mu) \propto 1$ and $\pi(\sigma) \propto 1/\sigma$, mirroring translation and scale invariance.

### 5.2 Bernoulli Distribution $\operatorname{Bin}(n, \theta)$

The log-likelihood is

$$
\ell(\theta) = x \ln \theta + (n - x) \ln (1 - \theta).
$$

The Fisher information is $I(\theta) = n / [\theta (1 - \theta)]$, so the Jeffreys prior is $\pi(\theta) \propto [\theta (1 - \theta)]^{-1/2}$. This corresponds to a $\operatorname{Beta}(1/2, 1/2)$ prior that is symmetric around $\theta = 0.5$ and avoids bias toward the boundaries.

---

## 6. Summary Table

| Parameter Type | Example Distribution | Fisher Information $I(\theta)$ | Jeffreys Prior $\pi(\theta)$ | Interpretation |
| --- | --- | --- | --- | --- |
| Location ($\mu$) | $\mathcal{N}(\mu, \sigma^2)$ | constant | $\pi(\mu) \propto 1$ | Translation invariance |
| Scale ($\sigma$) | $\mathcal{N}(\mu, \sigma^2)$ | $1/\sigma^2$ | $\pi(\sigma) \propto 1/\sigma$ | Scale invariance |
| Both ($\mu, \sigma$) | $\mathcal{N}(\mu, \sigma^2)$ | $\det I \propto 1/\sigma^4$ | $\pi(\mu, \sigma) \propto 1/\sigma^2$ | Combined invariance |
| Bernoulli ($\theta$) | $\operatorname{Bin}(n, \theta)$ | $n/[\theta (1 - \theta)]$ | $\pi(\theta) \propto [\theta (1 - \theta)]^{-1/2}$ | Symmetric Beta$(1/2, 1/2)$ prior |

---

## 7. Conceptual Unification

Jeffreys priors unify noninformative priors under the idea that priors should not depend on parameterization. Location problems yield flat priors, scale problems yield $1/\sigma$, and general models use $\sqrt{\det I(\theta)}$. The construction formalizes earlier heuristics about equal ignorance.

---

## 8. Closing Thoughts

Empirical Bayes offers a data-driven bridge between Bayesian and frequentist reasoning. Moment methods and ML-II provide practical hyperparameter estimates, while Jeffreys priors ensure that ostensibly noninformative priors remain coherent under reparameterization. Modern Bayesian software often defaults to Jeffreys or weakly informative priors for precisely this balance of objectivity and practicality.

## 1. Definition

Let $X = (X_1, \dots, X_n)$ be a random sample from a family $\{ f(x;\theta) \mid \theta \in \Theta \}$.

A statistic $T(X)$ is the MVUE of a target $g(\theta)$ if both of the following hold:

1. **Unbiasedness**  
   $\mathbb{E}_\theta[T(X)] = g(\theta)$ for every $\theta \in \Theta$.
2. **Minimum variance**  
   Among all unbiased estimators of $g(\theta)$, $T(X)$ has the smallest variance:

   $$
   \operatorname{Var}_{\theta}[T(X)] \le \operatorname{Var}_{\theta}[\tilde{T}(X)]
   $$

   for any other unbiased estimator $\tilde{T}(X)$.

## 2. Existence and Uniqueness

If an MVUE exists, it is unique. Suppose $T_1$ and $T_2$ are both MVUEs for $g(\theta)$. Then

$$
\mathbb{E}_\theta[T_1 - T_2] = 0
$$

and

$$
\operatorname{Var}_\theta[T_1 - T_2] = 0,
$$

which implies $T_1 = T_2$ almost surely.

## 3. Finding the MVUE (Lehmann-Scheffe)

The Lehmann-Scheffe theorem provides a constructive recipe:

1. **Find a sufficient statistic** $S(X)$ using, for example, the factorization theorem.
2. **Check completeness** of $S(X)$. If $S(X)$ is complete and sufficient, then for any unbiased estimator
   $U(X)$ of $g(\theta)$, the conditional expectation
   $$
   T(X) = \mathbb{E}[U(X)\mid S(X)]
   $$
   is the unique MVUE of $g(\theta)$.

## 4. Examples

### Example 1: Mean of a Normal Distribution

Let $X_1, \dots, X_n \sim \mathcal{N}(\mu, \sigma^2)$ with $\sigma^2$ known.
The sample mean $\bar{X} = \frac{1}{n} \sum_i X_i$ is unbiased for $\mu$ and satisfies
$\operatorname{Var}(\bar{X}) = \sigma^2 / n$. No other unbiased estimator has smaller variance,
so $\bar{X}$ is the MVUE of $\mu$.

### Example 2: Variance of a Normal Distribution

Let $X_1, \dots, X_n \sim \mathcal{N}(\mu, \sigma^2)$.

- If $\mu$ is known, then $S^2 = \frac{1}{n}\sum_i (X_i - \mu)^2$ is unbiased for $\sigma^2$ and is the MVUE.
- If $\mu$ is unknown, the adjusted estimator
  $\hat{\sigma}^2 = \frac{1}{n-1}\sum_i (X_i - \bar{X})^2$ remains unbiased and is still the MVUE of $\sigma^2$.

## 5. Connection to the Cramer-Rao Bound

For an unbiased estimator $T(X)$ of $g(\theta)$,
$$
\operatorname{Var}_\theta[T(X)] \ge \frac{(g'(\theta))^2}{I(\theta)},
$$
where $I(\theta)$ is the Fisher information. If equality holds, $T(X)$ is **efficient**,
meaning it achieves the Cramer-Rao lower bound and is therefore the MVUE.

## 6. Summary

| Attribute | Unbiased | Minimum variance | Unique | Constructive tool | Relation to CRLB |
|-----------|----------|------------------|--------|-------------------|------------------|
| Meaning | $\mathbb{E}[T] = g(\theta)$ | No other unbiased estimator has smaller variance | At most one MVUE exists for a given $g(\theta)$ | Lehmann-Scheffe via complete sufficient statistics | Efficient estimators hit the Cramer-Rao bound |

## 7. References

- E. L. Lehmann and G. Casella, *Theory of Point Estimation*. Springer, 1998.
- G. Casella and R. L. Berger, *Statistical Inference*. Duxbury Press, 2002.
- C. R. Rao, "Information and Accuracy Attainable in the Estimation of Statistical Parameters," *Bulletin of the Calcutta Mathematical Society*, 1945.

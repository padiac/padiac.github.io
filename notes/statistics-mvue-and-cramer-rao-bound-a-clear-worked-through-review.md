## What this note covers (checklist)

- **Full CRLB derivation** (with the score function and the differentiation step you asked about)
- **Uniform(0, $\theta$)**: two unbiased estimators, $2 \bar X$ and $\frac{n+1}{n} X_{(n)}$, and which one wins
- Clear **definition of MVUE** and how to prove it via **Lehmann-Scheffe**
- Statement and use of the **Cramer-Rao Lower Bound**
- **What is $\theta$?** Parameter choices (for example, exponential rate $\lambda$ versus variance $1/\lambda^2$) and why results agree
- **Normal model**: $\sigma^2$ known estimate $\mu$, $\mu$ known estimate $\sigma^2$, and $\mu, \sigma^2$ both unknown why the CRLB is not attained for $\sigma^2$ even though $\bar X$ and the unbiased sample variance stay optimal within their classes

---

## 1) Cramer-Rao Lower Bound: derivation step-by-step

Let $X_1, \ldots, X_n$ have joint log-likelihood $\ell(\theta) = \sum_{i=1}^n \log f(X_i; \theta)$. Define the **score** $S(\theta) = \partial \ell(\theta) / \partial \theta$. Under the usual regularity conditions (swap differentiation and integration, tail behavior, and so on),

$$
\mathrm{E}_{\theta}[S(\theta)] = 0, \qquad \mathrm{Var}_{\\theta}(S(\theta)) = n I(\theta),
$$

where $I(\theta)$ is the **per-sample Fisher information**.

Suppose $\hat g = \hat g(X_1, \ldots, X_n)$ is **unbiased** for $g(\theta)$, so $\mathrm{E}_{\theta}[\hat g] = g(\theta)$. Differentiate both sides with respect to $\theta$ (this is the spot that was subtle earlier):

$$
\frac{d}{d\theta}\mathrm{E}_{\theta}[\hat g]
= \frac{d}{d\theta}\int \hat g(x) f(x; \theta) dx
= \int \hat g(x) \frac{\partial}{\partial \theta} f(x; \theta) dx
= \mathrm{E}_{\\theta}\left[\hat g \frac{\partial}{\partial \theta} \log f(X; \theta)\right].
$$

Because the left-hand side is $g'(\theta)$ and the right-hand side is $\mathrm{Cov}_{\\theta}(\hat g, S) + \mathrm{E}_{\\theta}[\hat g] \mathrm{E}_{\\theta}[S]$ with $\mathrm{E}_{\\theta}[S] = 0$, we obtain

$$
g'(\theta) = \mathrm{Cov}_{\theta}(\hat g, S).
$$

Apply Cauchy-Schwarz to the covariance:

$$
\big(\mathrm{Cov}(A, B)\big)^2 \le \mathrm{Var}(A) \mathrm{Var}(B),
$$

and substitute $A = \hat g$, $B = S$ to get

$$
\big(g'(\theta)\big)^2 \le \mathrm{Var}_{\\theta}(\hat g) \mathrm{Var}_{\\theta}(S) = \mathrm{Var}_{\\theta}(\hat g) n I(\theta).
$$

Therefore

$$
\mathrm{Var}_{\\theta}(\hat g) \ge \frac{\big(g'(\theta)\big)^2}{n I(\theta)}
$$

is the **Cramer-Rao Lower Bound**. Equality holds if and only if $\hat g - a(\theta)$ is a linear function of the score, where $a(\theta)$ is a scalar depending on $\theta$.

---

## 2) MVUE: definition and how to prove it

- **MVUE**: among all unbiased estimators of $g(\theta)$, the Minimum Variance Unbiased Estimator has the smallest variance for every $\theta$.
- **Lehmann-Scheffe**: if $T(X)$ is complete and sufficient for $\theta$ and $\tilde g(X)$ is any unbiased estimator of $g(\theta)$, then $\mathrm{E}[\tilde g \mid T]$ is the unique MVUE.

---

## 3) Uniform$(0, \theta)$: two unbiased estimators and the winner

Let $X_1, \ldots, X_n$ be i.i.d. $\mathrm{Unif}(0, \theta)$.

1. **Method of moments**: $\hat \theta_1 = 2 \bar X$. Since $\mathrm{E}[\bar X] = \theta / 2$, $\hat \theta_1$ is unbiased. Its variance is $\mathrm{Var}(\hat \theta_1) = \theta^2 / (3n)$ because $\mathrm{Var}(\bar X) = \theta^2 / (12n)$.
2. **Max-based estimator**: $\hat \theta_2 = \dfrac{n+1}{n} X_{(n)}$ with $X_{(n)} = \max_i X_i$. Here $\mathrm{E}[X_{(n)}] = \dfrac{n}{n+1} \theta$, so the bias correction gives an unbiased estimator. Its variance is $\mathrm{Var}(\hat \theta_2) = \dfrac{\theta^2}{n (n+2)}$.

Compare the variances:

$$
\mathrm{Var}(\hat \theta_1) = \frac{\theta^2}{3n}, \qquad \mathrm{Var}(\hat \theta_2) = \frac{\theta^2}{n (n+2)}.
$$

For every $n \ge 1$, $\frac{1}{n (n+2)} < \frac{1}{3n}$, so the estimator based on the sample maximum has smaller variance. Because $X_{(n)}$ is complete and sufficient for $\theta$ in this family, the Lehmann-Scheffe theorem implies $\hat \theta_2$ is the MVUE.

---

## 4) What is theta? Parameterization and consistency (exponential example)

Take $X_i \sim \mathrm{Exp}(\lambda)$.

- If $\theta = \lambda$ (the rate), then $I(\lambda) = 1 / \lambda^2$ and the CRLB for an unbiased estimator $\hat \lambda$ is $\lambda^2 / n$.
- If instead $\theta = \sigma^2 = 1 / \lambda^2$ (the mean-square or variance), Fisher information transforms through the chain rule:

$$
I_\theta(\theta) = I_\lambda(\lambda) \left(\frac{d\lambda}{d\theta}\right)^2,
$$

and the target $g(\theta)$ transforms accordingly. The CRLB statements remain consistent because Fisher information and the target derivative both pick up the same Jacobian factors.

---

## 5) Normal model: three common scenarios

Assume $X_i \sim \mathcal{N}(\mu, \sigma^2)$.

1. **Known $\sigma^2$, estimate $\mu$**: $\bar X$ is unbiased with $\mathrm{Var}(\bar X) = \sigma^2 / n$ and it reaches the CRLB, so $\bar X$ is efficient and MVUE.
2. **Known $\mu$, estimate $\sigma^2$**: $\hat \sigma^2 = \tfrac{1}{n} \sum (X_i - \mu)^2$ is unbiased with $\mathrm{Var}(\hat \sigma^2) = 2 \sigma^4 / n$, reaching the CRLB and therefore efficient MVUE.
3. **Both $\mu$ and $\sigma^2$ unknown**: $S^2 = \tfrac{1}{n-1} \sum (X_i - \bar X)^2$ is unbiased with $\mathrm{Var}(S^2) = 2 \sigma^4 / (n-1)$, which is larger than the single-parameter CRLB $2 \sigma^4 / n$. The bound is unattainable in this two-parameter setting, yet by sufficiency and completeness of $(\bar X, S^2)$, $\bar X$ is the MVUE for $\mu$ and $S^2$ is the MVUE for $\sigma^2$.

---

## 6) Extra: exponential variance and the true lower bound

If the target is the exponential variance $1 / \lambda^2$, the CRLB derived via the $\lambda$ parameterization gives $4 / (n \lambda^4)$. No unbiased estimator attains this bound exactly. Projecting an unbiased estimator onto the complete sufficient statistic $T = \sum X_i$ still leaves a variance slightly above the bound; the gap vanishes asymptotically as $n$ grows.

---

## 7) Wrap-up mapped to the checklist

- The CRLB derivation is explicit, including the differentiation-under-the-integral step that leads to the covariance with the score.
- For $\mathrm{Unif}(0, \theta)$, $2 \bar X$ and $\dfrac{n+1}{n} X_{(n)}$ are both unbiased, but the max-based estimator has strictly smaller variance and is the MVUE.
- The MVUE definition and Lehmann-Scheffe path to proving optimality are summarized.
- The Cramer-Rao bound and its equality condition are stated clearly.
- Re-parameterizations (such as exponential rate versus variance) produce consistent conclusions once the chain rule for Fisher information is applied.
- In the normal model, the single-parameter cases attain the CRLB, whereas the two-parameter case leaves $S^2$ as MVUE but not efficient.

---

Last updated: 2025-10-20T03:12:20Z





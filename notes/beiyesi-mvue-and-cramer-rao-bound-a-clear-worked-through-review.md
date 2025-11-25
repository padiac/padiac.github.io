## What This Note Covers (Checklist)

- **Full CRLB derivation** (with the score function and the exact step you asked about)
- **Uniform(0, $\theta$)**: two unbiased estimators - $2\bar X$ and $\tfrac{n+1}{n}X_{(n)}$ - and which one is better
- Clean **definition of MVUE** and how to prove it via **Lehmann-Scheffe**
- Statement and use of the **Cramer-Rao Lower Bound**
- **"What is $\theta$?"** - parameterization choices (for example, Exponential: rate $\lambda$ vs variance $1/\lambda^2$) and why results are consistent
- **Normal model**: ($\sigma^2$ known) estimate $\mu$, ($\mu$ known) estimate $\sigma^2$, and ($\mu$, $\sigma^2$ both unknown): why the CRLB is not attained for $\sigma^2$ but $\bar X$ and the unbiased sample variance are still best in their classes

---

## 1) Cramer-Rao Lower Bound - Derivation You Asked About

Let $X_1, \ldots, X_n$ have joint log-likelihood $\ell(\theta) = \sum_{i=1}^n \log f(X_i; \theta)$. Define the **score** $S(\theta) = \partial \ell(\theta) / \partial \theta$. Under regularity (differentiate under the integral, etc.) we have

$$
\mathbb{E}_\theta[S(\theta)] = 0
$$

$$
\operatorname{Var}_\theta(S(\theta)) = n I(\theta)
$$

where $I(\theta)$ is the **per-sample Fisher information**.

Suppose $\hat g = \hat g(X_1, \ldots, X_n)$ is **unbiased** for $g(\theta)$: $\mathbb{E}_\theta[\hat g] = g(\theta)$. Differentiate both sides with respect to $\theta$ (this is the step you were stuck on earlier):

$$
\frac{d}{d\theta} \mathbb{E}_\theta[\hat g] = \frac{d}{d\theta} \int \hat g(x) f(x; \theta)  dx
$$

Differentiating under the integral sign yields

$$
\frac{d}{d\theta} \int \hat g(x) f(x; \theta)  dx = \int \hat g(x) \frac{\partial}{\partial \theta} f(x; \theta)  dx
$$

and therefore

$$
\int \hat g(x) \frac{\partial}{\partial \theta} f(x; \theta)  dx = \mathbb{E}_\theta\left[\hat g \frac{\partial}{\partial \theta} \log f(X; \theta)\right]
$$

The left side equals $g'(\theta)$; the right side equals $\operatorname{Cov}(\hat g, S) + \mathbb{E}[\hat g] \mathbb{E}[S]$. Because $\mathbb{E}[S] = 0$,

$$
g'(\theta) = \operatorname{Cov}_\theta(\hat g, S)
$$

By Cauchy-Schwarz (covariance inequality),

$$
\left(\operatorname{Cov}(A, B)\right)^2 \leq \operatorname{Var}(A) \operatorname{Var}(B)
$$

Taking $A = \hat g$ and $B = S$ gives

$$
\left(g'(\theta)\right)^2 \leq \operatorname{Var}(\hat g) \operatorname{Var}(S) = \operatorname{Var}(\hat g)  n I(\theta)
$$

Thus

$$
\operatorname{Var}_\theta(\hat g) \geq \frac{\left(g'(\theta)\right)^2}{n I(\theta)}
$$

Equality holds if and only if $\hat g - a(\theta)$ is a linear function of the score.

---

## 2) MVUE - Definition and How to Show It

- **MVUE**: among all **unbiased** estimators of the same target $g(\theta)$, the one with **minimum variance** (for every $\theta$).
- **Lehmann-Scheffe**: if $T(X)$ is **complete and sufficient** for $\theta$ and $\tilde g(X)$ is **any** unbiased estimator of $g(\theta)$, then $\mathbb{E}[\tilde g \mid T]$ is the **unique MVUE**.

---

## 3) Uniform(0, $\theta$): Two Unbiased Estimators and Which Is Better

Let $X_1, \ldots, X_n \overset{\text{iid}}{\sim} \operatorname{Unif}(0, \theta)$. Two classic unbiased estimators of $\theta$:

1. **Method of moments**: $\hat\theta_1 = 2\bar X$. Because $\mathbb{E}[\bar X] = \theta/2$, we have $\mathbb{E}[2\bar X] = \theta$. Also, $\operatorname{Var}(\bar X) = \theta^2/(12n)$, so $\operatorname{Var}(\hat\theta_1) = \theta^2/(3n)$.
2. **Based on the maximum**: $\hat\theta_2 = \dfrac{n+1}{n} X_{(n)}$ where $X_{(n)} = \max_i X_i$. Because $\mathbb{E}[X_{(n)}] = \dfrac{n}{n+1} \theta$, we get $\mathbb{E}[\hat\theta_2] = \theta$. Moreover, $\operatorname{Var}(X_{(n)}) = \dfrac{n \theta^2}{(n+1)^2 (n+2)}$, so $\operatorname{Var}(\hat\theta_2) = \dfrac{\theta^2}{n (n+2)}$.

Comparison:

$$
\operatorname{Var}(\hat\theta_1) = \frac{\theta^2}{3n}
$$

$$
\operatorname{Var}(\hat\theta_2) = \frac{\theta^2}{n (n+2)}
$$

For all $n \geq 1$, $\dfrac{1}{n (n+2)} < \dfrac{1}{3n}$. Hence the **max-based estimator** has smaller variance. Via completeness and sufficiency of $X_{(n)}$ in this model, $\hat\theta_2$ is the **MVUE**.

---

## 4) "What Is $\theta$?" - Parameterization (Exponential Example)

For $X_i \sim \operatorname{Exp}(\lambda)$, you may choose different **parameters**:

- If $\theta = \lambda$ (rate), then $I(\lambda) = 1/\lambda^2$ and the CRLB for unbiased $\hat\lambda$ is $\lambda^2/n$.
- If you instead take $\theta = \sigma^2 = 1/\lambda^2$ (variance), Fisher information transforms by

$$
I_\theta(\theta) = I_\lambda(\lambda) \left(\frac{d\lambda}{d\theta}\right)^2
$$

and $g$ transforms accordingly. The CRLB conclusion is **consistent** across parameterizations (invariance under smooth re-parameterization).

---

## 5) Normal Model - Three Scenarios

Let $X_i \sim \mathcal{N}(\mu, \sigma^2)$.

1. **$\sigma^2$ known, estimate $\mu$**: $\bar X$ is unbiased, $\operatorname{Var}(\bar X) = \sigma^2/n$, and it **attains the CRLB**, so it is an efficient MVUE.
2. **$\mu$ known, estimate $\sigma^2$**: $\hat\sigma^2 = \tfrac{1}{n} \sum (X_i - \mu)^2$ is unbiased, $\operatorname{Var}(\hat\sigma^2) = 2\sigma^4/n$, and it **attains the CRLB**, so it is an efficient MVUE.
3. **$\mu$ and $\sigma^2$ both unknown**: $S^2 = \tfrac{1}{n-1} \sum (X_i - \bar X)^2$ is unbiased with $\operatorname{Var}(S^2) = 2\sigma^4/(n-1)$, which exceeds the CRLB $2\sigma^4/n$. The CRLB is not attained, but by completeness and sufficiency, $\bar X$ is the MVUE for $\mu$ and $S^2$ is the MVUE for $\sigma^2$.

---

## 6) Extra: Exponential Variance and the "True" Lower Bound

Target variance $1/\lambda^2$. The ($\lambda$-parameterized) CRLB gives $4/(n \lambda^4)$, but **no unbiased estimator attains it**. The UMVU built from $T = \sum X_i$ (complete sufficient) has variance slightly above the CRLB; asymptotically the gap closes.

---

## 7) Wrap-Up

- The CRLB **derivation step** you flagged is now explicit (d/d$\theta$ of expectation becomes covariance with the score).
- **Uniform(0, $\theta$)**: compared $2\bar X$ vs $\tfrac{n+1}{n} X_{(n)}$; the max-based estimator wins and is MVUE.
- Clear **MVUE definition** plus a reminder of **Lehmann-Scheffe**.
- **CRLB** stated and the equality condition explained.
- **$\theta$ parameterization** ($\lambda$ vs variance) stays consistent via information and chain-rule transforms.
- **Normal family**: two single-parameter cases are efficient; in the two-parameter case, $S^2$ is MVUE but not efficient.

---

_Last updated: 2025-10-20 02:52:40Z_



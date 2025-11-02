### Uniformly Minimum Variance Unbiased Estimator

## 1. The Goal: Estimating a Function of Parameters

We start with a **parametric model**, a family of probability distributions:

$$
\{ f(x \mid \theta), \theta \in \Theta \}.
$$

Here:

- $ \theta $ (Greek theta) is the **parameter** (possibly a vector).
- $ \Theta $ is the **parameter space**, i.e., all possible values of $ \theta $.
- $ X = (X_1, X_2, \ldots, X_n) $ is a random sample drawn from that distribution.

Our aim is **not necessarily to estimate $ \theta $ itself**, but rather some function of it, denoted by $ g(\theta) $.

Examples:

| Target Quantity | Parameter Function $ g(\theta) $ | Example |
|-----------------|----------------------------------|---------|
| Mean | $ g(\theta) = \mu $ | Normal distribution |
| Variance | $ g(\theta) = \sigma^2 $ | Normal distribution |
| Rate squared | $ g(\theta) = \lambda^2 $ | Exponential distribution |
| Reliability | $ g(\theta) = e^{-\lambda t} $ | Poisson or exponential |

We want an **estimator** $ \delta(X) $, a function of the data, that satisfies two conditions:

1. **Unbiasedness:** $ E_\theta[\delta(X)] = g(\theta) $ for all $ \theta \in \Theta $.
2. **Minimum variance:** among all unbiased estimators of $ g(\theta) $, $ \delta(X) $ should have the smallest variance.

Such an estimator is called the **UMVUE** (Uniformly Minimum Variance Unbiased Estimator).

## 2. The Role of the Sufficient Statistic $ T(X) $

The data $ X $ may contain redundant randomness that carries no information about $ \theta $. A **sufficient statistic** $ T(X) $ is a function of the sample that retains all information about $ \theta $. Formally, $ T(X) $ is sufficient if

$$
f(x \mid \theta) = g(T(x), \theta) h(x),
$$

which implies that, conditional on $ T(X) $, the sample $ X $ provides no additional information about $ \theta $.

So, in principle:

> Once $ T(X) $ is known, the rest of the data can be ignored for inference on $ \theta $.

## 3. The Lehmann-Scheffe Theorem

This theorem is the foundation of UMVUE theory.

**Lehmann-Scheffe Theorem:** If $ T(X) $ is a **complete and sufficient statistic** for parameter $ \theta $, and $ \delta(X) $ is any unbiased estimator of $ g(\theta) $, then

$$
\delta^*(X) = E[\delta(X) \mid T(X)]
$$

is the **unique** unbiased estimator of $ g(\theta) $ that has **minimum variance** among all unbiased estimators.

### 3.1 Interpretation of the Symbols

- $ \delta(X) $: any unbiased estimator (may use the whole data sample).
- $ T(X) $: the sufficient statistic, summarizing all information about $ \theta $.
- $ E[\delta(X) \mid T(X)] $: conditional expectation that averages out irrelevant randomness while keeping the information in $ T(X) $.
- $ \delta^*(X) $: the improved estimator after conditioning.
- $ g(\theta) $: the true parameter function we are estimating.

### 3.2 Why This Works

Conditioning on $ T(X) $ removes noise while preserving the signal. Formally,

$$
E_\theta[\delta^*(X)] = E_\theta[E[\delta(X) \mid T(X)]] = E_\theta[\delta(X)] = g(\theta),
$$

and by the law of total variance,

$$
\mathrm{Var}_{\theta}\big(\delta^*(X)\big) \le \mathrm{Var}_{\theta}\big(\delta(X)\big).
$$

If $ T(X) $ is **complete**, no other unbiased estimator can have the same mean and smaller variance, so $ \delta^*(X) $ is the **unique** UMVUE.

## 4. Geometric Intuition

Think of random variables as vectors in a Hilbert space with inner product

$$
\langle f, g \rangle = E[f(X) g(X)].
$$

The subspace of all functions of $ T(X) $ represents "the space of all information about $ \theta $." Taking $ E[\delta(X) \mid T(X)] $ is the orthogonal projection of $ \delta(X) $ onto that subspace.

So geometrically:

> $ \delta^*(X) $ is the part of $ \delta(X) $ that is purely about $ \theta $, with all irrelevant noise removed. Conditioning gives smaller variance because it removes random orthogonal noise.

## 5. The Direct Construction Method

When the distribution of $ T(X) $ is explicitly known, we can often find $ \delta(T) $ satisfying $ E_\theta[\delta(T)] = g(\theta) $ directly, without explicitly performing the conditional expectation. This is called the **Direct Construction Method**, and in simple exponential-family cases, it leads to the same result as the Lehmann-Scheffe theorem.

## 6. Example: Exponential Distribution

Let $ X_i \sim \mathrm{Exp}(\lambda) $ with density $ f(x \mid \lambda) = \lambda e^{-\lambda x} $.

### 6.1 Step 1 - Find the Sufficient Statistic

The joint likelihood is

$$
L(\lambda; x_1, \ldots, x_n) = \lambda^n \exp\left(-\lambda \sum_{i=1}^n x_i\right).
$$

This can be rewritten as

$$
L(\lambda; x) = \lambda^n e^{-\lambda T(x)}, \quad T(x) = \sum_{i=1}^n x_i.
$$

By the factorization theorem, $ T(X) = \sum X_i $ is a sufficient statistic for $ \lambda $. Moreover, since the exponential family has a canonical form, $ T(X) $ is not only sufficient but also **complete**.

### 6.2 Step 2 - Distribution of the Sufficient Statistic

Because each $ X_i $ is exponential, their sum $ T = \sum X_i $ follows a Gamma distribution:

$$
T \sim \mathrm{Gamma}(n, \lambda)
$$

with density

$$
f_T(t \mid \lambda) = \frac{\lambda^n t^{n-1} e^{-\lambda t}}{(n-1)!}, \quad t > 0.
$$

Its moments are

$$
E[T] = \frac{n}{\lambda}, \quad \mathrm{Var}(T) = \frac{n}{\lambda^2}, \quad E[T^{-2}] = \frac{\lambda^2}{(n-1)(n-2)} \quad (n > 2).
$$

### 6.3 Step 3 - The Parameter Function We Want

Suppose we want to estimate $ g(\lambda) = \lambda^2 $. We look for a function $ \delta(T) $ of the sufficient statistic $ T $ whose expectation equals $ \lambda^2 $.

### 6.4 Step 4 - Construct the Function $ \delta(T) $

We use the known moment formula

$$
E[T^{-2}] = \frac{\lambda^2}{(n-1)(n-2)}.
$$

To make the expectation equal to $ \lambda^2 $, multiply both sides by $ (n-1)(n-2) $:

$$
E[(n-1)(n-2) T^{-2}] = \lambda^2.
$$

Hence we define the estimator

$$
\delta(T) = (n-1)(n-2) T^{-2}.
$$

### 6.5 Step 5 - Check the Conditions

1. **Unbiasedness:** $ E[\delta(T)] = \lambda^2 $.
2. **Dependence only on the sufficient statistic $ T $:** $ \delta(T) $ is a function of $ T $ only.
3. **Completeness of $ T $:** For the exponential family, $ T $ is complete, so $ \delta(T) $ is the **unique** UMVUE for $ \lambda^2 $.

### 6.6 Step 6 - Connection to Lehmann-Scheffe

If we had started from any unbiased estimator $ \delta(X) $, the theorem says

$$
E[\delta(X) \mid T(X)] = (n-1)(n-2) T^{-2}.
$$

So, even though we never explicitly computed the conditional expectation, our direct construction is exactly what the theorem would produce automatically.

In other words:

> The manual estimator you derived by algebra is the same as what Lehmann-Scheffe would give by conditioning.

## 7. Interpretation of the Symbols

| Symbol | Meaning | Typical Example |
|--------|---------|-----------------|
| $ \theta $ | Parameter vector | $ (\mu, \sigma^2) $ or $ \lambda $ |
| $ g(\theta) $ | Function of parameter to be estimated | $ \mu $, $ \sigma^2 $, $ \lambda^2 $ |
| $ X $ | Sample data | $ X_1, \ldots, X_n $ |
| $ T(X) $ | Sufficient statistic summarizing all info about $ \theta $ | $ \sum X_i $ or $ \bar{X} $ |
| $ \delta(X) $ | Any unbiased estimator (may depend on all $ X_i $) | Sample mean, etc. |
| $ E[\delta(X) \mid T(X)] $ | Conditioning operation that removes irrelevant randomness | Projector to informative subspace |
| $ \delta^*(X) $ | Final estimator, function of $ T(X) $, the UMVUE | $ (n-1)(n-2) T^{-2} $ in this example |

## 8. The Unified Picture

| Step | Concept | Description |
|------|---------|-------------|
| 1 | Define what you want | Identify $ g(\theta) $ |
| 2 | Identify sufficient statistic | Use factorization theorem |
| 3 | Ensure completeness | Check exponential-family structure |
| 4 | Construct or project | Find $ \delta(T) $ or compute $ E[\delta \mid T] $ |
| 5 | Verify unbiasedness | Confirm $ E_\theta[\delta] = g(\theta) $ |
| 6 | Conclude optimality | By Lehmann-Scheffe, it is the UMVUE |

## 9. One-Sentence Summary

> The UMVUE is the unique unbiased estimator that depends only on the sufficient statistic. Equivalently, it is what you get when you project any unbiased estimator onto the information carried by that statistic; in the exponential example, both the direct construction and the Lehmann-Scheffe approach lead to $ \delta(T) = (n-1)(n-2) T^{-2} $.



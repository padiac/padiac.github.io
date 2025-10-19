## 1. Definition

Consider random variables $X$ and $Y$ with finite variances. Denote

$$
\mu_X = \mathbb{E}[X], \quad \mu_Y = \mathbb{E}[Y], \quad
\sigma_X^2 = \operatorname{Var}(X), \quad \sigma_Y^2 = \operatorname{Var}(Y).
$$

Their covariance and correlation are

$$
\operatorname{Cov}(X, Y) = \mathbb{E}\big[(X - \mu_X)(Y - \mu_Y)\big], \qquad
\rho_{X,Y} = \frac{\operatorname{Cov}(X, Y)}{\sigma_X \sigma_Y} \in [-1, 1].
$$

Variables are **uncorrelated** when $\operatorname{Cov}(X, Y) = 0$. This only removes linear dependence; independence requires stronger conditions.

---

## 2. Fundamental Inequalities

The Cauchy-Schwarz inequality yields

$$
\lvert \operatorname{Cov}(X, Y) \rvert \le \sigma_X \sigma_Y,
$$

so $\lvert \rho_{X,Y} \rvert \le 1$. Equality holds precisely when constants $a \ne 0$ and $b$ exist such that $Y = aX + b$ almost surely; the relationship is then perfectly linear.

---

## 3. Uncorrelated vs Independent

| Relationship | Meaning | Implication |
|--------------|---------|-------------|
| Independent | $f(x, y) = f_X(x) f_Y(y)$ | $\Rightarrow$ uncorrelated |
| Uncorrelated | $\operatorname{Cov}(X, Y) = 0$ | $\nRightarrow$ independent (in general) |

Being uncorrelated rules out only linear dependence; nonlinear dependence can remain.

---

## 4. Counterexamples

**Example A (uniform on a disk).** Let $(X, Y)$ be uniformly distributed on the unit disk. Symmetry gives $\mathbb{E}[X] = \mathbb{E}[Y] = 0$ and $\mathbb{E}[XY] = 0$, so $\operatorname{Cov}(X, Y) = 0$. Yet knowing $X$ restricts the possible values of $Y$, so the pair is dependent.

**Example B (nonlinear dependence).** Let $X \sim \operatorname{Uniform}(-1, 1)$ and set $Y = X^2$. Then $\mathbb{E}[X] = \mathbb{E}[X^3] = 0$, giving

$$
\operatorname{Cov}(X, Y) = \mathbb{E}[X^3] - \mathbb{E}[X]\,\mathbb{E}[Y] = 0,
$$

but $Y$ is fully determined by $X$, so the variables are dependent.

---

## 5. Correlation Interpretation

- $\rho_{X,Y} > 0$: $Y$ tends to increase with $X$ (upward trend).
- $\rho_{X,Y} < 0$: $Y$ tends to decrease with $X$ (downward trend).
- $\rho_{X,Y} = 0$: no linear trend, though nonlinear structure may remain.

Graphically, $\lvert \rho_{X,Y} \rvert = 1$ means the scatter lies on a line, while $\lvert \rho_{X,Y} \rvert \approx 0$ looks like an unstructured cloud.

---

## 6. Bivariate Normal Exception

If $(X, Y)$ is bivariate normal,

$$
(X, Y) \sim \mathcal{N}_2(\mu_1, \mu_2, \sigma_1^2, \sigma_2^2, \rho),
$$

then

$$
\operatorname{Cov}(X, Y) = 0 \quad \Longleftrightarrow \quad X \text{ and } Y \text{ are independent}.
$$

When $\rho = 0$, the joint density factorizes as $f(x, y) = f_X(x) f_Y(y)$. This equivalence between zero correlation and independence is specific to the multivariate normal family.

---

## 7. Summary

- Independence $\Rightarrow$ uncorrelated.
- Uncorrelated $\nRightarrow$ independence (except in the Gaussian case).
- $\lvert \operatorname{Cov}(X, Y) \rvert \le \sigma_X \sigma_Y$.
- $\lvert \rho_{X,Y} \rvert \le 1$, with equality only for linear relationships.

---

## 8. Supplement: Why PCA Uses the Sample Covariance Matrix

Principal Component Analysis (PCA) finds directions along which variance is maximized.

### 8.1 Theoretical covariance

For a random vector $X$ with mean $\mu$,

$$
\Sigma = \mathbb{E}\big[(X - \mu)(X - \mu)^\top\big]
$$

is the population covariance matrix that PCA uses in theory.

### 8.2 Practical estimation

Given samples $x_1, \dots, x_n$, practitioners estimate $\Sigma$ with the sample covariance matrix

$$
S = \frac{1}{n-1} \sum_{i=1}^{n} (x_i - \hat{\mu})(x_i - \hat{\mu})^\top, \qquad
\hat{\mu} = \frac{1}{n} \sum_{i=1}^{n} x_i.
$$

### 8.3 Statistical justification

The estimator $S$ is unbiased:

$$
\mathbb{E}[S] = \Sigma.
$$

This mirrors the unbiased sample mean and variance. Geometrically, $S$ captures the observed data cloud, so its eigenvectors reveal the sample’s principal directions.

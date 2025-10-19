## 1. Discrete Additive Families ("Three Classic Brothers")

The multinomial, binomial, and Poisson distributions form a discrete hierarchy where additivity is preserved.

| Attribute | Multinomial | Binomial | Poisson |
|:----------|:-----------|:--------|:--------|
| **Parameters** | $(n; p_1,\dots,p_k)$ | $(n, p)$ | $\lambda$ |
| **Additive rule** | Merge categories $\Rightarrow$ add probabilities | Add trials $n$ | Add rates $\lambda$ |
| **Interpretation** | Multi-category trials | Success or failure experiments | Limit of rare events |

---

## 2. Continuous Transformation and Convolution

### 2.1 Jacobian Transformation

Variable transformation $(X_1,\dots,X_n) \mapsto (Y_1,\dots,Y_n)$ introduces a Jacobian factor to preserve probability:

$$
f_Y(y_1,\dots,y_n) = f_X\bigl(h_1(y),\dots,h_n(y)\bigr) \left|\det\left(\frac{\partial h}{\partial y}\right)\right|.
$$

### 2.2 Sum of Variables (Convolution)

For $Y = X_1 + X_2$,

$$
l(y) = \int_{-\infty}^{\infty} f_1(x_1) f_2(y - x_1)\,dx_1,
$$

which defines the convolution of $f_1$ and $f_2$.

### 2.3 Normal Distribution Closure

If $X_1 \sim \mathcal{N}(\mu_1, \sigma_1^2)$ and $X_2 \sim \mathcal{N}(\mu_2, \sigma_2^2)$ are independent, then

$$
Y = X_1 + X_2 \sim \mathcal{N}(\mu_1 + \mu_2, \sigma_1^2 + \sigma_2^2).
$$

Normal distributions are closed under addition, a property sometimes nicknamed the "earthworm theorem."

---

| Concept | Variable transformation | Sum of variables | Normal closure |
|:--------|:------------------------|:------------------|:-----------------|
| **Formula** | $f_Y = f_X \times \lvert\det(\partial h / \partial y)\rvert$ | $l(y) = \int f_1(x) f_2(y - x)\,dx$ | $\mathcal{N}(\mu_1 + \mu_2, \sigma_1^2 + \sigma_2^2)$ |
| **Interpretation** | Density scaling by coordinate change | Convolution integral | Gaussian stability |

---

## 3. Gamma, Beta, and Chi-Square Distributions

### 3.1 Gamma Function $\Gamma(x)$

$$
\Gamma(x) = \int_0^\infty t^{x-1} e^{-t}\,dt
$$

Properties: $\Gamma(1) = 1$, $\Gamma(1/2) = \sqrt{\pi}$, $\Gamma(x+1) = x \Gamma(x)$. For integer $n$, $\Gamma(n) = (n-1)!$; for half-integers, $\Gamma(n/2)$ has a closed form involving $\sqrt{\pi}$.

### 3.2 Beta Function $B(x,y)$

$$
B(x,y) = \int_0^1 t^{x-1}(1-t)^{y-1}\,dt = \frac{\Gamma(x)\Gamma(y)}{\Gamma(x+y)}.
$$

### 3.3 Chi-Square Distribution $\chi^2_n$

If $X_1,\dots,X_n$ are i.i.d. $\mathcal{N}(0,1)$, then $Y = \sum_i X_i^2 \sim \chi^2_n$ with density

$$
k_n(x) = \frac{1}{2^{n/2} \Gamma(n/2)} x^{n/2-1} e^{-x/2}, \quad x > 0.
$$

Properties:

- Additivity: $\chi^2_{n_1} + \chi^2_{n_2} = \chi^2_{n_1+n_2}$.
- Connection: $\chi^2_n \equiv \Gamma(k = n/2, \theta = 2)$.
- Basis: the square of a standard normal variable is $\chi^2_1$.

---

## 4. Ratios and Derived Families: $t$ and $F$ Distributions

With the square of a normal linked to $\chi^2$, ratios of these quantities yield two cornerstone distributions in inferential statistics.

### 4.1 $t$ Distribution

Let $X \sim \mathcal{N}(0,1)$, $Z \sim \chi^2_n$, and $X$ and $Z$ be independent. Then

$$
T = \frac{X}{\sqrt{Z/n}}
$$

follows a Student's $t$ distribution with $n$ degrees of freedom. Its density is

$$
t_n(y) = \frac{\Gamma\left(\frac{n+1}{2}\right)}{\sqrt{n\pi}\,\Gamma\left(\frac{n}{2}\right)} \left(1 + \frac{y^2}{n}\right)^{-(n+1)/2}.
$$

As $n \to \infty$, $t_n$ converges to $\mathcal{N}(0,1)$.

### 4.2 $F$ Distribution

If $X_1 \sim \chi^2_n / n$ and $X_2 \sim \chi^2_m / m$ are independent, then

$$
F = \frac{X_1}{X_2}
$$

follows an $F$ distribution with $(n,m)$ degrees of freedom. Its density is

$$
f_{n,m}(y) = \frac{\Gamma\left(\frac{m+n}{2}\right)}{\Gamma\left(\frac{m}{2}\right)\Gamma\left(\frac{n}{2}\right)} \left(\frac{m}{n}\right)^{m/2} y^{m/2-1} \left(1 + \frac{m}{n} y\right)^{-(m+n)/2}, \quad y > 0.
$$

The $F$ distribution measures the ratio of two independent variance estimates, central to ANOVA and regression model testing.

### 4.3 Relationships Among $\chi^2$, $t$, and $F$

| Attribute | $\chi^2_n$ | $t_n$ | $F_{n,m}$ |
|:----------|:-----------|:------|:-----------|
| **Constructed from** | $\sum \mathcal{N}(0,1)^2$ | $\mathcal{N}(0,1) / \sqrt{\chi^2_n / n}$ | $(\chi^2_n / n) / (\chi^2_m / m)$ |
| **Expression** | Square-sum | Ratio (mean over std estimate) | Ratio of variances |
| **Degrees of freedom** | $n$ | $n$ | $(n,m)$ |

---

## 5. Sample Statistics and Their Distributions

### 5.1 Sample Variance

Given $X_1,\dots,X_n \sim \mathcal{N}(\mu,\sigma^2)$,

$$
\bar{X} = \frac{1}{n} \sum_i X_i, \qquad S^2 = \frac{1}{n-1} \sum_i (X_i - \bar{X})^2.
$$

Then $(n-1)S^2 / \sigma^2 \sim \chi^2_{n-1}$.

### 5.2 $t$ Statistic for Mean Testing

The statistic

$$
T = \frac{\sqrt{n} (\bar{X} - \mu)}{S}
$$

follows $t_{n-1}$, forming the basis of the $t$-test when $\sigma^2$ is unknown.

### 5.3 $F$ Statistic for Variance Comparison

For two independent samples $X_1,\dots,X_n \sim \mathcal{N}(\mu_1,\sigma_1^2)$ and $Y_1,\dots,Y_m \sim \mathcal{N}(\mu_2,\sigma_2^2)$,

$$
F = \frac{S_1^2 / \sigma_1^2}{S_2^2 / \sigma_2^2} \sim F_{n-1,m-1},
$$

which underpins ANOVA and variance comparison tests.

---

## 6. The Chi-Square, $t$, and $F$ Chain

In classical statistics, the distributions $\chi^2$, $t$, and $F$ form a single logical chain derived from the normal distribution. Each arises through squaring, summing, normalizing, or taking ratios.

### 6.1 Starting Point: Sample Statistics

Samples: $X_1, X_2, \dots, X_n \sim \mathcal{N}(\mu, \sigma^2)$ (independent and identically distributed)

Sample mean: $\bar{X} = \frac{1}{n} \sum_i X_i$

Sample variance: $S^2 = \frac{1}{n-1} \sum_i (X_i - \bar{X})^2$

### 6.2 Sample Variance $\Rightarrow$ Chi-Square Distribution

Standardized variable: $Z_i = \frac{X_i - \mu}{\sigma} \sim \mathcal{N}(0, 1)$

Sum of squares: $\sum_i Z_i^2 \sim \chi^2_n$

Equivalently: $\frac{(n-1) S^2}{\sigma^2} \sim \chi^2_{n-1}$

Meaning: the sample variance is random and its uncertainty follows a $\chi^2$ law, underpinning confidence intervals for $\sigma^2$ and variance tests.

### 6.3 Distribution of the Sample Mean

$\bar{X} \sim \mathcal{N}\bigl(\mu, \sigma^2 / n\bigr)$

If $\sigma$ is known: $\frac{\bar{X} - \mu}{\sigma / \sqrt{n}} \sim \mathcal{N}(0, 1)$

In practice, $\sigma$ is unknown and replaced by $S$.

### 6.4 Replacing $\sigma$ with $S$ $\Rightarrow$ $t$ Distribution

$$
T = \frac{\bar{X} - \mu}{S / \sqrt{n}} \sim t_{n-1}
$$

Structural form: $t = \frac{\text{Normal}}{\sqrt{\text{Chi-square} / \text{df}}}$

Meaning: the $t$ distribution compensates for the extra uncertainty of using the sample variance to estimate the population variance. As $n$ increases, $t_{n-1}$ approaches $\mathcal{N}(0,1)$.

### 6.5 Ratio of Two Sample Variances $\Rightarrow$ $F$ Distribution

Two independent samples:

$X_1,\dots,X_n \sim \mathcal{N}(\mu_1, \sigma_1^2)$, $Y_1,\dots,Y_m \sim \mathcal{N}(\mu_2, \sigma_2^2)$

$\frac{(n-1) S_1^2}{\sigma_1^2} \sim \chi^2_{n-1}$, $\frac{(m-1) S_2^2}{\sigma_2^2} \sim \chi^2_{m-1}$

$$
F = \frac{S_1^2 / \sigma_1^2}{S_2^2 / \sigma_2^2} \sim F_{n-1, m-1}
$$

Structural form: $F = \frac{\text{Chi-square}_1 / \text{df}_1}{\text{Chi-square}_2 / \text{df}_2}$

### 6.6 Summary Table

| Attribute | $\chi^2$ | $t$ | $F$ |
|:----------|:--------|:----|:----|
| **Construction** | $\sum \left(\frac{X_i - \mu}{\sigma}\right)^2$ or $\frac{(n-1) S^2}{\sigma^2}$ | $\frac{\bar{X} - \mu}{S / \sqrt{n}}$ | $\frac{S_1^2 / \sigma_1^2}{S_2^2 / \sigma_2^2}$ |
| **Degrees of freedom** | $n$ or $n-1$ | $n-1$ | $(n-1, m-1)$ |
| **Common use** | Variance estimation and testing | Mean testing and confidence intervals | Variance ratio tests, ANOVA |

Conceptual chain:

Normal $\rightarrow$ square and sum $\rightarrow$ Chi-square $\rightarrow$ divide by $\sqrt{\chi^2 / \text{df}}$ $\rightarrow$ $t$

Two $(\chi^2 / \text{df})$ ratios $\rightarrow$ $F$

---

## 7. Summary: The Continuous Family Hierarchy

| Attribute | Normal | Chi-square | $t$ | $F$ |
|:----------|:-------|:----------|:----|:----|
| **Generated from** | Base | Sum of squares | Normal over $\sqrt{\chi^2 / n}$ | $(\chi^2_1 / n_1) / (\chi^2_2 / n_2)$ |
| **Key form** | $\mathcal{N}(\mu,\sigma^2)$ | $\sum X_i^2$ | Ratio | Ratio |
| **Characteristic** | Symmetric bell shape | Positive-only variance measure | Standardized mean deviation | Variance ratio |

---

**Conceptual chain:** Normal $\rightarrow$ Chi-square $\rightarrow$ ($t$, $F$). Each stage arises from combining or scaling the previous one by summing squares, taking ratios, and normalizing variances, revealing how classical inferential statistics stem from the properties of the normal distribution.

---

## Appendix: Clarifications on Sample Variance and $t$ Distribution

This appendix highlights two common misunderstandings:

1. How the sample variance relates to the chi-square distribution.
2. Where the $\sqrt{n}$ factor in the $t$ distribution originates.

### A. Sample Variance and the Chi-Square Distribution

The sample variance is defined as

$$
S^2 = \frac{1}{n-1} \sum_i (X_i - \bar{X})^2.
$$

#### A.1 Scaled chi-square, not chi-square

Textbooks sometimes say "the sample variance is chi-square distributed"; strictly speaking this is not precise. The standardized quantity

$$
Q = \frac{(n-1) S^2}{\sigma^2} \sim \chi^2_{n-1}
$$

follows a chi-square distribution if $\sigma^2$ is known. Therefore

$$
S^2 = \frac{\sigma^2}{n-1} Q,
$$

so $S^2$ itself is a scaled chi-square random variable rather than an exact chi-square.

#### A.2 Gamma characterization when $\sigma^2$ is unknown

With unknown $\sigma^2$, the distribution of $S^2$ is better described as a gamma distribution:

$$
S^2 \sim \Gamma\left(k = \frac{n-1}{2}, \theta = \frac{2\sigma^2}{n-1}\right).
$$

The chi-square is a special case of the gamma, so the sample variance belongs to the wider gamma family.

### B. The $t$ Distribution and the $\sqrt{n}$ Factor

The sample mean and population mean satisfy

$$
\bar{X} = \frac{1}{n} \sum_i X_i, \qquad \operatorname{Var}(\bar{X}) = \frac{\sigma^2}{n}.
$$

When $\sigma$ is known,

$$
Z = \frac{\bar{X} - \mu}{\sigma / \sqrt{n}} \sim \mathcal{N}(0,1),
$$

showing that $\sqrt{n}$ arises from the reduction in variance due to averaging.

If $\sigma$ is unknown, replace it with $S$:

$$
T = \frac{\bar{X} - \mu}{S / \sqrt{n}} \sim t_{n-1}.
$$

Because the denominator involves $S$ (derived from a chi-square), the ratio follows a $t$ distribution.

| Scenario | Statistic | Distribution | $\sqrt{n}$ source |
|:---------|:---------|:-------------|:------------------|
| **$\sigma$ known** | $\frac{\bar{X} - \mu}{\sigma / \sqrt{n}}$ | Normal | $\operatorname{Var}(\bar{X}) = \sigma^2 / n$ |
| **$\sigma$ unknown** | $\frac{\bar{X} - \mu}{S / \sqrt{n}}$ | $t$ | Same variance scaling, replacing $\sigma$ with $S$ |

---

In summary:

1. The sample variance is a scaled chi-square (gamma) variable, not chi-square itself.
2. The $\sqrt{n}$ in the $t$ statistic stems from the variance of the sample mean.

---

## Appendix B: Expectation of the Chi-Square, $t$, and $F$ Distributions

This appendix records core expectation properties for the three classic distributions that arise from the normal family.

### 1. Expectation of the Chi-Square Distribution

Let $X_i \sim \mathcal{N}(0,1)$, independent, and define $\chi_n^2 = \sum_{i=1}^{n} X_i^2$. Then

$$
\mathbb{E}[X_i^2] = 1,\quad \mathbb{E}[\chi_n^2] = n,\quad \operatorname{Var}(\chi_n^2) = 2n.
$$

Thus the expectation of a chi-square random variable equals its degrees of freedom.

### 2. Expectation of the $t$ Distribution

Let

$$
T = \frac{Z}{\sqrt{V / n}},
$$

where $Z \sim \mathcal{N}(0,1)$, $V \sim \chi_n^2$, and $Z, V$ are independent. Then $\mathbb{E}[T]$ exists and equals $0$ when $n > 1$. The variance is

$$
\operatorname{Var}(T) = \frac{n}{n-2}, \quad n > 2.
$$

For $n = 1$, the expectation does not exist.

### 3. Expectation of the $F$ Distribution

Let

$$
F = \frac{(X_1^2 / m)}{(X_2^2 / n)},
$$

with $X_1^2 \sim \chi_m^2$, $X_2^2 \sim \chi_n^2$, independent. Then

$$
\mathbb{E}[F] = \frac{n}{n-2}, \quad n > 2.
$$

### Summary

| Distribution | Definition | Expectation | Condition |
|:-------------|:-----------|:------------|:----------|
| **$\chi_n^2$** | $\sum X_i^2$ | $n$ | Always |
| **$t_n$** | $Z / \sqrt{V / n}$ | $0$ | $n > 1$ |
| **$F_{m,n}$** | $(X_1^2 / m) / (X_2^2 / n)$ | $n / (n - 2)$ | $n > 2$ |



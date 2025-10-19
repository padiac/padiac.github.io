## 1. Discrete Additive Families ("Three Classic Brothers")

The multinomial, binomial, and Poisson distributions form a discrete hierarchy where additivity is preserved.

| Distribution | Parameters | Additive rule | Interpretation |
|--------------|------------|---------------|----------------|
| Multinomial | $(n; p_1,\dots,p_k)$ | Merge categories $\Rightarrow$ add probabilities | Multi-category trials |
| Binomial | $(n, p)$ | Add trials $n$ | Success or failure experiments |
| Poisson | $\lambda$ | Add rates $\lambda$ | Limit of rare events |

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
\ell(y) = \int_{-\infty}^{\infty} f_1(x_1) f_2(y - x_1)\,dx_1,
$$

which defines the convolution of $f_1$ and $f_2$.

### 2.3 Normal Distribution Closure

If $X_1 \sim \mathcal{N}(\mu_1, \sigma_1^2)$ and $X_2 \sim \mathcal{N}(\mu_2, \sigma_2^2)$ are independent, then

$$
Y = X_1 + X_2 \sim \mathcal{N}(\mu_1 + \mu_2, \sigma_1^2 + \sigma_2^2).
$$

Normal distributions are closed under addition, a property sometimes nicknamed the "earthworm theorem."

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

| Distribution | Constructed from | Expression | Degrees of freedom |
|--------------|------------------|------------|--------------------|
| $\chi^2_n$ | $\sum \mathcal{N}(0,1)^2$ | Square-sum | $n$ |
| $t_n$ | $\mathcal{N}(0,1) / \sqrt{\chi^2_n / n}$ | Ratio (mean over std estimate) | $n$ |
| $F_{n,m}$ | $(\chi^2_n / n) / (\chi^2_m / m)$ | Ratio of variances | $(n,m)$ |

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

## 6. The Chi-Square–$t$–$F$ Chain

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

| Distribution | Construction (random variable form) | Degrees of freedom | Common use |
|--------------|-------------------------------------|--------------------|------------|
| $\chi^2$ | $\sum \left(\frac{X_i - \mu}{\sigma}\right)^2$ or $\frac{(n-1) S^2}{\sigma^2}$ | $n$ or $n-1$ | Variance estimation and testing |
| $t$ | $\frac{\bar{X} - \mu}{S / \sqrt{n}}$ | $n-1$ | Mean testing and confidence intervals |
| $F$ | $\frac{S_1^2 / \sigma_1^2}{S_2^2 / \sigma_2^2}$ | $(n-1, m-1)$ | Variance ratio tests, ANOVA |

Conceptual chain:

Normal $\rightarrow$ square and sum $\rightarrow$ Chi-square $\rightarrow$ divide by $\sqrt{\chi^2 / \text{df}}$ $\rightarrow$ $t$

Two $(\chi^2 / \text{df})$ ratios $\rightarrow$ $F$

---

## 8. Summary: The Continuous Family Hierarchy

| Family | Generated from | Key form | Characteristic |
|--------|----------------|----------|----------------|
| Normal | Base | $\mathcal{N}(\mu,\sigma^2)$ | Symmetric bell shape |
| Chi-square | Sum of squares | $\sum X_i^2$ | Positive-only variance measure |
| $t$ | Normal over $\sqrt{\chi^2 / n}$ | Ratio | Standardized mean deviation |
| $F$ | $(\chi^2_1 / n_1) / (\chi^2_2 / n_2)$ | Ratio | Variance ratio |

---

**Conceptual chain:** Normal $\rightarrow$ Chi-square $\rightarrow$ ($t$, $F$). Each stage arises from combining or scaling the previous one by summing squares, taking ratios, and normalizing variances, revealing how classical inferential statistics stem from the properties of the normal distribution.

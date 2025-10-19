# Probability - Distribution Combination and Transformation

## 1. Discrete Case: Three Classic Brothers

This section summarizes three classical examples that demonstrate the additive property of distributions: when independent random variables from the same family are summed, the result remains in the same family with parameters added.

---

### 1.1 Multinomial Distribution

**Setup**  
Consider $n$ independent trials, each having $k$ possible outcomes (events $A_1, A_2, \dots, A_k$) with probabilities $p_1, p_2, \dots, p_k$ satisfying $\sum_{i=1}^k p_i = 1$.

**Result**  
Let $X_i$ be the number of occurrences of event $A_i$. Then $(X_1, X_2, \dots, X_k) \sim \mathrm{Mult}(n; p_1, p_2, \dots, p_k)$.

**Additivity**  
If we define $Y = X_1 + X_2$, then $Y \sim \mathrm{Bin}(n, p_1 + p_2)$.

**Meaning**  
The multinomial distribution is a multidimensional generalization of the binomial; combining categories corresponds to adding their probabilities.

---

### 1.2 Binomial Distribution

**Setup**  
Let $X_1 \sim \mathrm{Bin}(n_1, p)$ and $X_2 \sim \mathrm{Bin}(n_2, p)$, independent.

**Goal**  
Find $Y = X_1 + X_2$.

**Derivation**  
Using independence and the identity $\sum_{k_1+k_2=k} \binom{n_1}{k_1} \binom{n_2}{k_2} = \binom{n_1 + n_2}{k}$, we obtain $Y \sim \mathrm{Bin}(n_1 + n_2, p)$.

**Meaning**  
Two independent binomial variables with the same success probability $p$ add up to a new binomial variable whose trial count is the sum of the two.

---

### 1.3 Poisson Distribution

**Setup**  
Let $X_1 \sim \mathrm{Pois}(\lambda_1)$ and $X_2 \sim \mathrm{Pois}(\lambda_2)$, independent.

**Goal**  
Find $Y = X_1 + X_2$.

**Result**  
$Y \sim \mathrm{Pois}(\lambda_1 + \lambda_2)$.

**Reasoning**  
Expanding the joint probability and using $\sum_{k_1+k_2=k} \frac{k!}{k_1!k_2!} \lambda_1^{k_1} \lambda_2^{k_2} = (\lambda_1 + \lambda_2)^k$ leads to the conclusion.

**Meaning**  
The Poisson distribution is the limiting case of the binomial as $n \to \infty$, $p \to 0$ with $np = \lambda$. Its additivity mirrors that of the binomial: intensities add.

---

### 1.4 Summary Table

| Distribution | Parameters | Additive rule | Interpretation |
|--------------|------------|---------------|----------------|
| Multinomial | $(n; p_1, \dots, p_k)$ | Combine categories $\Rightarrow$ add probabilities | Multi-category experiment |
| Binomial | $(n, p)$ | Add trial counts $n$ | Success or failure trials |
| Poisson | $\lambda$ | Add rates $\lambda$ | Rare-events limit model |

Logical hierarchy: multinomial $\Rightarrow$ binomial ($k=2$) $\Rightarrow$ Poisson (limit).

---

## 2. Continuous Case: Transformation and Additivity

This section discusses how continuous random variables transform and combine through integrals. Two key concepts are the Jacobian transformation and the sum distribution (convolution).

---

### 2.1 Jacobian Transformation

**Idea**  
When transforming variables $(X_1, \dots, X_n) \mapsto (Y_1, \dots, Y_n)$, probability density must be adjusted by a volume-scaling factor, the Jacobian determinant.

If $Y_i = g_i(X_1, \dots, X_n)$ and the inverse transform exists with $X_i = h_i(Y_1, \dots, Y_n)$, then the joint density of $Y$ is

$$
f_Y(y_1, \dots, y_n) = f_X\bigl(h_1(y), \dots, h_n(y)\bigr) \left|\det \frac{\partial h}{\partial y}\right|.
$$

**Meaning**  
This preserves total probability under variable transformation; it is the continuous analog of the change-of-variables formula in integration.

---

### 2.2 Sum of Continuous Random Variables

**Setup**  
Given joint density $f(x_1, x_2)$, define $Y = X_1 + X_2$.

**Goal**  
Find the density $l(y)$ of $Y$.

**Procedure**
1. Express $P(Y \le y) = \iint_{x_1+x_2 \le y} f(x_1, x_2)\,dx_1 dx_2$.
2. Differentiate with respect to $y$ to obtain
$$
l(y) = \int_{-\infty}^{\infty} f(x_1, y - x_1)\,dx_1.
$$

If $X_1$ and $X_2$ are independent, then $f(x_1, x_2) = f_1(x_1) f_2(x_2)$ and

$$
l(y) = \int_{-\infty}^{\infty} f_1(x_1) f_2(y - x_1)\,dx_1,
$$

which is the convolution of $f_1$ and $f_2$.

---

### 2.3 The Closure of the Normal Distribution

**Setup**  
Let $X_1 \sim \mathcal{N}(\mu_1, \sigma_1^2)$ and $X_2 \sim \mathcal{N}(\mu_2, \sigma_2^2)$, independent. Find $Y = X_1 + X_2$.

**Computation**  
Using the convolution formula above, the result simplifies to $Y \sim \mathcal{N}(\mu_1 + \mu_2, \sigma_1^2 + \sigma_2^2)$.

**Meaning**  
The family of normal distributions is closed under addition: the sum of independent normals remains normal, with mean and variance both additive. This property is sometimes described as the earthworm theorem—bell-shaped curves stretch but remain bell-shaped under convolution.

---

### 2.4 Summary Table

| Concept | Formula or property | Interpretation |
|---------|---------------------|----------------|
| Variable transformation | $f_Y = f_X \times |\det(\partial h / \partial y)|$ | Density scaling by coordinate change |
| Sum of variables | $l(y) = \int f_1(x) f_2(y - x)\,dx$ | Convolution integral |
| Normal closure | $\mathcal{N}(\mu_1 + \mu_2, \sigma_1^2 + \sigma_2^2)$ | Gaussian stability |

---

## 3. Overall Logical Chain

1. **Discrete case:** sums correspond to combinatorial convolution, and parameters add.
2. **Continuous case:** sums correspond to integral convolution, and densities convolve.
3. **Gaussian case:** convolution of normals remains normal, forming a stable family.
4. **Jacobian factor:** ensures transformations of variables preserve total probability.

Together, these show how probability distributions transform, combine, and retain structure across discrete and continuous domains.

---

**Summary thought**  
Additivity and transformation are twin pillars of probability structure. They explain why binomial, Poisson, and normal distributions are fundamental: they describe outcomes and remain self-consistent under aggregation and change of variables.


This note reviews the expectation (and whenever informative, the variance) of three distributions that are built from sums or ratios of squared normal variables: chi-square, Student's $t$, and the $F$ distribution. For each we record a concise derivation so the formulas are logically traceable.

---

## 1. Chi-Square Distribution

Let

$$
X_i \sim \mathcal{N}(0,1), \qquad i = 1,\dots,n, \qquad \chi_n^2 = \sum_{i=1}^{n} X_i^2.
$$

The variable $\chi_n^2$ has the Gamma density

$$
f_{\chi_n^2}(x) = \frac{1}{2^{n/2}\,\Gamma(n/2)} x^{\frac{n}{2}-1} e^{-x/2}, \qquad x>0,
$$

so it is $\mathrm{Gamma}(k=n/2,\;\theta=2)$. Using the Gamma moments

$$
\mathbb{E}[X] = k\theta, \qquad \operatorname{Var}(X) = k\theta^2,
$$

we obtain

$$
\mathbb{E}[\chi_n^2] = n, \qquad \operatorname{Var}(\chi_n^2) = 2n.
$$

More generally, $\mathbb{E}[(\chi_n^2)^r] = 2^r\,\Gamma(n/2 + r)/\Gamma(n/2)$ whenever $r > -n/2$.

---

## 2. Student's $t$ Distribution

A Student $t$ variable with $n$ degrees of freedom can be written as

$$
T = \frac{Z}{\sqrt{V/n}},
$$

where $Z \sim \mathcal{N}(0,1)$, $V \sim \chi_n^2$, and $Z$ and $V$ are independent. Its density is

$$
f_T(t) = \frac{\Gamma((n+1)/2)}{\sqrt{n\pi}\,\Gamma(n/2)} \left(1 + \frac{t^2}{n}\right)^{-(n+1)/2}.
$$

Because $f_T$ is an even function, $\mathbb{E}[T] = 0$ whenever the integral converges. The tails behave like $|t|^{-(n+1)}$, so the first absolute moment exists if and only if $n>1$. Hence

$$
\mathbb{E}[T] = \begin{cases}
0, & n>1, \\\
\text{undefined}, & n \le 1.
\end{cases}
$$

The variance exists for $n>2$ and equals

$$
\operatorname{Var}(T) = \frac{n}{n-2}, \qquad n>2.
$$

Higher moments follow $\mathbb{E}[|T|^k] < \infty$ if and only if $n>k$.

---

## 3. $F$ Distribution

Let

$$
F = \frac{(X_1^2 / m)}{(X_2^2 / n)},
$$

where $X_1^2 \sim \chi_m^2$, $X_2^2 \sim \chi_n^2$, and the two chi-squares are independent. Writing $U = X_1^2$ and $V = X_2^2$, note that

$$
\frac{n}{m}F = \frac{U}{m} \cdot \frac{n}{V}.
$$

Since $U$ and $V$ are independent, so are $U$ and $1/V$. Using the Gamma moment identity $\mathbb{E}[X^{-1}] = 1/(k\theta)$ for $X \sim \mathrm{Gamma}(k,\theta)$ with $k>1$, we have $\mathbb{E}[1/V] = 1/(n-2)$ for $n>2$. Therefore

$$
\mathbb{E}[F] = \frac{n}{m} \mathbb{E}[U] \mathbb{E}\!\left[\frac{1}{V}\right] = \frac{n}{m} \cdot m \cdot \frac{1}{n-2} = \frac{n}{n-2}, \qquad n>2.
$$

The mean is undefined when $n \le 2$. Variance exists for $n>4$ and is

$$
\operatorname{Var}(F) = \frac{2n^2(m+n-2)}{m(n-2)^2(n-4)}.
$$

---

## Summary Table

| Distribution | Representation | Mean | Variance | Notes |
|:-------------|:---------------|:-----|:---------|:------|
| **$\chi_n^2$** | $\sum X_i^2$ | $n$ | $2n$ | All moments finite (Gamma family). |
| **$t_n$** | $Z / \sqrt{V / n}$ | $0$ (requires $n>1$) | $\dfrac{n}{n-2}$ (requires $n>2$) | $\mathbb{E}[|T|^k] < \infty$ iff $n>k$. |
| **$F_{m,n}$** | $(X_1^2/m)/(X_2^2/n)$ | $\dfrac{n}{n-2}$ (requires $n>2$) | $\dfrac{2n^2(m+n-2)}{m(n-2)^2(n-4)}$ (requires $n>4$) | Mean undefined for $n \le 2$; higher moments demand larger $n$. |

---

**Key ideas:**

1. Chi-square moments follow immediately from the Gamma interpretation.
2. A $t_n$ mean exists only when $n>1$; the tails are too heavy otherwise.
3. The $F_{m,n}$ mean (and higher moments) depend on the denominator degrees of freedom; randomness in the denominator pushes the mean above $1$ whenever it exists.
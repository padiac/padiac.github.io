## 1. Definition

Given random variables X and Y with finite variances.

```
μ_X = E[X],   μ_Y = E[Y]
σ_X² = Var(X), σ_Y² = Var(Y)

Covariance:      Cov(X, Y) = E[(X - μ_X)(Y - μ_Y)]
Uncorrelated:    Cov(X, Y) = 0
Correlation:     Corr(X, Y) = Cov(X, Y) / (σ_X σ_Y) ∈ [-1, 1]
```

"Uncorrelated" means the linear dependence between X and Y is zero. It does **not** imply full independence unless additional distributional structure is given.

---

## 2. Fundamental Inequalities

```
|Cov(X, Y)| ≤ σ_X σ_Y       (Cauchy–Schwarz Inequality)
|Corr(X, Y)| ≤ 1
```

Equality holds if and only if there exist constants `a ≠ 0` and `b` such that `Y = aX + b` almost surely, meaning the variables are perfectly linearly related.

---

## 3. Uncorrelated vs Independent

| Relationship | Meaning | Implication |
|--------------|---------|-------------|
| Independent | `f(x, y) = f_X(x) f_Y(y)` | ⇒ Uncorrelated |
| Uncorrelated | `Cov(X, Y) = 0` | ⇏ Independent (in general) |

"Uncorrelated" only rules out linear dependence; nonlinear dependence can remain.

---

## 4. Counterexamples

**Example A (Uniform on a circle).** Let `(X, Y)` be uniformly distributed on a unit disk.

```
E[X] = E[Y] = 0
Cov(X, Y) = E[XY] = 0  (symmetry cancels)
```

However, X and Y are not independent because the value of X constrains the possible values of Y.

**Example B (Nonlinear dependence).** Let `X ~ Uniform(-1, 1)` and define `Y = X²`.

```
E[X] = 0,  E[X³] = 0
Cov(X, Y) = E[X · X²] - E[X]E[Y] = 0
```

Y depends entirely on X, so the variables are dependent despite being uncorrelated.

---

## 5. Correlation Interpretation

- `Corr(X, Y) > 0`: Y tends to increase with X (upward trend).
- `Corr(X, Y) < 0`: Y tends to decrease with X (downward trend).
- `Corr(X, Y) = 0`: No linear trend, though nonlinear structure can remain.

Graphically:

```
|Corr| = 1 → all points lie on a line
|Corr| ≈ 0 → cloud-shaped scatter, no line
```

---

## 6. Bivariate Normal Exception

If `(X, Y)` follows a bivariate normal distribution,

```
(X, Y) ~ N₂(μ₁, μ₂, σ₁², σ₂², ρ)
```

then

```
Cov(X, Y) = 0  ⇔  X and Y are independent
```

When `ρ = 0`, the joint density factorizes as `f(x, y) = f_X(x) f_Y(y)`. This independence-from-uncorrelatedness equivalence is unique to the multivariate normal family.

---

## 7. Summary

```
Independence  ⇒  Uncorrelated
Uncorrelated  ⇏  Independence  (except in Gaussian case)

|Cov(X, Y)| ≤ σ_X σ_Y
|Corr(X, Y)| ≤ 1
|Corr| = 1 ⇔ Y = aX + b
```

---

## 8. Supplement: Why PCA Uses the Sample Covariance Matrix

Principal Component Analysis (PCA) finds the directions along which variance is maximized.

### 8.1 Theoretical covariance

```
Σ = E[(X - μ)(X - μ)ᵀ]
```

In theory, PCA requires the true covariance matrix Σ of the underlying distribution.

### 8.2 Practical estimation

Given samples `x₁, x₂, …, xₙ`, use the sample covariance matrix

```
S = (1/(n-1)) Σ (xᵢ - μ̂)(xᵢ - μ̂)ᵀ,   μ̂ = (1/n) Σ xᵢ
```

### 8.3 Statistical justification

The sample covariance is an unbiased estimator of the true covariance.

```
E[S] = Σ
```

This mirrors the unbiasedness of the sample mean and variance. Geometrically, `S` captures the data cloud, so its eigenvectors surface the sample’s principal directions.


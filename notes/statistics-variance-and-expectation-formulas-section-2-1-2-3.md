## 1. Definition of Variance

For a random variable X with mean E(X), the variance measures the dispersion of X around its mean:

```
Var(X) = E[(X - E(X))^2] = E(X^2) - [E(X)]^2
```

This is the most convenient computational form, obtained by expanding (X - E(X))^2 and applying the linearity of expectation.

---

## 2. Properties of Variance (Theorem 2.1)

Let C be a constant and X a random variable. Then:

1. Var(X + C) = Var(X)
2. Var(CX) = C^2 Var(X)

**Proof Sketch:**

- Adding a constant shifts all values equally, so dispersion does not change.
- Multiplying by C scales all deviations by C, hence variance scales by C^2.

---

## 3. Variance of the Sum of Independent Random Variables (Theorem 2.2)

If X_1, X_2, ..., X_n are independent, then

```
Var(X_1 + X_2 + ... + X_n) = Var(X_1) + Var(X_2) + ... + Var(X_n)
```

This follows because for independent variables, E(X_i X_j) = E(X_i)E(X_j) when i <> j, so the cross-terms in the expansion of (sum X_i - sum E(X_i))^2 vanish.

---

## 4. Example 2.1 - Standardization

If X is a random variable with mean E(X) = a and variance Var(X) = sigma^2, define

```
Y = (X - a) / sigma
```

Then E(Y) = 0 and Var(Y) = 1.

This process converts any variable into a standardized form with zero mean and unit variance.

---

## 5. Example 2.2 - Poisson Distribution

Let X ~ P(lambda). Then

```
E(X) = lambda
Var(X) = lambda
```

**Proof Sketch:**

Using E(X^2) = sum i^2 e^(-lambda) lambda^i / i!, rewrite i^2 = i(i - 1) + i and evaluate separately:

```
E(X^2) = lambda^2 + lambda
Var(X) = E(X^2) - [E(X)]^2 = lambda
```

Thus for the Poisson distribution, the mean and variance are equal to the same parameter lambda.

---

## 6. Example 2.3 - Binomial Distribution

Let X ~ B(n, p). Then

```
E(X) = n p
Var(X) = n p (1 - p)
```

**Derivation:**

Write X = X_1 + X_2 + ... + X_n, where each X_i ~ B(1, p) is Bernoulli and independent.

Then by Theorem 2.2:

```
Var(X) = sum Var(X_i) = n Var(X_1) = n p (1 - p)
```

---

## Summary Table

| Distribution | Mean E(X) | Variance Var(X) |
|--------------|-----------|-----------------|
| General | E(X) | E(X^2) - [E(X)]^2 |
| Poisson P(lambda) | lambda | lambda |
| Binomial B(n, p) | n p | n p (1 - p) |

---

## Conceptual Summary

- Variance quantifies dispersion (the average squared deviation from the mean).
- It is invariant under translation and scales quadratically with constants.
- For independent variables, variances add but do not interact.
- Poisson and Binomial distributions are classic discrete examples where mean and variance have elegant relationships to their parameters.

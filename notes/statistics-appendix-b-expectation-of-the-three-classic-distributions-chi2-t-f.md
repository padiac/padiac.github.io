This appendix summarizes the derivation and expectation properties of the three fundamental distributions in statistics ¡ª the chi-square, t, and F distributions.

---

## 1. Expectation of the Chi-Square Distribution

Let

```
X_i ~ N(0,1),  i = 1, ..., n
¦Ö_n^2 = ¦²_{i=1}^{n} X_i^2
```

Then

```
E[X_i^2] = 1
E[¦Ö_n^2] = n
Var(¦Ö_n^2) = 2n
```

**Therefore:** the expectation of a chi-square distribution equals its degrees of freedom.

---

## 2. Expectation of the t Distribution

Definition:

```
T = Z / ¡Ì(V / n)
```

where

```
Z ~ N(0,1)
V ~ ¦Ö_n^2
Z ? V
```

- When n = 1, the integral diverges, so E[T] does not exist.
- When n > 1, symmetry implies E[T] = 0.
- Variance exists only when n > 2, with Var(T) = n / (n ? 2).

---

## 3. Expectation of the F Distribution

Definition:

```
F = (X_1^2 / m) / (X_2^2 / n)
```

where X_1^2 ~ ¦Ö_m^2, X_2^2 ~ ¦Ö_n^2, and they are independent. Using

```
E[X_1^2] = m,    E[1 / X_2^2] = 1 / (n ? 2)
```

(valid for n > 2), we obtain

```
E[F] = n / (n ? 2).
```

---

## Summary Table

| Distribution | Definition | Expectation | Condition |
|:-------------|:-----------|:------------|:----------|
| **¦Ö_n^2** | ¦² X_i^2 | n | Always |
| **t_n** | Z / ¡Ì(V / n) | 0 | n > 1 |
| **F_{m,n}** | (X_1^2 / m) / (X_2^2 / n) | n / (n ? 2) | n > 2 |

---

**Insight:** the chi-square expectation equals its degrees of freedom, the t distribution stays centered at zero (once n > 1), and the F distribution¡¯s mean exceeds one due to the stochastic denominator.

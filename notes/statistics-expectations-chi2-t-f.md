This note summarizes the expectation and variance properties of three foundational distributions derived from the normal family: chi-square, Student¡¯s t, and F.

---

## 1. Chi-Square Distribution

Let

```
X_i ~ N(0, 1),  i = 1, ..., n
¦Ö_n^2 = ¦²_{i=1}^n X_i^2
```

Key moments:

```
E[X_i^2] = 1
E[¦Ö_n^2] = n
Var(¦Ö_n^2) = 2n
```

> The mean of a chi-square variable equals its degrees of freedom.

---

## 2. Student¡¯s t Distribution

Define

```
T = Z / ¡Ì(V / n)
```

with

```
Z ~ N(0, 1)
V ~ ¦Ö_n^2
Z ? V
```

- When n = 1, the expectation does not exist (integral diverges).
- When n > 1, symmetry yields E[T] = 0.
- The variance exists only if n > 2 and equals n / (n ? 2).

---

## 3. F Distribution

Let

```
F = (X_1^2 / m) / (X_2^2 / n)
```

where X_1^2 ~ ¦Ö_m^2 and X_2^2 ~ ¦Ö_n^2 are independent. Using

```
E[X_1^2] = m
E[1 / X_2^2] = 1 / (n ? 2)
```

(valid for n > 2), we get

```
E[F] = n / (n ? 2).
```

---

## Summary Table

| Distribution | Definition | Expected value | Existence condition |
|:-------------|:-----------|:---------------|:--------------------|
| **¦Ö_n^2** | ¦² X_i^2 | n | Always |
| **t_n** | Z / ¡Ì(V / n) | 0 | n > 1 |
| **F_{m,n}** | (X_1^2 / m) / (X_2^2 / n) | n / (n ? 2) | n > 2 |

---

**Takeaway:** chi-square means track degrees of freedom, the t distribution stays centered at zero once n exceeds one, and the F distribution¡¯s mean exceeds one because the denominator is itself random.

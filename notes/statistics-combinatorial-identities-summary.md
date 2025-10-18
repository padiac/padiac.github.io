# Combinatorial Identities Summary

## 1. Generalized Binomial Coefficient for Non-Positive Integers

For negative integers, the binomial coefficient is defined as:

$$
\binom{-1}{r} = \frac{(-1)(-2)\cdots(-r)}{r!} = (-1)^r
$$

This extends the definition of combinations beyond positive integers.

---

## 2. Common Identities from the Binomial Theorem

From the expansion:
$$
(a + b)^n = \sum_{i=0}^{n} \binom{n}{i} a^i b^{n-i}
$$

Two frequently used results are:

1. When $a = b = 1$:
   $$
   \binom{n}{0} + \binom{n}{1} + \cdots + \binom{n}{n} = 2^n
   $$

2. When $a = 1, b = -1$:
   $$
   \binom{n}{0} - \binom{n}{1} + \binom{n}{2} - \cdots + (-1)^n \binom{n}{n} = 0
   $$

---

## 3. Convolution Identity of Binomial Coefficients

$$
\boxed{\binom{m+n}{r} = \sum_{i=0}^{r} \binom{m}{i} \binom{n}{r-i}}
$$

This is derived by expanding $(1 + x)^{m+n} = (1 + x)^m (1 + x)^n$ and comparing coefficients of $x^r$.

---

## 4. Multinomial Coefficient (Distribution Formula)

When dividing $n$ distinct items into $k$ groups of sizes $r_1, r_2, \ldots, r_k$:

$$
\boxed{\frac{n!}{r_1! r_2! \cdots r_k!}}
$$

This represents the number of distinct distributions.

---

## 5. Hockey-Stick Identity (斜锯式恒等式)

$$
\boxed{1 + \binom{n}{1} + \binom{n+1}{2} + \cdots + \binom{n+m-1}{m} = \binom{n+m}{m}}
$$

If you omit the first term ($\binom{n-1}{0}=1$), then:

$$
\binom{n}{1} + \binom{n+1}{2} + \cdots + \binom{n+m-1}{m} = \binom{n+m}{m} - 1
$$

---

## 6. Pascal’s Identity (帕斯卡恒等式)

$$
\boxed{\binom{n}{r} = \binom{n-1}{r} + \binom{n-1}{r-1}}
$$

This is the recursive basis for the binomial triangle.

---

> ✅ **Summary Table**

| Category | Formula | Description |
|-----------|----------|-------------|
| Negative Integer Combination | $\binom{-1}{r} = (-1)^r$ | Extends binomial definition |
| Binomial Sum | $\sum \binom{n}{i} = 2^n$ | Basic binomial identity |
| Alternating Sum | $\sum (-1)^i \binom{n}{i} = 0$ | Sign-alternating version |
| Convolution | $\binom{m+n}{r} = \sum \binom{m}{i}\binom{n}{r-i}$ | Combination folding rule |
| Multinomial | $\frac{n!}{r_1! r_2! \cdots r_k!}$ | Partition of items |
| Hockey-Stick | $1+\binom{n}{1}+\cdots+\binom{n+m-1}{m}=\binom{n+m}{m}$ | Cumulative diagonal sum |
| Pascal’s Identity | $\binom{n}{r}=\binom{n-1}{r}+\binom{n-1}{r-1}$ | Recursive property |

---

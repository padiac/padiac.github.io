## 1. Generalized Binomial Coefficient for Non-Positive Integers

For negative integers, the binomial coefficient is defined as:

$$
\binom{-1}{r} = \frac{(-1)(-2)\cdots(-r)}{r!} = (-1)^r
$$

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
\boxed{\displaystyle \binom{m+n}{r} = \sum_{i=0}^{r} \binom{m}{i} \binom{n}{r-i}}
$$

**Derivation:**  
Expand $(1 + x)^{m+n} = (1 + x)^m (1 + x)^n$ and compare the coefficients of $x^r$.

---

## 4. Multinomial Coefficient (with short derivation)

When dividing $n$ distinct items into $k$ groups of sizes $r_1, r_2, \ldots, r_k$:

$$
\boxed{\displaystyle \frac{n!}{r_1! r_2! \cdots r_k!}}
$$

**Derivation:**  
Choose $r_1$ for group 1: $\binom{n}{r_1}$;  
then $r_2$ from the remaining $n - r_1$: $\binom{n - r_1}{r_2}$;  
continue similarly until group $k$.  
Multiplying all:
$$
\binom{n}{r_1} \binom{n - r_1}{r_2} \cdots = \frac{n!}{r_1! r_2! \cdots r_k!}.
$$

---

## 5. Hockey-Stick Identity



$$
\boxed{\binom{n-1}{0} + \binom{n}{1} + \binom{n+1}{2} + \cdots + \binom{n+m-1}{m} = \binom{n+m}{m}}
$$

**Proof (telescoping via Pascal):** Using $\binom{k}{r}=\binom{k-1}{r}+\binom{k-1}{r-1}$,
$$
\binom{n-1+i}{i}=\binom{n-2+i}{i}+\binom{n-2+i}{i-1}.
$$
Summing over $i$ along the diagonal cancels interior terms and leaves $\binom{n+m}{m}$.



---

## 6. Pascal’s Identity

$$
\boxed{\displaystyle \binom{n}{r} = \binom{n-1}{r} + \binom{n-1}{r-1}}
$$

**Derivation:**  
From $n$ labeled items, choose $r$.  
Fix one special item $x$.  
Either the chosen set **excludes** $x$: $\binom{n-1}{r}$;  
or it **includes** $x$: then pick the remaining $r-1$ from the other $n-1$: $\binom{n-1}{r-1}$.  
Adding both cases gives the formula.

---

## Summary Table

| Attribute | Negative integer combination | Binomial sum | Alternating sum | Convolution | Multinomial | Hockey-stick | Pascal’s identity |
|-----------|-----------------------------|--------------|----------------|-------------|------------|--------------|------------------|
| Formula | $\displaystyle \binom{-1}{r} = (-1)^r$ | $\displaystyle \sum \binom{n}{i} = 2^n$ | $\displaystyle \sum (-1)^i \binom{n}{i} = 0$ | $\displaystyle \binom{m+n}{r} = \sum \binom{m}{i}\binom{n}{r-i}$ | $\displaystyle \frac{n!}{r_1! r_2! \cdots r_k!}$ | $\displaystyle \binom{n-1}{1}+\binom{n}{2}+\cdots+\binom{n+m-2}{m}=\binom{n+m-1}{m+1}$ | $\displaystyle \binom{n}{r}=\binom{n-1}{r}+\binom{n-1}{r-1}$ |
| Description | Extension to negative integers | Basic binomial identity | Alternating sign version | Binomial folding rule | Number of distributions | Cumulative diagonal sum | Recursive relation |

---

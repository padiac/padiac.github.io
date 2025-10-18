# Discrete Distributions - Quick Notes

Below: pmf, support, intuition, and very short derivations.  
(Geometric is the special case of Negative Binomial with $r=1$.)

---

## 1) Binomial $\mathrm{Bin}(n,p)$

**pmf**

$$
\Pr(X=i)=\binom{n}{i}p^{\,i}(1-p)^{n-i},\quad i=0,1,\dots,n.
$$

**story**: $n$ independent Bernoulli trials (success prob $p$); count successes.  
**one-line derivation**: choose the $i$ success positions $\binom{n}{i}$; multiply independent probabilities.

---

## 2) Poisson $\mathrm{Poi}(\lambda)$

**pmf**

$$
\Pr(X=i)=e^{-\lambda}\frac{\lambda^i}{i!},\quad i=0,1,2,\dots
$$

**story**: counts of rare, independent events over a fixed window with mean rate $\lambda$.  
**short proof (as Binomial limit)**: let $X_n\sim \mathrm{Bin}(n,\lambda/n)$.

$$
\Pr(X_n=i)=\binom{n}{i}\left(\frac{\lambda}{n}\right)^i
\left(1-\frac{\lambda}{n}\right)^{n-i}
\xrightarrow[n\to\infty]{}
\frac{\lambda^i}{i!}e^{-\lambda}.
$$

---

## 3) Hypergeometric $\mathrm{Hyp}(N,M,n)$

**pmf**

$$
\Pr(X=m)=\frac{\binom{M}{m}\binom{N-M}{\,n-m\,}}{\binom{N}{n}},
\quad m=\max(0,n-(N-M)),\dots,\min(n,M).
$$

**story**: sample $n$ without replacement from a population of size $N$ with $M$ "successes"; $X$ counts the sampled successes.  
**one-line derivation**: favorable draws over total draws.

---

## 4) Negative Binomial $\mathrm{NB}(r,p)$

<!-- *(number of failures before the $r$-th success; $r\in\mathbb{N}$)*

**pmf**

$$
\Pr(X=i)=\binom{i+r-1}{r-1}p^{\,r}(1-p)^{i},\quad i=0,1,2,\dots
$$

**story**: i.i.d. Bernoulli$(p)$ trials; stop when the $r$-th success occurs; $X$ counts failures.  
**one-line derivation**: last trial must be a success ($p$); among the first $i+r-1$ trials choose the $r-1$ earlier successes: $\binom{i+r-1}{r-1}$; multiply by $(1-p)^i p^{r-1}$. -->


---

### 4.1. Probability Mass Function

Let each Bernoulli trial have success probability $p$, and define $X$ as the number of failures before the $r$-th success.  
That is, the final trial is the $r$-th success.

To get $X=i$:

- The total number of trials is $i+r$.
- The last one must be a success (probability $p$).
- Among the first $i+r-1$ trials, choose $r-1$ successes.

$$
P(X=i) = \binom{i+r-1}{r-1}p^{r}(1-p)^{i}, \quad i=0,1,2,\dots
$$

---

### 4.2. Normalization via the Negative Binomial Expansion

We must verify that

$$
\sum_{i=0}^{\infty} \binom{i+r-1}{r-1}p^{r}(1-p)^{i}=1.
$$

Take $x=1-p$. Then we need

$$
\sum_{i=0}^{\infty}\binom{i+r-1}{r-1}x^i=(1-x)^{-r}.
$$


Thus the distribution is properly normalized.

---

### 4.3. Proof of the Expansion Using Taylor Series

Let $f(x)=(1-x)^{-r}$, where $r\in\mathbb{N}$.  
Since $f$ is analytic for $|x|<1$, it admits a Taylor expansion:

$$
f(x)=\sum_{k=0}^{\infty}\frac{f^{(k)}(0)}{k!}x^{k}.
$$

Compute derivatives:

$$
f^{(k)}(x)=r(r+1)\cdots(r+k-1)(1-x)^{-r-k},
$$

so

$$
f^{(k)}(0)=r(r+1)\cdots(r+k-1)=\frac{(r+k-1)!}{(r-1)!}.
$$

Hence

$$
(1-x)^{-r}=\sum_{k=0}^{\infty}\frac{(r+k-1)!}{(r-1)!k!}x^{k}
=\sum_{k=0}^{\infty}\binom{r+k-1}{k}x^{k}.
$$

---

### 4.4. Conversion from $\binom{-r}{i}$ to $\binom{r+i-1}{i}$

Starting from the generalized binomial definition:

$$
\binom{-r}{i}
=\frac{(-r)(-r-1)\cdots(-r-i+1)}{i!}
=(-1)^i\frac{(r+i-1)!}{(r-1)!i!}
=(-1)^i\binom{r+i-1}{i}.
$$

Then in the expansion:

$$
(1-x)^{-r}
=\sum_{i=0}^{\infty}\binom{-r}{i}(-x)^i
=\sum_{i=0}^{\infty}\binom{r+i-1}{i}x^i,
$$

since $(-1)^i\cdot(-1)^i=1$.

This completes the derivation of the negative binomial form and connects its coefficients directly with the combinatorial interpretation.

---

## 5) Geometric $\mathrm{Geo}(p)$ (special case $r=1$)

**pmf**

$$
\Pr(X=i)=p(1-p)^{i},\quad i=0,1,2,\dots
$$

**story**: number of failures before the first success.  
**one-line derivation**: "$i$ failures then 1 success": $(1-p)^i p$.

---

## Quick Relationships

- $\mathrm{Poi}(\lambda)$ is the limit of $\mathrm{Bin}(n,\lambda/n)$ (rare-event regime).
- $\mathrm{Geo}(p) = \mathrm{NB}(r=1,p)$.
- $\mathrm{Hyp}(N,M,n) \to \mathrm{Bin}(n,\,p)$ when the sampling fraction is tiny: $p \approx M/N$, $n\ll N$.

---


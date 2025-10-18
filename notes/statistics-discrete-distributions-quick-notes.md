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

*(number of failures before the $r$-th success; $r\in\mathbb{N}$)*

**pmf**

$$
\Pr(X=i)=\binom{i+r-1}{r-1}p^{\,r}(1-p)^{i},\quad i=0,1,2,\dots
$$

**story**: i.i.d. Bernoulli$(p)$ trials; stop when the $r$-th success occurs; $X$ counts failures.  
**one-line derivation**: last trial must be a success ($p$); among the first $i+r-1$ trials choose the $r-1$ earlier successes: $\binom{i+r-1}{r-1}$; multiply by $(1-p)^i p^{r-1}$.

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


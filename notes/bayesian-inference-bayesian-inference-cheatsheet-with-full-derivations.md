**Scope.** This note consolidates our discussion about Bayesian inference in the normal model, Jeffreys priors, scale versus parameterization invariance, chi-square, t, and F connections, and conjugate priors. It emphasizes detailed derivations with display math.

---

## 0) Probability Identities Behind "Integrate Out"

### Chain rule and marginalization (law of total probability)

For random variables $A,B$ with a joint density $p(A,B)$, the **conditional density** is defined by

$$
p(A\mid B)=\frac{p(A,B)}{p(B)} \quad \Longrightarrow \quad p(A,B)=p(A\mid B) p(B).
$$

The **marginal density** of $A$ is obtained by integrating the joint density over $B$:

$$
p(A)=\int p(A,B) dB.
$$

Combining the two gives the continuous **law of total probability**:

$$
p(A)=\int p(A\mid B) p(B) dB.
$$

**Bayesian version (conditioning on observed data $X$):**

$$
p(\theta\mid X)=\int p(\theta,\phi\mid X) d\phi = \int p(\theta\mid X,\phi) p(\phi\mid X) d\phi.
$$

This is exactly what we do when we "integrate out" a nuisance parameter $\phi$ (for example $\sigma^2$) to obtain the **marginal posterior** of the parameter of interest $\theta$.

---

## 1) Chi-Square, $n$ vs $n-1$, and the Role of $S$ vs $S^2$

### 1.1 Definition

If $Z_1,\dots,Z_\nu \overset{\mathrm{iid}}{\sim} N(0,1)$, then

$$
W=\sum_{i=1}^{\nu} Z_i^2 \sim \chi^2_{\nu},
$$

with pdf

$$
f_{\chi^2}(w;\nu)=\frac{1}{2^{\nu/2}\Gamma(\nu/2)} w^{\nu/2-1} \exp\left(-\frac{w}{2}\right), \qquad w>0.
$$

### 1.2 Known mean vs unknown mean

Let $X_i\overset{\mathrm{iid}}{\sim} N(\mu,\sigma^2)$.

- **Known mean $\mu$** (use the population mean): define

  $$
  \tilde S^2=\frac{1}{n}\sum_{i=1}^n (X_i-\mu)^2.
  $$

  Then

  $$
  \frac{n \tilde S^2}{\sigma^2}\sim \chi^2_n.
  $$

- **Unknown mean** (use the sample mean $\bar X$): define

  $$
  S^2=\frac{1}{n-1}\sum_{i=1}^n (X_i-\bar X)^2.
  $$

  Then

  $$
  \frac{(n-1) S^2}{\sigma^2}\sim \chi^2_{n-1}.
  $$

The loss of one degree of freedom ($n$ to $n-1$) arises from estimating $\mu$ by $\bar X$ and the identity

$$
\sum_{i=1}^n (X_i-\mu)^2=\sum_{i=1}^n (X_i-\bar X)^2 + n(\bar X-\mu)^2,
$$

which splits the total sum of squares into an $(n-1)$-df part and a $1$-df part.

### 1.3 Density for $S^2$ and the "missing $-1$"

From $\frac{n S^2}{\sigma^2}\sim \chi^2_n$, a change of variables gives the density of $S^2$ (treating $S^2$ as the variable):

$$
f_{S^2}(s^2\mid \sigma^2)=\frac{(n/2)^{n/2}}{\Gamma(n/2)} (\sigma^2)^{-n/2} (s^2)^{n/2-1} \exp\left(-\frac{n s^2}{2\sigma^2}\right).
$$

When forming a likelihood **only in $\sigma^2$**, factors independent of $\sigma^2$ (including $(s^2)^{n/2-1}$) are dropped; that is why the exponent "$-1$" appears missing in likelihood expressions.

### 1.4 Using $S$ instead of $S^2$

If one uses $S=\sqrt{S^2}$ as the variable, then the Jacobian factor $2s$ changes the power of $s$, yielding

$$
f_S(s)=\frac{2 n^{n/2}}{2^{n/2}\Gamma(n/2)} (\sigma^2)^{-n/2} s^{n-1} \exp\left(-\frac{n s^2}{2\sigma^2}\right), \qquad s>0.
$$

That extra factor is exactly the derivative $\frac{d}{ds}(s^2)=2s$.

---

## 2) Normal Model with Known $\sigma^2$: Posterior for $\theta$ under a Normal Prior

Assume $X_i\overset{\mathrm{iid}}{\sim} N(\theta,\sigma^2)$ with known $\sigma^2$. With a normal prior $\theta\sim N(\mu,\tau^2)$, the likelihood (based on the sufficient statistic $\bar X$) is

$$
\bar X\mid \theta \sim N\left(\theta,\frac{\sigma^2}{n}\right).
$$

The posterior is obtained by completing the square:

$$
\pi(\theta\mid \bar x)\propto \exp\left[-\frac{(\theta-\mu)^2}{2\tau^2}-\frac{(\bar x-\theta)^2}{2(\sigma^2/n)}\right] = \exp\left[-\frac{A}{2}\left(\theta-\frac{B}{A}\right)^2\right],
$$

where

$$
A=\frac{1}{\tau^2}+\frac{n}{\sigma^2}, \qquad B=\frac{\mu}{\tau^2}+\frac{n\bar x}{\sigma^2}.
$$

Thus

$$
\theta\mid \bar x \sim N\left(\mu_n,\eta_n^2\right), \qquad \mu_n=\frac{B}{A}=\frac{\mu/\tau^2+n\bar x/\sigma^2}{1/\tau^2+n/\sigma^2}, \qquad \eta_n^2=\frac{1}{A}=\frac{1}{1/\tau^2+n/\sigma^2}.
$$

A useful "weighted-average" form:

$$
\mu_n=\frac{(\sigma^2/n)\mu+\tau^2\bar x}{\sigma^2/n+\tau^2}, \qquad \eta_n^2=\frac{\sigma^2\tau^2}{n\tau^2+\sigma^2}.
$$

---

## 3) Unknown $\sigma^2$ (Mean Known): Jeffreys Prior and Inverse-Gamma Posterior

With mean $\theta$ **known** and variance $\sigma^2$ **unknown**, the likelihood based on $S^2=\frac{1}{n}\sum (X_i-\theta)^2$ is

$$
L(\sigma^2; s^2) \propto (\sigma^2)^{-n/2} \exp\left(-\frac{n s^2}{2\sigma^2}\right).
$$

### 3.1 Jeffreys prior for scale

For the variance parameter $\sigma^2$, the **Jeffreys prior** (equivalently, the scale prior) is

$$
\pi(\sigma^2)\propto \frac{1}{\sigma^2}, \qquad \sigma^2>0.
$$

(Equivalently in standard deviation, $\pi(\sigma)\propto 1/\sigma$. The two are consistent via change of variables.)

### 3.2 Posterior

$$
\pi(\sigma^2\mid s^2)\propto L(\sigma^2;s^2) \pi(\sigma^2)\propto (\sigma^2)^{-(n/2+1)} \exp\left(-\frac{n s^2}{2\sigma^2}\right),
$$

which is an **Inverse-Gamma** distribution:

$$
\sigma^2\mid s^2 \sim \text{Inv-Gamma}\left(\alpha=\frac{n}{2}, \beta=\frac{n s^2}{2}\right).
$$

Equivalently, $\sigma^2\mid s^2 \sim \chi^{-2}_{n}(s^2)$ (inverse-chi-square form).

---

## 4) Unknown $\theta$ and $\sigma^2$: Integrating Out $\sigma^2$ Yields a Student-$t$

Assume $X_i\overset{\mathrm{iid}}{\sim} N(\theta,\sigma^2)$ with both $\theta,\sigma^2$ unknown and Jeffreys joint prior

$$
\pi(\theta,\sigma^2)\propto \frac{1}{\sigma^2}.
$$

The joint posterior is

$$
p(\theta,\sigma^2\mid X)\propto (\sigma^2)^{-(n/2+1)} \exp\left[-\frac{1}{2\sigma^2}\sum_{i=1}^n (X_i-\theta)^2\right].
$$

Using the sum-of-squares decomposition

$$
\sum_{i=1}^n (X_i-\theta)^2=(n-1)S^2+n(\bar X-\theta)^2,
$$

we get

$$
p(\theta,\sigma^2\mid X)\propto (\sigma^2)^{-(n/2+1)} \exp\left[-\frac{(n-1)S^2+n(\bar X-\theta)^2}{2\sigma^2}\right].
$$

### 4.1 Marginal posterior of $\theta$ (integrating out $\sigma^2$)

Compute

$$
p(\theta\mid X)=\int_0^\infty p(\theta,\sigma^2\mid X) d\sigma^2 \propto \int_0^\infty (\sigma^2)^{-(n/2+1)} \exp\left(-\frac{A}{\sigma^2}\right) d\sigma^2,
$$

where

$$
A=\frac{(n-1)S^2+n(\bar X-\theta)^2}{2}.
$$

Substitute $u=1/\sigma^2$ to recognize a Gamma integral:

$$
\int_0^\infty u^{\frac{n}{2}-1} \exp(-A u) du=\Gamma\left(\frac{n}{2}\right) A^{-n/2}.
$$

Hence

$$
p(\theta\mid X)\propto A^{-n/2}=\left[(n-1)S^2+n(\bar X-\theta)^2\right]^{-n/2}\propto\left[1+\frac{n(\bar X-\theta)^2}{(n-1)S^2}\right]^{-n/2}.
$$

Let

$$
T=\frac{\bar X-\theta}{S/\sqrt{n}},
$$

then $T\sim t_{ n-1}$. Equivalently,

$$
\theta\mid X \sim t_{ n-1}\left(\bar X,\frac{S}{\sqrt{n}}\right).
$$

**Interpretation.** Mixing normals with different variances (due to uncertainty in $\sigma^2$) yields a heavier-tailed distribution—the Student-$t$.

---

## 5) Jeffreys Prior: Motivation and Invariances

### 5.1 Motivation

Naively "flat" priors are not invariant under reparameterization (uniform in $\theta$ is not uniform in $\phi=g(\theta)$). Jeffreys proposed a coordinate-free construction using Fisher information:

$$
\pi_J(\theta)\propto \sqrt{I(\theta)}, \qquad I(\theta)=E\left[-\frac{\partial^2}{\partial\theta^2}\log f(X\mid \theta)\right].
$$

This prior is **parameterization invariant**: for $\phi=g(\theta)$,

$$
\pi_J(\phi) d\phi=\pi_J(\theta) d\theta.
$$

### 5.2 Relation to location/scale "uninformative" priors

- **Location family** $f(x-\mu)$: $I(\mu)$ is constant $\Rightarrow \pi_J(\mu)\propto 1$.
- **Scale family** $f(x/\sigma)/\sigma$: $I(\sigma)\propto 1/\sigma^2 \Rightarrow \pi_J(\sigma)\propto 1/\sigma$.

Changing variable to $\tau=\sigma^2$ yields $\pi_J(\tau)\propto 1/\tau$. Thus for pure scale models, Jeffreys coincides with the classical scale-invariant prior.

### 5.3 Scale vs parameterization invariance

- **Scale invariance** demands invariance under unit changes $\sigma\to k\sigma$ (a physical symmetry).
- **Jeffreys invariance** demands invariance under reparameterization $\theta\to g(\theta)$ (a mathematical symmetry).

They coincide for pure scale/location families; in more complex models, Jeffreys preserves the latter, not necessarily the former.

---

## 6) Why Textbooks Switch from $\sigma$ to $\sigma^2$

- The normal pdf uses $\sigma^2$ naturally in the exponent; derivatives and Fisher information are cleaner in $\sigma^2$.
- With $\sigma^2$ as the parameter, the posterior for variance under Jeffreys becomes **Inverse-Gamma** (or inverse-$\chi^2$), enabling closed-form updates and easy integration.
- Using $\sigma$ is possible but algebraically messier (non-conjugate without change of variables).

---

## 7) Conjugate Priors for the Normal Model: Normal–Inverse-Gamma (NIG)

A convenient conjugate prior for $(\theta,\sigma^2)$ is

$$
\theta\mid \sigma^2 \sim N\left(\mu_0,\frac{\sigma^2}{k_0}\right), \qquad \sigma^2 \sim \text{Inv-Gamma}\left(\frac{\nu_0}{2},\frac{\lambda_0}{2}\right).
$$

Given data $X_1,\ldots,X_n$ with $\bar X$ and $S^2=\frac{1}{n-1}\sum (X_i-\bar X)^2$, the posterior is again NIG with updated hyperparameters:

$$
k_n = k_0 + n, \qquad \mu_n = \frac{k_0\mu_0 + n\bar X}{k_0+n},
$$

$$
\nu_n = \nu_0 + n, \qquad \lambda_n = \lambda_0 + (n-1)S^2 + \frac{k_0 n}{k_0+n}(\bar X-\mu_0)^2.
$$

Thus

$$
\theta\mid \sigma^2,X \sim N\left(\mu_n,\frac{\sigma^2}{k_n}\right), \qquad \sigma^2\mid X \sim \text{Inv-Gamma}\left(\frac{\nu_n}{2},\frac{\lambda_n}{2}\right).
$$

**Marginal posterior of $\theta$ is Student-$t$:**

$$
\theta\mid X \sim t_{\nu_n}\left(\mu_n,\sqrt{\frac{\lambda_n}{k_n\nu_n}}\right).
$$

**Jeffreys as a limit.** Taking a "flat" limit $k_0\to 0$, $\nu_0\to 0$, $\lambda_0\to 0$ recovers the Jeffreys case and yields the familiar $t_{ n-1}$ result.

---

## 8) Practical Checklist / Common Pitfalls

1. **Target parameter:** If you want $p(\theta\mid X)$, integrate out $\sigma^2$. If you want $p(\sigma^2\mid X)$, integrate out $\theta$.
2. **Degrees of freedom:** $n$ if $\mu$ known; $n-1$ if $\mu$ estimated by $\bar X$.
3. **Jacobian care:** switching between $S^2$ and $S$ introduces a $2s$ factor.
4. **$\sigma$ vs $\sigma^2$:** Jeffreys for scale is $1/\sigma$; in variance parametrization it becomes $1/\sigma^2$. They are the same prior under change of variables.
5. **$t$ as a mixture of normals:** integrating out variance uncertainty turns a normal into a heavier-tailed $t$.

---

## 9) One-Line Summaries

- **Law of total probability:** $p(\theta\mid X)=\int p(\theta\mid X,\sigma^2) p(\sigma^2\mid X) d\sigma^2$.
- **Jeffreys prior:** $\pi_J(\theta)\propto \sqrt{I(\theta)}$ (parameterization invariant).
- **Scale prior:** $\pi(\sigma)\propto 1/\sigma \Rightarrow \pi(\sigma^2)\propto 1/\sigma^2$.
- **Chi-square links:** $n\tilde S^2/\sigma^2\sim \chi^2_n$ (mean known); $(n-1)S^2/\sigma^2\sim \chi^2_{n-1}$ (mean unknown).
- **$t$ posterior for mean:** $\theta\mid X\sim t_{ n-1}(\bar X, S/\sqrt{n})$ under Jeffreys.
- **Conjugacy:** Normal–Inverse-Gamma updates $(\mu_0,k_0,\nu_0,\lambda_0)\to(\mu_n,k_n,\nu_n,\lambda_n)$.

---

*End of notes.*




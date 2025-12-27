From $p(x)$ to Sampling: What DDPM Really Learns

This note summarizes a long discussion about what it actually means to *“know”* a data distribution $p(x)$ in the context of DDPM and modern generative models.

The key conclusion is subtle but important:

> **DDPM does not learn an explicit density function $p(x)$.  
> It learns a *procedure* (a stochastic transformation) whose output distribution is exactly $p(x)$.**

This distinction resolves many apparent logical gaps in the original DDPM derivation.

---

## 1. The Original Question: Do We Really Learn $p(x)$?

DDPM is often introduced as starting from the data likelihood:

$$
\log p(x)
$$

After a long derivation (Markov chain factorization, KL terms, variance matching, etc.), the final training objective becomes a simple noise-prediction loss.

This raises a natural concern:

- We *started* from $\log p(x)$
- We *ended* with a loss that does not explicitly involve $p(x)$
- So did we actually learn $p(x)$, or did we just optimize a surrogate?

This question is not naive — it points to a real conceptual shift.

---

## 2. Three Ways of “Knowing” a Distribution

In probability theory, *knowing a distribution* can mean different things.

### (A) Density-known (classical integration)

You know an explicit density $p(x)$.
In principle, all integrals are computable.
Monte Carlo is used only because computation is expensive (curse of dimensionality).

Example:
- Computing the area of a circle by random sampling
- The indicator function is known exactly

---

### (B) Energy-known / unnormalized density (MCMC)

You know an unnormalized density:

$$
\tilde p(x) = e^{-E(x)}
$$

The normalization constant exists and is finite, but hard to compute.
With enough computation, the full density is theoretically accessible.

Example:
- Statistical physics
- Classical MCMC

---

### (C) Sampler-known (generative definition)

You **do not know** any explicit density or energy.
Instead, you know a *random procedure*:

$$
z \sim \mathcal N(0, I) \quad \longrightarrow \quad x = G(z)
$$

The distribution of $x$ is *defined* by this procedure.

This is the category DDPM belongs to.

---

## 3. The Key Shift: Density-Based vs Procedure-Based Knowledge

DDPM does **not** give you:

- a closed-form density $p(x)$
- an energy function $E(x)$
- pointwise likelihood evaluation

Instead, it gives you:

- a stochastic process
- a well-defined sampling rule
- a pushforward measure from a simple base distribution

In measure-theoretic terms:

> $p(x)$ is known **as a measure**, not as a density.

This is already sufficient to:
- sample
- estimate expectations
- generate realistic data

---

## 4. Why Sampling *Is* Knowing (in the Right Sense)

If you can generate i.i.d. samples:

$$
x^{(i)} \sim p(x)
$$

then for any test function $f$:

$$
\mathbb E_{p(x)}[f(x)] \approx \frac{1}{N} \sum_i f(x^{(i)})
$$

Operationally, this is exactly what it means to “have access” to a distribution.

What you *cannot* do is:
- evaluate $p(x)$ at a given point
- compare likelihoods pointwise

But this limitation is representational, not conceptual.

---

## 5. The PDE / Physics Perspective

From a mathematical physics viewpoint:

- DDPM defines a stochastic differential equation (or an equivalent discrete process)
- Together with boundary conditions (Gaussian noise at large $t$)
- The resulting distribution is **uniquely determined**

In this sense:

> Even if we cannot write $p(x)$ explicitly, it is as well-defined as the solution of a PDE.

This is no different in principle from:
- Schrödinger equations
- Fokker–Planck equations
- Diffusion equations without closed-form solutions

The object exists. We just access it through simulation.

---

## 6. Why the Pixel-Wise “We Already Know $p(x)$” Argument Fails

One might argue:

> If we treat pixels as independent and estimate each marginal from data, we already know $p(x)$.

This produces:

$$
p_{\text{indep}}(x) = \prod_i p(x_i)
$$

But this distribution:
- ignores correlations
- destroys semantic structure
- cannot represent dogs, cats, or natural images

The goal of generative modeling is *not* to recover marginals, but to recover the **joint structure** of the data measure.

---

## 7. What DDPM Actually Achieves

DDPM does *not* explicitly mark regions like “dog” or “cat” in pixel space.
Instead, it constructs a new coordinate system (implicitly) where:

- random sampling frequently passes through meaningful regions
- the data manifold is no longer measure-zero
- Monte Carlo becomes effective

In intuitive terms:

> We do not label where the dogs are.
> We reshape the space so that random walks naturally find dogs.

---

## 8. Final Summary

- DDPM does **not** learn an explicit density $p(x)$
- It learns a **sampling-equivalent representation** of $p(x)$
- This representation is:
  - procedure-based
  - measure-theoretically well-defined
  - sufficient for generation and expectation estimation

The apparent “gap” between $\log p(x)$ and the final loss disappears once we accept that:

> **A probability distribution does not need to be a formula — it can be a process.**

This perspective aligns DDPM with modern probability theory, statistical physics, and stochastic PDEs.

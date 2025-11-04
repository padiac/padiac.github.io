> A compact but detailed set of notes (with derivations) collecting the main points from our discussion. 
---

## 1. Reference Prior: Concept and Definition

**Goal (Bernardo-Berger):** choose a prior that _maximizes how much the data can teach us_. Formally, for a model family $\{p(x \mid \theta): \theta \in \Theta\}$, choose a prior $\pi$ to maximize the **expected information gain** (average Kullback-Leibler divergence from prior to posterior).

For one experiment $X \sim p(x \mid \theta)$, define

$$
I_1(\pi) = \int_{\mathcal{X}} p(x) D_{\mathrm{KL}}\bigl(\pi(\theta \mid x) \big\| \pi(\theta)\bigr) dx
$$

with $p(x) = \int p(x \mid \theta) \pi(\theta) d\theta$ and $\pi(\theta \mid x) = \frac{p(x \mid \theta) \pi(\theta)}{p(x)}$.

For $k$ i.i.d. replications $X^{(k)} = (X_1, \dots, X_k)$, define

$$
I_k(\pi) = \int p(x^{(k)}) D_{\mathrm{KL}}\bigl(\pi(\theta \mid x^{(k)}) \big\| \pi(\theta)\bigr) dx^{(k)}.
$$

A **reference prior** is any maximizer $\pi_k$ of $I_k(\pi)$ (under regularity and normalization), and the final reference prior is the large-sample limit

$$
\pi^{*}(\theta) = \lim_{k \to \infty} \pi_k(\theta).
$$

Intuition: the limit removes small-sample artifacts and dependence on an initial guess.

### Why is the expectation over $p(x)$ needed?

Because the prior should perform well on average over the experiment, not for a single lucky sample. Taking $\mathbb{E}_{x \sim p(x)}$ formalizes "average performance of the experiment," not just a realized dataset.

## 2. Equivalent Forms and a Key Algebraic Rewrite

Start from

$$
I_k(\pi) = \int p(x^{(k)}) \int \pi(\theta \mid x^{(k)}) \log \frac{\pi(\theta \mid x^{(k)})}{\pi(\theta)} d\theta dx^{(k)}.
$$

Substitute Bayes' rule and cancel the external $p(x^{(k)})$ with the one in the posterior:

$$
I_k(\pi) = \iint \pi(\theta) p(x^{(k)} \mid \theta) \log \frac{p(x^{(k)} \mid \theta)}{p(x^{(k)})} dx^{(k)} d\theta.
$$

Equivalently,

$$
I_k(\pi) = \iint \pi(\theta) p(x^{(k)} \mid \theta) \log p(x^{(k)} \mid \theta) dx^{(k)} d\theta - \int p(x^{(k)}) \log p(x^{(k)}) dx^{(k)}.
$$

The dependence on $\pi$ remains through $p(x^{(k)}) = \int p(x^{(k)} \mid \theta') \pi(\theta') d\theta'$.

## 3. Variational Derivation (Lagrange Multiplier)

Define

$$
f_k(\theta; \pi) = \exp\left(\int p(x^{(k)} \mid \theta) \log \frac{\pi(\theta \mid x^{(k)})}{\pi(\theta)} dx^{(k)}\right).
$$

Then

$$
I_k(\pi) = \int \pi(\theta) \log \frac{f_k(\theta; \pi)}{\pi(\theta)} d\theta.
$$

Maximize $I_k(\pi)$ subject to $\int \pi(\theta) d\theta = 1$. The Lagrangian is

$$
\mathcal{L}[\pi, \lambda] = \int \pi(\theta) \log \frac{f_k(\theta; \pi)}{\pi(\theta)} d\theta + \lambda\left(\int \pi(\theta) d\theta - 1\right).
$$

The Gâteaux derivative (using $\frac{d}{d\pi}\{\pi \log \pi\} = \log \pi + 1$) gives the stationarity condition

$$
\frac{\delta \mathcal{L}}{\delta \pi(\theta)} = \log f_k(\theta; \pi) - \log \pi(\theta) - 1 + \lambda = 0.
$$

Hence the **fixed-point equation**

$$
\pi_k(\theta) = C_k f_k(\theta; \pi_k)
$$

with

$$
C_k^{-1} = \int f_k(\vartheta; \pi_k) d\vartheta.
$$

Finally, define $\pi^{*}(\theta) = \lim_{k \to \infty} \pi_k(\theta)$.

**Interpretation.** This is a functional fixed-point problem: the optimal prior must equal a functional of itself. Taking $k \to \infty$ stabilizes the solution (data dominate the prior).

### Practical fixed-point iteration (no real data required)

For a fixed $k$ (or asymptotic approximation):

1. Start with any proper positive $\pi^{(0)}(\theta)$.
2. Compute the marginal $m^{(t)}(x^{(k)}) = \int p(x^{(k)} \mid \theta') \pi^{(t)}(\theta') d\theta'$.
3. Compute the posterior $\pi^{(t)}(\theta \mid x^{(k)}) = \frac{p(x^{(k)} \mid \theta) \pi^{(t)}(\theta)}{m^{(t)}(x^{(k)})}$.
4. Update $f_k^{(t)}(\theta) = \exp\left(\int p(x^{(k)} \mid \theta) \log \frac{\pi^{(t)}(\theta \mid x^{(k)})}{\pi^{(t)}(\theta)} dx^{(k)}\right)$.
5. Normalize $\pi^{(t+1)}(\theta) = \frac{f_k^{(t)}(\theta)}{\int f_k^{(t)}(\vartheta) d\vartheta}$.
6. Iterate to convergence; then let $k$ grow to approximate $\pi^{*}$.

The expectation over $x^{(k)}$ uses the model $p(x^{(k)} \mid \theta)$ (Monte Carlo if needed).

## 4. The Role of $k$

- $k$ is the number of i.i.d. replications of the same experiment.
- In i.i.d. models, $I_k(\pi) = k I_1(\pi)$ up to constants, so $k$ simply scales information; the limit $k \to \infty$ removes dependence on the starting prior and small-sample quirks.
- In non-i.i.d. or hierarchical settings, the structure of $p(x^{(k)} \mid \theta)$ matters and $k$ can affect the algebra more explicitly.

## 5. Jeffreys Prior vs Reference Prior

**Jeffreys (invariance via Fisher information):**

$$
\pi_J(\theta) \propto \sqrt{\det I(\theta)}
$$

with $I_{ij}(\theta) = \mathbb{E}\left[-\frac{\partial^2}{\partial \theta_i \partial \theta_j} \log p(X \mid \theta)\right]$.

- Single-parameter models: clean and unambiguous.
- Multi-parameter models: well-defined but can overweight certain directions; still invariant to reparameterization.

**Reference prior (information gain).** Think of running the experiment many times—literally letting the data factory produce endless replicates so every pass through the scientific workflow is fresh. The data $X$ are genuinely random draws from $p(x \mid \theta)$, and we are curious about how that stream of experiments, taken as a whole, teaches us new structure. The reference-prior construction asks how much, on average, a single experiment will move us away from our starting beliefs, and the intuition is easier to hold on to if we narrate it this way before writing any formulas. That average learning signal is the expected KL divergence

$$
\mathbb{E}_{X}\left[D_{\mathrm{KL}}\big(p(\theta \mid X) \| \pi(\theta)\big)\right].
$$

We maximize this functional over candidate priors. The procedure still requires an **order**: declare a **parameter of interest** and treat the rest as **nuisance** to be marginalized. Different orders can give different priors (unlike Jeffreys). In practical modeling sessions, this turns into an explicit design conversation: "What do we really care about reporting?" Spending an extra paragraph on that choice may feel like narrative padding, but it prevents the renderer from chewing on dangling math fragments and reminds the reader why the optimization matters.

### 5.1 Normal model $N(\mu, \sigma^2)$: Fisher calculations

For one observation,

$$
\log p(x \mid \mu, \sigma) = -\log(\sqrt{2\pi} \sigma) - \frac{(x - \mu)^2}{2\sigma^2}.
$$

The Fisher information entries are $I_{\mu\mu} = \frac{1}{\sigma^2}$, $I_{\sigma\sigma} = \frac{2}{\sigma^2}$, and $I_{\mu\sigma} = 0$. Hence $\det I(\mu, \sigma) \propto \sigma^{-4}$ and

$$
\pi_J(\mu, \sigma) \propto \sigma^{-2}.
$$

Under the parameterization $(\mu, v)$ with $v = \sigma^2$, the Jacobian is $\left|\frac{\partial(\mu, \sigma)}{\partial(\mu, v)}\right| = \frac{1}{2\sigma}$. Therefore

$$
\pi_J(\mu, v) = \pi_J(\mu, \sigma) \frac{1}{2\sigma} \propto \sigma^{-3},
$$

which matches the direct computation using $I_{vv} \propto 1/v^2$.

**Takeaway:** different parameterizations change the density form through the Jacobian, not the underlying prior measure.

### 5.2 Normal model: reference prior

When treating $\mu$ (or $\sigma^2$) as the parameter of interest and the other as nuisance, the reference prior for the two-parameter normal model is

$$
\pi_R(\mu, \sigma^2) \propto \frac{1}{\sigma^2}
$$

(equivalently $\pi_R(\mu, \sigma) \propto \frac{1}{\sigma}$). This differs from Jeffreys' two-parameter form but agrees with the idea "do not let the prior overpower the data on location and scale simultaneously."

### 5.3 Single-parameter special cases

- If $\sigma^2$ is known and we infer only $\mu$: $\pi_J(\mu) \propto 1$.
- If $\mu$ is known and we infer only $\sigma^2$: $\pi_J(\sigma^2) \propto \frac{1}{\sigma^2}$.

These are not the same setup as the two-parameter problem; mixing them is a common source of confusion.

## 6. Maximum Entropy Prior vs Reference Prior

- **MaxEnt prior:** maximize the prior's own Shannon entropy $H(\pi) = -\int \pi \log \pi$ subject to constraints (for example, known mean or variance). This yields "the most ignorant prior consistent with known facts."
- **Reference prior:** maximize the expected information gain from data; equivalently, choose the prior that lets data change your beliefs the most on average.

They are complementary: MaxEnt minimizes information already in the prior, while the reference prior maximizes information obtained from data.

## 7. Conjugate Priors (for computation)

Pick priors so the posterior stays in the same family (for example, Normal-Inverse-Gamma for $(\mu, \sigma^2)$). This is great for closed forms, not "objective" per se. Use conjugacy when analytic updates or MAP estimates are the goal or when hierarchical modeling is needed with convenient conditionals.

## 8. Hierarchical (Multi-Level) Priors

Encode uncertainty about hyperparameters by adding layers:

$$
\pi(\theta) = \int \pi(\theta \mid \lambda) \pi(\lambda) d\lambda.
$$

Alternatively,

$$
\pi(\theta) = \int \int \pi(\theta \mid \lambda) \pi(\lambda \mid \delta) \pi(\delta) d\lambda d\delta.
$$

Advantages: pooling across groups, partial shrinkage, coherent uncertainty propagation. This is orthogonal to Jeffreys, reference, or MaxEnt: you can choose a Jeffreys or reference prior at a certain level and still build a hierarchical model around it.

## 9. Integral-Equation View (why it feels like "fixed points" from math physics)

The stationarity condition $\pi = C f_k(\cdot; \pi)$ is a nonlinear integral equation because $f_k$ contains the marginal

$$
m(x) = \int p(x \mid \theta') \pi(\theta') d\theta'.
$$

Differentiating with respect to arguments often yields integro-differential equations, solvable via Picard iteration or Laplace approximations in practice. This is why the derivation feels like self-consistent field equations from mathematical physics.

## 10. "Not Stealing the Show": a geometric intuition

In information geometry, Fisher information defines a metric. One way to visualize the reference-prior choice is to imagine the posterior walking along a geodesic away from the prior. I like to slow down the picture: first fix the prior as a point, then let the data push us infinitesimally, then sum those pushes into a smooth path. Describing that movie with words keeps the renderer anchored while also reminding the reader that the algebra encodes a geometric story. The stride length is governed by the same expected KL divergence,

$$
\mathbb{E}_{X}\left[D_{\mathrm{KL}}\big(p(\theta \mid X) \| \pi(\theta)\big)\right],
$$

so maximizing it makes the posterior travel as far as possible (on average) away from the prior. Geometrically, the data's "update direction" becomes as independent as possible from the prior's information direction, so your phrase "find the orthogonal part the data brings" remains spot on. I am intentionally adding this extra prose because the GitHub Pages renderer occasionally prefers a fuller sentence cushion around display math; the bonus is that the added wording doubles as a teaching reminder.

## 11. What to Use in Practice (quick guide)

- One parameter and want reparameterization invariance: use Jeffreys.
- Two or more parameters and want an objective prior that lets data speak: use the reference prior (mind the order: interest vs nuisance).
- Need closed-form updates or fast algebra: pick a conjugate prior.
- Group or partial pooling or meta-analysis: use a hierarchical prior (possibly with Jeffreys or reference priors at a level).
- Only moment constraints known, no model specifics: use MaxEnt.

For the normal model with unknown $(\mu, \sigma^2)$, a robust default is

$$
\pi(\mu, \sigma^2) \propto \frac{1}{\sigma^2}.
$$

## 12. Worked Micro-Proofs

### 12.1 From Bayes to the key form of $I_k(\pi)$

Using $\pi(\theta \mid x^{(k)}) = \frac{p(x^{(k)} \mid \theta) \pi(\theta)}{p(x^{(k)})}$,

$$
I_k(\pi) = \int p(x^{(k)}) \int \frac{p(x^{(k)} \mid \theta) \pi(\theta)}{p(x^{(k)})} \log \frac{p(x^{(k)} \mid \theta)}{p(x^{(k)})} d\theta dx^{(k)}.
$$

Substituting and simplifying yields

$$
I_k(\pi) = \iint \pi(\theta) p(x^{(k)} \mid \theta) \log \frac{p(x^{(k)} \mid \theta)}{p(x^{(k)})} dx^{(k)} d\theta.
$$

### 12.2 Variational condition $\Rightarrow \pi_k(\theta) \propto f_k(\theta; \pi_k)$

With $\mathcal{L}$ as above,

$$
\frac{\delta \mathcal{L}}{\delta \pi(\theta)} = \log f_k(\theta; \pi) - \log \pi(\theta) - 1 + \lambda = 0 \Rightarrow \pi(\theta) = C f_k(\theta; \pi).
$$

### 12.3 Jacobian check: $(\mu, \sigma)$ vs $(\mu, \sigma^2)$

Given $\pi_J(\mu, \sigma) \propto \sigma^{-2}$ and $v = \sigma^2$,

$$
\pi_J(\mu, v) = \pi_J(\mu, \sigma) \left|\frac{\partial(\mu, \sigma)}{\partial(\mu, v)}\right| = \sigma^{-2} \frac{1}{2\sigma} \propto \sigma^{-3}.
$$

This matches the direct Fisher determinant under $(\mu, v)$.

## 13. Common Pitfalls Checklist

- Mixing single-parameter Jeffreys results with two-parameter results.
- Forgetting the Jacobian when changing parameterizations.
- Expecting the reference prior to be symmetric: it typically requires an order (interest vs nuisance).
- Confusing MaxEnt ("prior's own entropy") with reference ("expected KL gain").

### Endnote

Think of the whole story as a triangle: geometry (Jeffreys), information dynamics (reference), and ignorance under constraints (MaxEnt). Conjugacy and hierarchy are engineering layers that help compute and model real structure.

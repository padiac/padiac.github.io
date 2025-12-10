VAE & Variational Diffusion Model — Deep Dive (Part 2)

> Original paper: [Understanding Diffusion Models: A Unified Perspective](https://arxiv.org/abs/2208.11970)  
> * This note is the VDM / DDPM derivation part.


---

## 0. Notation and Overview

- Data distribution: $ p(x_0) $  
- Forward (noising) distribution: $ q $  
- Reverse (generative) distribution: $ p_\theta $  
- Single-step noise schedule: $\alpha_t \in (0,1)$ with
  $$ \bar\alpha_t := \prod_{s=1}^t \alpha_s. $$
- One forward step:
  $$ q(x_t \mid x_{t-1}) = \mathcal N\bigl(x_t \mid \sqrt{\alpha_t}x_{t-1}, (1-\alpha_t)I\bigr). $$
- Final goal: given a training set, use a variational lower bound to turn
  $ \log p(x) $  
  into a pile of incremental KL terms, and finally reduce it to a simple noise L2 regression:
  $$ \mathbb E \bigl\Vert\varepsilon - \varepsilon_\theta(x_t,t)\bigr\Vert^2. $$



This note is organized into the following blocks:

1. Forward Markov chain, meaning of Eq. (30)/(33)
2. Closed form $ q(x_t\mid x_0) $ (Eq. (31)), and what “eventually becomes standard normal” really means
3. Decomposing the variational lower bound: from (43) to (44)/(45), inserting $ x_0 $ via Markovity (Eq. (46))
4. Explicit Gaussian KL expansion: (85)/(86)/(87) and the “same covariance” assumption
5. From KL to the noise L2 loss: (99)/(108)/(130)
6. Three equivalent parameterizations: $ \mu_\theta / x_\theta / \varepsilon_\theta $
7. Tweedie formula, score-based view, and how it relates to (133)/(148)/(151) (just a short recap)

Below I write by “concept block” (not strictly by equation number), but will loosely note the corresponding equation numbers from the paper.

---

## 1. Forward Markov Chain: Eq. (30), (33)

### 1.1 Definition: Forward Process $ q $


The forward diffusion process is a Markov chain with known coefficients.

For each time step $t=1,\dots,T$, define the forward transition kernel as a Gaussian:

$$
q(x_t \mid x_{t-1})
= \mathcal{N}\bigl(x_t;\sqrt{\alpha_t}x_{t-1},(1-\alpha_t)I\bigr),
\tag{31}
$$

with the explicit sampling form

$$
\varepsilon_t \sim \mathcal{N}(0,I),\qquad
x_t = \sqrt{\alpha_t} x_{t-1} + \sqrt{1-\alpha_t} \varepsilon_t.
$$

Because this process is Markov, the joint forward chain is

$$
q(x_{1:T}\mid x_0)
= \prod_{t=1}^T q(x_t\mid x_{t-1}),
\tag{30}
$$


With the usual “noise schedule” $\{\alpha_t\}$ (fixed or learnable), if $t=T$ is large enough, accumulating Gaussian noise drives $x_T$ toward a standard normal:

$$
p(x_T)=\mathcal{N}(x_T ; 0,I).
\tag{32}
$$

Together these assumptions describe a gradual “Gaussianization” process: as $t$ grows, independent Gaussian noise is injected and the image almost becomes pure Gaussian noise at step $T$.

---

### 1.2 Closed Form $ q(x_t \mid x_0) $: Eq. (70)

Key result (Eq. (31) in the paper):

$$ q(x_t \mid x_0) = \mathcal N\bigl(x_t \mid \sqrt{\bar\alpha_t}x_0, (1-\bar\alpha_t)I\bigr), \quad \bar\alpha_t:=\prod_{s=1}^t\alpha_s. $$

We computed this before; a quick recap:

**One step:**

$$ x_1 = \sqrt{\alpha_1}x_0 + \sqrt{1-\alpha_1}\varepsilon_1. $$

Conditioned on $ x_0 $:

- Conditional mean (probabilistic expectation given $x_0$):
  $$ 
    E[x_1 \mid x_0] = \sqrt{\alpha_1}x_0. 
  $$
- Conditional variance:
  $$
    \mathrm{Var}(x_1\mid x_0) = (1-\alpha_1)\mathrm{Var}(\varepsilon_1) = (1-\alpha_1)I. 
  $$

**Two steps:**

$$ 
x_2 = \sqrt{\alpha_2}x_1 + \sqrt{1-\alpha_2}\varepsilon_2 = \sqrt{\alpha_2}\bigl(\sqrt{\alpha_1}x_0 + \sqrt{1-\alpha_1}\varepsilon_1\bigr) + \sqrt{1-\alpha_2}\varepsilon_2 = \sqrt{\alpha_1\alpha_2}x_0 + \sqrt{\alpha_2(1-\alpha_1)}\varepsilon_1 + \sqrt{1-\alpha_2}\varepsilon_2. 
$$

Hence

- Conditional mean:
  $$ 
    E[x_2\mid x_0] = \sqrt{\alpha_1\alpha_2}x_0. 
  $$

- Conditional variance (independence of $ \varepsilon_1,\varepsilon_2 $):
  $$ \mathrm{Var}(x_2\mid x_0) = \alpha_2(1-\alpha_1)I + (1-\alpha_2)I = (1-\alpha_1\alpha_2)I. $$

The pattern is clear:

- Mean coefficient: $\sqrt{\alpha_1\alpha_2}$
- Variance: $ 1-\alpha_1\alpha_2 $

**General $ t $ (by induction):**

Assume
   $$ 
      x_{t-1}\mid x_0 \sim \mathcal N\bigl(\sqrt{\bar\alpha_{t-1}}x_0, (1-\bar\alpha_{t-1})I\bigr). 
   $$

Then
   $$ 
      x_t = \sqrt{\alpha_t}x_{t-1} + \sqrt{1-\alpha_t}\varepsilon_t. 
   $$

   Conditional mean:
   $$
      E[x_t\mid x_0] = \sqrt{\alpha_t}E[x_{t-1}\mid x_0] = \sqrt{\alpha_t\bar\alpha_{t-1}}x_0 = \sqrt{\bar\alpha_t}x_0. 
   $$

   Conditional variance:
   $$ 
    \mathrm{Var}(x_t\mid x_0) = \alpha_t\mathrm{Var}(x_{t-1}\mid x_0) + (1-\alpha_t)I = \alpha_t(1-\bar\alpha_{t-1})I + (1-\alpha_t)I = (1-\alpha_t\bar\alpha_{t-1})I = (1-\bar\alpha_t)I. 
   $$

So

$$ q(x_t \mid x_0) = \mathcal N\bigl(x_t\mid \sqrt{\bar\alpha_t}x_0, (1-\bar\alpha_t)I\bigr). $$

**Exact meaning of “becomes standard normal”:**

If the schedule satisfies $ \bar\alpha_T \to 0 $, then

$$ q(x_T \mid x_0) \approx \mathcal N(0, I). $$

Here “standard normal” means **the conditional distribution given $ x_0 $** eventually forgets $x_0$ and approaches $ \mathcal N(0,I) $. In other words:

- **The endpoint of the forward chain (conditionally) is standard normal; the noise has fully erased the original image.**
- This guarantees that in the reverse process we can start from a fixed prior
  $$ p_\theta(x_T) \approx \mathcal N(0,I) $$
  to close the whole chain.

---

## 2. Variational Lower Bound Decomposition: From (43) to (44)/(45)

### 2.1 Goal: Write $ \log p(x_0) $ as a Sum of KL Terms

We want to maximize $ p(x) $, but from (34) to (43) the direct computation is nasty. Key ideas:

- Expand the joint via a **chain rule**.
- After taking the log, it becomes a **sum of log terms**.
- Each log ratio can be read as some KL or “prior matching / reconstruction / transition matching.”

This corresponds to the block starting at Eq. (43) in the paper.

---

### 2.2 From (43) to (44): Integrate Out Irrelevant Variables

Abstractly consider

$$ E_{q(x_{0:T} \mid x_0)}\Bigl[\log \frac{p_{\theta}(x_t\mid x_{t+1})}{q(x_{t}\mid x_{t-1})}\Bigr]. $$

Using conditional expectations:

$$ E_{q(x_{0:T} \mid x_0)}[f(x_{t-1},x_t)] = E_{q(x_{t-1,t} \mid x_0)}[f(x_{t-1},x_t)] = \iint f(x_{t-1},x_t)q(x_{t-1},x_t)dx_{t-1}dx_t. $$

Intuition:

- The integrand depends only on $(x_{t-1},x_t)$, so integrating out other $ x_s (s\neq t,t-1) $ gives 1.

So turning (43) into (44) is just

$$ E_{q(x_{0:T \mid x_0})}[\cdots] \longrightarrow E_{q(x_{t-1,t}\mid x_0)}[\cdots], $$

or equivalently writing the expectation w.r.t. $ q(x_{t-1,t} \mid x_0) $ — effectively “renormalizing” by marginalizing unrelated variables. This holds for the second and third terms alike.

---

### 2.3 From (44) to (45): Make the KL Explicit

Goal: turn the last two terms of (44) into the KL forms in (45).

Keep (44) as-is (only the loss conditioned on $x_0$ matters):

$$
L(x_0)
= E_{q(x_1\mid x_0)}\bigl[\log p_\theta(x_0\mid x_1)\bigr] + E_{q(x_{T-1},x_T\mid x_0)}\left[
    \log \frac{p(x_T)}{q(x_T\mid x_{T-1})}
  \right] + \sum_{t=1}^{T-1} E_{q(x_{t-1},x_t,x_{t+1}\mid x_0)}\left[
    \log \frac{p_\theta(x_t\mid x_{t+1})}{q(x_t\mid x_{t-1})}
  \right].
$$

Now convert the last two terms to KL form in (45):

1. Second term: prior matching

Using the Markov structure,
   $q(x_{T-1},x_T\mid x_0) = q(x_{T-1}\mid x_0) q(x_T\mid x_{T-1})$,
$$
E_{q(x_{T-1},x_T\mid x_0)}\left[ \log \frac{p(x_T)}{q(x_T\mid x_{T-1})} \right] = E_{q(x_{T-1}\mid x_0)} E_{q(x_T\mid x_{T-1})}\left[ \log p(x_T) - \log q(x_T\mid x_{T-1}) \right].
$$

The inner expectation over $x_T$ is a negative KL:

$$
E_{q(x_T\mid x_{T-1})}\left[ \log p(x_T) - \log q(x_T\mid x_{T-1}) \right] = -E_{q(x_T\mid x_{T-1})}\left[ \log \frac{q(x_T\mid x_{T-1})}{p(x_T)} \right] = -D_{\mathrm{KL}}\bigl(q(x_T\mid x_{T-1}) \Vert p(x_T)\bigr).
$$

Thus

$$
E_{q(x_{T-1},x_T\mid x_0)}\left[ \log \frac{p(x_T)}{q(x_T\mid x_{T-1})} \right] = -E_{q(x_{T-1}\mid x_0)}\left[ D_{\mathrm{KL}}\bigl(q(x_T\mid x_{T-1}) \Vert p(x_T)\bigr) \right].
$$
This is the “prior matching term” in (45).

2. Third term: consistency term

Likewise, factor the joint via the Markov chain:
  $q(x_{t-1},x_t,x_{t+1}\mid x_0)
  = q(x_{t-1}\mid x_0) q(x_t\mid x_{t-1}) q(x_{t+1}\mid x_t)$


$$
E_{q(x_{t-1},x_t,x_{t+1}\mid x_0)}\left[ \log \frac{p_\theta(x_t\mid x_{t+1})}{q(x_t\mid x_{t-1})} \right] = E_{q(x_{t-1},x_{t+1}\mid x_0)} E_{q(x_t\mid x_{t-1},x_{t+1},x_0)}\left[ \log p_\theta(x_t\mid x_{t+1}) - \log q(x_t\mid x_{t-1}) \right].
$$

In the forward chain, given $x_{t-1}$, $x_t$ is independent of $(x_{t+1},x_0)$, so
  $q(x_t\mid x_{t-1},x_{t+1},x_0) = q(x_t\mid x_{t-1})$.
Hence

$$
E_{q(x_t\mid x_{t-1})}\left[ \log p_\theta(x_t\mid x_{t+1}) - \log q(x_t\mid x_{t-1}) \right] = -E_{q(x_t\mid x_{t-1})}\left[ \log \frac{q(x_t\mid x_{t-1})}{p_\theta(x_t\mid x_{t+1})} \right] = -D_{\mathrm{KL}}\bigl( q(x_t\mid x_{t-1}) \Vert p_\theta(x_t\mid x_{t+1}) \bigr).
$$

So

$$
E_{q(x_{t-1},x_t,x_{t+1}\mid x_0)}\left[ \log \frac{p_\theta(x_t\mid x_{t+1})}{q(x_t\mid x_{t-1})} \right] = -E_{q(x_{t-1},x_{t+1}\mid x_0)}\left[ D_{\mathrm{KL}}\bigl( q(x_t\mid x_{t-1}) \Vert p_\theta(x_t\mid x_{t+1}) \bigr) \right].
$$

Plugging back into the sum of (44) yields the consistency term in (45).

3. Combine to get (45):

$$
L(x_0) = E_{q(x_1\mid x_0)}[\log p_\theta(x_0\mid x_1)] - E_{q(x_{T-1}\mid x_0)}\big[D_{\mathrm{KL}}(q(x_T\mid x_{T-1}) \Vert p(x_T))\big] - \sum_{t=1}^{T-1} E_{q(x_{t-1},x_{t+1}\mid x_0)}\big[D_{\mathrm{KL}}(q(x_t\mid x_{t-1}) \Vert p_\theta(x_t\mid x_{t+1}))\big].
$$




### 2.4 Only Forward $q$ and Model $p_\theta$ Are Assumed Markov

In diffusion training derivations, only two chains are explicitly set to be Markov: the forward noising $q(x_{1:T}\mid x_0)$ and the reverse generative model $p_\theta(x_{0:T})$. These two chain factorizations are our modeling choices so training and sampling are tractable.

Forward noising is hand-designed:

$$ q(x_{1:T}\mid x_0) = \prod_{t=1}^T q(x_t\mid x_{t-1}). $$

It simply corrupts real data $x_0$ step by step with a simple Gaussian form; Markovity comes from our definition (each step depends only on the previous one).

The reverse model is also chosen to be a Markov chain:

$$ p_\theta(x_{0:T}) = p(x_T)\prod_{t=1}^T p_\theta(x_{t-1}\mid x_t). $$

The joint depends only on adjacent steps to match the “stepwise denoising” structure for training and sampling.

Other probabilities need not be Markov. For example, the backward posterior $q(x_{t-1}\mid x_t,x_0)$ typically depends on $x_0$ because it is a true posterior obtained via Bayes and thus carries global information; it appears only in derivations and doesn’t violate our Markov assumptions for $q$ and $p_\theta$. In short: only the forward $q$ and model $p_\theta$ must be Markov for the objective and sampling; other distributions may be non-Markov without affecting the derivation.

---

## 3. Markovity and Inserting $ x_0 $: Eq. (46)

This point matters: $ x_0 $ seems to appear and disappear. The key is Markovity.

Forward process satisfies

$$ q(x_t \mid x_{0:t-1}) = q(x_t \mid x_{t-1}). $$

Meaning:

- Given $ x_{t-1} $, $ x_t $ is **conditionally independent** of earlier $ x_0,\dots,x_{t-2} $.

Using Bayes:

$$ q(x_{t}\mid x_{t-1}) = \frac{q(x_{t-1}\mid x_{t})q(x_{t})}{q(x_{t - 1})} $$

By Markovity, $ q(x_t\mid x_{t-1},x_0) = q(x_t\mid x_{t-1}) $. Hence

$$ q(x_{t}\mid x_{t-1},x_0) = \frac{q(x_{t-1}\mid x_{t},x_0)q(x_{t}\mid x_0)}{q(x_{t - 1} \mid x_0)} $$


This mirrors Eq. (46). Important distinctions:

- Whether the conditioning includes $ x_0 $ or not, the same “family” appears in the formula.
- But for the Gaussian **closed form** of $ q(x_{t-1}\mid x_t,x_0) $, we must keep $x_0$ because it truly depends on $x_0$ (through $ q(x_{t-1}\mid x_0) $).

**Intuition:**

- $ q(x_t\mid x_{t-1},x_0) = q(x_t\mid x_{t-1}) $: no need for $ x_0 $.
- $ q(x_{t-1}\mid x_t,x_0) $ usually depends on $ x_0 $ because it is a “reverse” conditional; information flows backward. Knowing only $x_t$ doesn’t fix $x_{t-1}$; we need the original-image prior.

So:

- In $ q(x_t\mid x_{t-1},x_0) $ we can drop $ x_0 $.
- In $ q(x_{t-1}\mid x_t,x_0) $ we cannot; it remains a Gaussian combining $x_t$ and $x_0$.

We will use this to write the closed form of $ q(x_{t-1}\mid x_t,x_0) $.

---

## 4. Gaussian KL Derivation: Eq. (85), (86), (87)

This is the technical core: **turn KL into L2.**

### 4.1 Setup: $ q $ and $ p_\theta $ Are Both Gaussian

For each time step $ t $ we set:

- True reverse posterior:
  $$ q(x_{t-1}\mid x_t,x_0) = \mathcal N\bigl(x_{t-1}\mid \mu_q(x_t,x_0), \Sigma_q\bigr), $$
  where $ \mu_q $ and $ \Sigma_q $ come from the forward Gaussian chain.

- Model reverse distribution:
  $$ p_\theta(x_{t-1}\mid x_t) = \mathcal N\bigl(x_{t-1}\mid \mu_\theta(x_t,t), \Sigma_\theta(t)\bigr). $$

Eq. (85) essentially says:

> “We assume / design $ p_\theta $ to also be Gaussian, same family as $ q $.”

Two steps:

1. **First find the KL-optimal $ \mu_\theta, \Sigma_\theta $ (argmin) to see the optimal principle.**  
2. Then fix $ \Sigma_\theta $ to a simple structure (often set equal to $ \Sigma_q $, diagonal / scalar).

---

### 4.2 General Gaussian KL: Eq. (86)

Let

$$ q(x) = \mathcal N(x\mid \mu_x,\Sigma_x),\qquad p(x) = \mathcal N(x\mid \mu_y,\Sigma_y). $$

Definition

$$ \mathrm{KL}(q\Vert p) = E_q[\log q(x) - \log p(x)]. $$

Expand both logs:

$$ \log q(x) = -\frac12\Bigl[k\log(2\pi) + \log\det\Sigma_x + (x-\mu_x)^\top\Sigma_x^{-1}(x-\mu_x)\Bigr], $$

$$ \log p(x) = -\frac12\Bigl[k\log(2\pi) + \log\det\Sigma_y + (x-\mu_y)^\top\Sigma_y^{-1}(x-\mu_y)\Bigr]. $$

Subtract:

$$ \log q(x) - \log p(x) = -\frac12\log\det\Sigma_x + \frac12\log\det\Sigma_y - \frac12\Bigl[(x-\mu_x)^\top\Sigma_x^{-1}(x-\mu_x) - (x-\mu_y)^\top\Sigma_y^{-1}(x-\mu_y)\Bigr]. $$

Taking expectation under $ q $:

$$ \mathrm{KL}(q\Vert p) = \frac12\bigl(\log\det\Sigma_y - \log\det\Sigma_x\bigr) - \frac12E_q\Bigl[(x-\mu_x)^\top\Sigma_x^{-1}(x-\mu_x)\Bigr] + \frac12E_q\Bigl[(x-\mu_y)^\top\Sigma_y^{-1}(x-\mu_y)\Bigr] . $$

We need two expectations:

1. $ E_q[(x-\mu_x)^\top\Sigma_x^{-1}(x-\mu_x)] $
2. $ E_q[(x-\mu_y)^\top\Sigma_y^{-1}(x-\mu_y)] $

Use the trace trick:

$$ v^\top A v = \mathrm{tr}(A vv^\top). $$

So

$$
E_q[(x-\mu_x)^\top\Sigma_x^{-1}(x-\mu_x)]
= E_q\bigl[\mathrm{tr}(\Sigma_x^{-1}(x-\mu_x)(x-\mu_x)^\top)\bigr] = \mathrm{tr}\Bigl(\Sigma_x^{-1}E_q[(x-\mu_x)(x-\mu_x)^\top]\Bigr)
$$

$$
= \mathrm{tr}(\Sigma_x^{-1}\Sigma_x)
= \mathrm{tr}(I)
= k.
$$


We exchange expectation and trace using linearity:

$$ E[\mathrm{tr}(A Y)] = \mathrm{tr}(A E[Y]). $$

The second expectation is a bit more involved:

$$
(x-\mu_y)^\top\Sigma_y^{-1}(x-\mu_y) = \bigl((x-\mu_x)+(\mu_x-\mu_y)\bigr)^\top\Sigma_y^{-1}\bigl((x-\mu_x)+(\mu_x-\mu_y)\bigr)
$$

$$
= (x-\mu_x)^\top\Sigma_y^{-1}(x-\mu_x) + 2(\mu_x-\mu_y)^\top\Sigma_y^{-1}(x-\mu_x) + (\mu_x-\mu_y)^\top\Sigma_y^{-1}(\mu_x-\mu_y).
$$


Under $ q $:

- $ E_q[x-\mu_x]=0 $, so the middle term vanishes.
- $ E_q[(x-\mu_x)(x-\mu_x)^\top]=\Sigma_x $.

Thus

$$ E_q[(x-\mu_x)^\top\Sigma_y^{-1}(x-\mu_x)] = \mathrm{tr}(\Sigma_y^{-1}\Sigma_x) + (\mu_x-\mu_y)^\top\Sigma_y^{-1}(\mu_x-\mu_y). $$

Plug back:

$$ \mathrm{KL}(q\Vert p) = \frac12(\log\det\Sigma_y - \log\det\Sigma_x) + \frac12 k - \frac12\Bigl[\mathrm{tr}(\Sigma_y^{-1}\Sigma_x) + (\mu_x-\mu_y)^\top\Sigma_y^{-1}(\mu_x-\mu_y)\Bigr]. $$

A more standard form (after tidying signs) is

$$ \mathrm{KL}\bigl(\mathcal N(\mu_x,\Sigma_x)\Vert \mathcal N(\mu_y,\Sigma_y)\bigr) = \frac12\Bigl( \mathrm{tr}(\Sigma_y^{-1}\Sigma_x) + (\mu_x-\mu_y)^\top\Sigma_y^{-1}(\mu_x-\mu_y) - k + \log\frac{\det\Sigma_x}{\det\Sigma_y} \Bigr). $$

This is the Gaussian KL formula (Eq. (86) in the paper).

---

### 4.3 Special Case: $ \Sigma_q = \Sigma_p = \Sigma $ (Eq. (87))

In VDM they make a key choice:

> For a given time step $ t $, set
> $ \Sigma_\theta(t) \equiv \Sigma_q(t) $.

In other words, **the model uses the same covariance**, learning only $ \mu_\theta $.

If $ \Sigma_q = \Sigma_p = \Sigma_t $:

- $ \Sigma_p^{-1}\Sigma_q = I $, so $\mathrm{tr}=k$.
- $ \log\det\Sigma_p - \log\det\Sigma_q = 0 $.

Plugging into the KL:

$$ \mathrm{tr}(\Sigma_p^{-1}\Sigma_q) - k = 0, $$

so

$$ \mathrm{KL}(q\Vert p) = \frac12 (\mu_p-\mu_q)^\top\Sigma_t^{-1}(\mu_p-\mu_q). $$

In diffusion, $ \Sigma_t $ is further chosen **diagonal / isotropic**:

- Typically $ \Sigma_t = \sigma_t^2 I $ or $ \tilde\beta_t I $.
- Hence
  $$ \mathrm{KL}(q\Vert p) = \frac{1}{2\sigma_t^2}\Vert\mu_p-\mu_q\Vert^2. $$

This is Eq. (87)’s key message:

- **The KL becomes “a constant coefficient × L2 loss.”**
- This gives a principled bridge from variational derivation to an MSE loss.

Your earlier intuition that “pixels aren’t correlated so a diagonal covariance feels right” shows up here: we could allow general $ \Sigma_q $, but for tractability and fewer parameters we **force** $ \Sigma_t $ to be diagonal / scalar — akin to a conjugate prior: closed form, simple loss.

---

## 5. From KL to Noise L2: (99), (108), (130)

Let’s ignore paper-specific wording and chain the algebra.

### 5.1 Goal: Turn
$ \mathrm{KL}\bigl(q(x_{t-1}\mid x_t,x_0)\Vert p_\theta(x_{t-1}\mid x_t)\bigr) $
into some $ \Vert\cdot\Vert^2 $

We already have

$$ \mathrm{KL}\bigl(q(x_{t-1}\mid x_t,x_0)\Vert p_\theta(x_{t-1}\mid x_t)\bigr) = \frac{1}{2\sigma_t^2} \bigl\Vert\mu_q(x_t,x_0) - \mu_\theta(x_t,t)\bigr\Vert^2 + \text{const}. $$

Next steps:

1. Write the closed form of $ \mu_q(x_{t-1}\mid x_t,x_0) $.
2. Parameterize $ \mu_\theta $ in a “reasonable” way so that it becomes **noise regression**.

### 5.2 Closed Form of $ \mu_q(x_{t-1}\mid x_t,x_0) $

Given:

- One forward step:
  $$ x_t = \sqrt{\alpha_t}x_{t-1} + \sqrt{1-\alpha_t}\varepsilon_t; $$
- Entire forward process:
  $$ x_t = \sqrt{\bar\alpha_t}x_0 + \sqrt{1-\bar\alpha_t}\varepsilon,\quad \varepsilon\sim\mathcal N(0,I). $$

Combine to get

$$ q(x_{t-1}\mid x_t,x_0) = \mathcal N\bigl(x_{t-1}\mid \mu_q(x_t,x_0),\Sigma_q(t)\bigr), $$

with mean

$$ \mu_q(x_t,x_0) = \frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1}) x_t + \sqrt{\bar\alpha_{t-1}}(1 -\alpha_t) x_0}{1-\bar{\alpha}_{t}}, $$

and $ \Sigma_q(t) $ is a fixed covariance (time-dependent only, image-independent) coming from the Gaussian posterior. Essentially it is linear regression: combine “forward Gaussian + prior” to get the posterior mean. Think of it as

> “In the Gaussian world where we only know $ x_t $ and $ x_0 $, compute the conditional mean of $ x_{t-1} $.”

Because the posterior is Gaussian, its mean must be a linear combination of $ x_t $ and $ x_0 $.

### 5.3 Noise Parameterization: $ \varepsilon_\theta $

We know the forward relation

$$ x_t = \sqrt{\bar\alpha_t}x_0 + \sqrt{1-\bar\alpha_t}\varepsilon. $$

Solve for $ \varepsilon $:

$$ \varepsilon = \frac{x_t - \sqrt{\bar\alpha_t}x_0}{\sqrt{1-\bar\alpha_t}}. $$

During training we **know $ x_0 $** and the sampled $ x_t $, so $ \varepsilon $ is a known label.

Now a small trick:

- Do not let the network directly predict $ \mu_\theta(x_t,t) $.
- Instead, let it predict $ \varepsilon_\theta(x_t,t) $ and define
  $$ \mu_\theta(x_t,t) := \mu_q\bigl(x_t, x_{\theta}(x_t,t)\bigr), $$
  or in the DDPM form,
  $$ \mu_\theta(x_t,t) = \frac{1}{\sqrt{\alpha_t}}\Bigl( x_t - \frac{1-\alpha_t}{\sqrt{1-\bar\alpha_t}} \varepsilon_\theta(x_t,t) \Bigr). $$

Then in the KL we get

$$ \bigl\Vert\mu_q(x_t,x_0) - \mu_\theta(x_t,t)\bigr\Vert^2 \propto \bigl\Vert\varepsilon - \varepsilon_\theta(x_t,t)\bigr\Vert^2. $$

Concrete check (simple version):

1. Closed-form mean:
   $$ \mu_q(x_t,x_0) = \frac{1}{\sqrt{\alpha_t}}\Bigl( x_t - \frac{1-\alpha_t}{\sqrt{1-\bar\alpha_t}}\varepsilon \Bigr). $$

2. Define $ \mu_\theta $ the same way but replace $ \varepsilon $ with $ \varepsilon_\theta $:
   $$ \mu_\theta(x_t,t) = \frac{1}{\sqrt{\alpha_t}}\Bigl( x_t - \frac{1-\alpha_t}{\sqrt{1-\bar\alpha_t}}\varepsilon_\theta(x_t,t) \Bigr). $$

3. Subtract:
   $$ \mu_q - \mu_\theta = -\frac{1-\alpha_t}{\sqrt{\alpha_t}\sqrt{1-\bar\alpha_t}} \bigl(\varepsilon - \varepsilon_\theta(x_t,t)\bigr). $$

4. L2:
   $$ \bigl\Vert\mu_q - \mu_\theta\bigr\Vert^2 = C_t^2\bigl\Vert\varepsilon - \varepsilon_\theta(x_t,t)\bigr\Vert^2, $$
   where $ C_t $ depends only on time and the schedule.

Multiplying by $ \tfrac{1}{2\sigma_t^2} $ in front of the KL leaves only **time-dependent constants**, so in training we can either:

- Keep these weights (getting a more precise time-weighted loss); or
- Treat them as “constant factors” and approximate with a simple uniform-weight loss.

The typical training objective (Eq. (99)/(108)/(130) family) is exactly this final result: **a noise L2 regression loss.**

---

## 6. Three Equivalent Parameterizations: $ \mu_\theta / x_{\theta} / \varepsilon_\theta $

This block is conceptual but crucial for implementation.

### 6.1 The Three Views Are the Same Thing

At each time step we may ask the network to predict one of three targets:

1. Mean $ \mu_\theta(x_t,t) $
2. Clean $ x_{\theta}(x_t,t) $ (an estimate of $x_0$)
3. Noise $ \varepsilon_\theta(x_t,t) $

They are equivalent under linear transforms. Starting from

$$ x_t = \sqrt{\bar\alpha_t}x_0 + \sqrt{1-\bar\alpha_t}\varepsilon, $$

one can solve for any of the three given the other two.

### 6.2 Practical Training Algorithm (corresponding to Eq. (130))

<!-- Assume dataset $ \{x_0^{(i)}\}_{i=1}^N$, network $\hat\varepsilon_\theta(x,t)$ (e.g., UNet). -->
Assume dataset $x_0^{(i)}$, network $\hat{\varepsilon}_{\theta}(x,t)$ (e.g., UNet).

One typical SGD/Adam training loop:

1. **Choose a noise schedule**

   Fix total steps $T$, specify $\beta_t$ (or directly $\alpha_t$):

   - $\alpha_t = 1-\beta_t$
   - $\bar\alpha_t = \prod_{s=1}^t \alpha_s$
   - From Eq. (85) set $\sigma_q^2(t)$:
     $$
     \sigma_q^2(t)
     = \frac{(1-\alpha_t)(1-\bar\alpha_{t-1})}{1-\bar\alpha_t},\qquad
     \Sigma_q(t) = \sigma_q^2(t)I.
     $$

   This is hyperparameter design before training.

2. **Define the overall objective (expectation form of Eq. (130))**

   Minimize

   $$
   \mathcal L(\theta)
   = E_{x_0} E_{t} E_{\varepsilon_0}
   \left[
     w_t \bigl\|\varepsilon_0 - \hat\varepsilon_\theta(x_t,t)\bigr\|_2^2
   \right],
   $$

   where $w_t$ is as above (or simplified). This is the final “weighted MSE on noise.”

3. **Mini-batch sampling each iteration**

   - Sample $B$ images $x_0^{(i)}$ from the dataset.
   - For each sample, draw a time step
     $$
     t^{(i)}\sim\text{Uniform}(1,\dots,T),
     $$
   - For each sample, draw Gaussian noise
     $$
     \varepsilon_0^{(i)} \sim \mathcal N(0,I).
     $$

4. **Use the closed-form forward to get $x_t$ in one shot**

   Using the direct formula (no need to iterate $x_0\to x_1\to\dots\to x_t$):

   $$
   x_t^{(i)} = \sqrt{\bar\alpha_{t^{(i)}}}x_0^{(i)} + \sqrt{1-\bar\alpha_{t^{(i)}}}\varepsilon_0^{(i)}.
   $$

   This makes training complexity independent of $T$ (depends only on batch size and network).

5. **Forward the network to predict noise**

   Feed $(x_t^{(i)}, t^{(i)})$:

   $$
   \hat\varepsilon_\theta^{(i)} = \hat\varepsilon_\theta\bigl(x_t^{(i)}, t^{(i)}\bigr).
   $$

   $t$ is usually embedded (e.g., sinusoidal time embedding) and fed into the UNet along with image features.

6. **Compute mini-batch loss (weighted Eq. (130))**

   Per-sample loss:

   $$
   L^{(i)}(\theta) = w_{t^{(i)}} \bigl\|\varepsilon_0^{(i)} - \hat\varepsilon_\theta^{(i)}\bigr\|_2^2
   $$





   Mini-batch loss is the mean (or sum):

   $$
   \mathcal L_{\text{batch}}(\theta) = \frac{1}{B}\sum_{i=1}^B L^{(i)}(\theta).
   $$

   This is Eq. (130) discretized onto concrete samples.

7. **Gradient descent / Adam update**

   Backprop on $\mathcal L_{\text{batch}}(\theta)$ to update network parameters:

   - For example Adam:
     $$
     \theta \leftarrow \text{AdamStep} \bigl(\theta,  \nabla_\theta \mathcal L_{\text{batch}}(\theta)\bigr),
     $$
   - Iterate for many epochs until convergence (loss stabilizes / generation quality is good).

The whole training loop is a stochastic-gradient approximation of Eq. (130):

- Expectation over $x_0$: iterate / sample from dataset.
- Expectation over $t$: random time step each iteration.
- Expectation over $\varepsilon_0$: add fresh Gaussian noise each time.

---

6.2.3 **Sampling (Inference) — Generating Images from Noise**

After training the noise predictor $\hat\varepsilon_\theta(x,t)$, sampling is **walking backward along the reverse Markov chain $p_\theta(x_{t-1}\mid x_t)$ step by step.**

Using the reverse-mean formula (Tweedie + linear Gaussian):

- General form:
  $$
  p_\theta(x_{t-1}\mid x_t) = \mathcal N\bigl(x_{t-1}; \mu_\theta(x_t,t),\Sigma_q(t)\bigr),
  $$
  where
  $$
  \mu_\theta(x_t,t) = \frac{1}{\sqrt{\alpha_t}}\left( x_t - \frac{1-\alpha_t}{\sqrt{1-\bar\alpha_t}} \hat\varepsilon_\theta(x_t,t) \right),
  $$
  and $\Sigma_q(t)=\sigma_q^2(t)I$ is the same as in training.

Sampling steps:

1. **Initialization**

   Sample the final state $x_T$ from the standard Gaussian prior:

   $$
   x_T \sim \mathcal N(0,I).
   $$

2. **Reverse iteration $t = T,T-1,\dots,1$**

   For each $t$:

   1. **Predict noise at current step**

      $$
      \hat\varepsilon_\theta(x_t,t) = \hat\varepsilon_\theta( x_t, t).
      $$

   2. **Compute reverse mean $\mu_\theta(x_t,t)$**

      Using the closed form:

      $$
      \mu_\theta(x_t,t) = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{1-\alpha_t}{\sqrt{1-\bar\alpha_t}} \hat\varepsilon_\theta(x_t,t) \right).
      $$

      Intuition: remove the predicted noise from the current noisy image and rescale to the previous step.

   3. **Add (or skip) sampling noise to get $x_{t-1}$**

      - If $t>1$: sample Gaussian noise
        $$
        z\sim\mathcal N(0,I), \qquad x_{t-1} = \mu_\theta(x_t,t) + \sigma_q(t)z,
        $$
        with $\sigma_q^2(t)$ the same as in training.
      - If $t=1$: many implementations just take the mean (no extra noise):
        $$
        x_0 = x_{t-1} = \mu_\theta(x_1,1).
        $$

      Then use $x_{t-1}$ as the next input.

3. **Obtain final sample**

   After the loop, we have $x_0$ — an image sampled from the trained VDM. For multiple samples, repeat independently.

---

6.2.4 **Mapping Between Training Objective and Sampling (Quick Summary)**

- Training:  
  One forward step is
  $$
  x_t = \sqrt{\bar\alpha_t}x_0 + \sqrt{1-\bar\alpha_t}\varepsilon_0,
  $$
  and the network learns to recover $\varepsilon_0$ from $(x_t,t)$. This is effectively learning the **score** of the noisy image.

- Sampling:  
  One reverse step is
  $$
  x_{t-1}
  = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{1-\alpha_t}{\sqrt{1-\bar\alpha_t}} \hat\varepsilon_\theta(x_t,t) \right) + \sigma_q(t)z.
  $$
  Use the learned noise predictor to denoise step by step with the preset $\alpha_t$ schedule until $t=0$.

- The whole VDM / DDPM story in one line:

  > **Training:** use the weighted MSE of Eq. (130) so the network predicts noise from $x_t$.  
  > **Sampling:** use the learned noise predictor in the reverse Gaussian chain to turn $x_T\sim\mathcal N(0,I)$ gradually into $x_0$.

This closes the loop from Eqs. (30)/(33)/(85)/(86)/(87)/(126)/(130) to the practical “how to train / how to sample.”

---

## 7. Tweedie Formula, Score-Based View, and Eqs. (133)/(148)/(151)

Here we only keep the map; no need to re-derive all SDE / score matching details.

### 7.1 Core of Tweedie’s Formula

Consider the basic additive-noise model $x_t = x_0 + \sigma_t \varepsilon$, with conditional $q(x_t\mid x_0)=\mathcal N(x_0,\sigma_t^2 I)$.

We want the posterior mean $E[x_0\mid x_t]$. Tweedie gives
$$E[x_0\mid x_t]=x_t+\sigma_t^2\nabla_{x_t}\log p(x_t)$$

Write the marginal $p(x_t)=\int p(x_0)q(x_t\mid x_0) dx_0$.

Gradient w.r.t. $x_t$:
$$\nabla_{x_t}p(x_t)=\nabla_{x_t}\int p(x_0)q(x_t\mid x_0)dx_0=\int p(x_0)\nabla_{x_t}q(x_t\mid x_0)dx_0$$

For $q$ in log form $\log q(x_t\mid x_0)=C-\frac{1}{2\sigma_t^2}\|x_t-x_0\|^2$,

$\nabla_{x_t}\log q(x_t\mid x_0)=-(x_t-x_0)/\sigma_t^2$.

Using $\nabla q=q\nabla\log q$,
$$\nabla_{x_t}q(x_t\mid x_0)=-(x_t-x_0)q(x_t\mid x_0)/\sigma_t^2$$

Plug back:
$$\nabla_{x_t}p(x_t)=\int p(x_0)\left[-\frac{x_t-x_0}{\sigma_t^2}q(x_t\mid x_0)\right]dx_0=-\frac{1}{\sigma_t^2}\int(x_t-x_0)p(x_0)q(x_t\mid x_0)dx_0$$

Note $p(x_0)q(x_t\mid x_0)=p(x_0,x_t)$, so
$$\nabla_{x_t}p(x_t)=-\frac{1}{\sigma_t^2}\int(x_t-x_0)p(x_0,x_t)dx_0$$
Factor out $p(x_t)$:
$$\nabla_{x_t}p(x_t)=-\frac{p(x_t)}{\sigma_t^2}\int(x_t-x_0)p(x_0\mid x_t)dx_0$$

Hence $\nabla_{x_t}p(x_t)=-\frac{p(x_t)}{\sigma_t^2}\left[x_t-E[x_0\mid x_t]\right]$.

Divide both sides by $p(x_t)$:
$$\nabla_{x_t}\log p(x_t)=-\frac{1}{\sigma_t^2}\left[x_t-E[x_0\mid x_t]\right]$$
Rearrange to Tweedie:
$$E[x_0\mid x_t]=x_t+\sigma_t^2\nabla_{x_t}\log p(x_t)$$


Apply to DDPM forward noise $q(x_t\mid x_0)=\mathcal N(\sqrt{\bar\alpha_t}x_0,(1-\bar\alpha_t)I)$, write as $x_t=\sqrt{\bar\alpha_t}x_0+\sigma_t\varepsilon,\quad \sigma_t^2=1-\bar\alpha_t$.

Let $\mu_t=\sqrt{\bar\alpha_t}x_0$. Tweedie on $\mu_t$ gives
$$E[\mu_t\mid x_t]=x_t+(1-\bar\alpha_t)\nabla_{x_t}\log p(x_t)$$

Using $\mu_t=\sqrt{\bar\alpha_t}x_0$ and dividing by $\sqrt{\bar\alpha_t}$,
$$E[x_0\mid x_t]=\frac{x_t+(1-\bar\alpha_t)\nabla_{x_t}\log p(x_t)}{\sqrt{\bar\alpha_t}}$$

Define score $s(x_t,t)=\nabla_{x_t}\log p(x_t)$, then
$$E[x_0\mid x_t]=\frac{x_t+(1-\bar\alpha_t)s(x_t,t)}{\sqrt{\bar\alpha_t}}$$ 

This is the essence of Eq. (133): “predict noise,” “predict $x_0$,” and “predict score” are equivalent in the Gaussian diffusion setting; only the parameterization differs.


---
### 7.2 From Discrete Diffusion to Continuous Score-Based Models: (148), (151)

Let time step $ t $ become continuous $ t\in[0,1] $ and $\sigma_t$ a continuous schedule:

- The forward process becomes an SDE.
- The reverse process becomes another SDE whose drift includes the score $ \nabla_x\log p_t(x) $.
- This unifies score-based generative models / diffusion SDEs.

Eqs. (148)/(151) basically say:

1. Train $ s_\theta(x_t,t) $ to approximate the score.
2. Plug it into the reverse SDE and integrate — effectively “reverse diffusion” in continuous time.

View it as:

> **The discrete DDPM Gaussian chain converges to a continuous-time SDE as $ \Delta t\to 0 $.  
> The learned $ \varepsilon_\theta $ / $ s_\theta $ are the same thing in two coordinate systems.**

---

## Closing Words

The core of Part 2 is to unpack the dense-looking VDM derivation into reusable modules:

1. Forward chain: Markov + closed-form $ \bar\alpha_t $ + terminal standard normal
2. Variational decomposition: split $ -\log p_\theta(x_0) $ into prior / recon / transition KL
3. Markovity: when $ x_0 $ can be dropped, when it must stay
4. Gaussian KL: general formula → equal-covariance special case → L2 loss
5. Noise parameterization: write $ \mu_q $ as a function of $ \varepsilon $, turning KL into $ \Vert\varepsilon - \varepsilon_\theta\Vert^2 $
6. Practical training: sample $ x_0,t,\varepsilon $, do one-step noisy reconstruction
7. Tweedie / score: “predict noise” vs “predict score” are coordinate changes; the later SDE story is the continuous-time limit of the same physics.


This note walks through the chain in Probabilistic Machine Learning ch.26: **CK → Kramers–Moyal → Fokker–Planck → forward/reverse SDE → Probability Flow ODE → Murphy’s DDPM formulas**, making it self-contained.

## 0. Notation

- Write a single coordinate $x$; in higher dims replace $\partial_x$ with $\nabla_x$, Laplacian with $\Delta_x$.
- Density $p_t(x)$ denotes the density of $X_t$.
- Forward Itô SDE: $dx_t = a(x_t,t) dt + b(x_t,t) d\omega$.
- For DDPM:  
  - Forward SDE: $dx_t = f(x_t,t) dt + g(t) d\omega$  
  - Common form: $dx_t = -\tfrac12 \beta(t) x_t dt + \beta(t) d\omega$, i.e. $g(t)=\beta(t)$ (used below).
- In Kramers–Moyal use $\Delta = x - y$ (new position minus old) so the FP first-order sign is $- \partial_x [A_1 p]$.

## 1. CK → Kramers–Moyal Expansion

### 1.1 Chapman–Kolmogorov Equation

Transition kernel $T(x,t+\Delta t \mid y,t)$, CK:

$$
p(x,t+\Delta t) = \int_{R} T(x,t+\Delta t \mid y,t) p(y,t) dy
$$

### 1.2 Change of Variables: $\Delta = x - y$

With $\Delta = x-y \Leftrightarrow y = x-\Delta$ and $dy = - d\Delta$ (flip limits to drop the minus):

$$
p(x,t+\Delta t) = \int_{R} T(x,t+\Delta t \mid x-\Delta,t) p(x-\Delta,t) d\Delta
$$

### 1.3 Taylor Expand at $x$

Define $F(x,t;\Delta) := T(x,t+\Delta t \mid x-\Delta,t) p(x-\Delta,t)$. Fix $\Delta$, expand w.r.t. $x$:

$$
F(x,t;\Delta) = \sum_{n=0}^{\infty} \frac{(-\Delta)^n}{n!} \partial_x^n \big[T(x,t+\Delta t \mid x,t) p(x,t)\big]
$$

Plug back, using $(-\Delta)^n = (-1)^n \Delta^n$:

$$
p(x,t+\Delta t) = \sum_{n=0}^{\infty} \frac{(-1)^n}{n!} \partial_x^n \left[p(x,t)\int \Delta^n T(x,t+\Delta t \mid x,t) d\Delta\right]
$$

### 1.4 Define $M_n$ and $A_n$

$M_n(x,t;\Delta t) := \int \Delta^n T(x,t+\Delta t \mid x,t) d\Delta = E[(\Delta X)^n \mid X_t=x]$, so

$$
p(x,t+\Delta t) = \sum_{n=0}^{\infty} \frac{(-1)^n}{n!} \partial_x^n [M_n(x,t;\Delta t) p(x,t)]
$$

Set $M_0=1$, drop the constant, divide by $\Delta t$, define

$$
A_n(x,t) := \lim_{\Delta t \to 0} \frac{M_n(x,t;\Delta t)}{\Delta t} = \lim_{\Delta t \to 0} \frac{1}{\Delta t} E[(\Delta X)^n \mid X_t=x]
$$

Then the Kramers–Moyal series:

$$
\partial_t p(x,t) = \sum_{n=1}^{\infty} \frac{(-1)^n}{n!} \partial_x^n [A_n(x,t) p(x,t)]
$$

### 1.5 Diffusion-Scaling Assumption → Truncate to 2nd Order → FP

If $E[\Delta X]=O(\Delta t)$, $E[(\Delta X)^2]=O(\Delta t)$, and $E[(\Delta X)^n]=O((\Delta t)^{n/2})$ for $n\ge3$, then $A_1,A_2$ finite and $A_{n\ge3}=0$, leaving

$$
\partial_t p(x,t) = -\partial_x[A_1 p] + \tfrac12 \partial_x^2[A_2 p]
$$

## 2. Fokker–Planck and Itô SDE

For SDE $dx_t = a(x_t,t) dt + b(x_t,t) d\omega$:

Consider 1D Itô SDE

$$
dx_t = a(x_t,t) dt + b(x_t,t) d\omega.
$$

Small-step increment

$$
\Delta X = X_{t+\Delta t} - X_t.
$$

By definition of an Itô process:

### (1) First Conditional Moment

$$
E[\Delta X \mid X_t=x] = a(x,t) \Delta t + o(\Delta t).
$$

### (2) Second Conditional Moment

$$
E[(\Delta X)^2 \mid X_t=x] = b(x,t)^2 \Delta t + o(\Delta t).
$$

From Kramers–Moyal definition

$$
A_n(x,t) := \lim_{\Delta t\to 0} \frac{1}{\Delta t} E[(\Delta X)^n\mid X_t=x],
$$

plugging above gives

$$
A_1(x,t) = a(x,t), \qquad A_2(x,t) = b(x,t)^2.
$$

Under the diffusion truncation ($A_{n\ge 3}=0$), Kramers–Moyal to 2nd order yields FP:

$$
\partial_t p(x,t) = -\partial_x\big[a(x,t) p(x,t)\big] + \frac12 \partial_x^2 \big[b(x,t)^2 p(x,t)\big].
$$

This links an Itô SDE with its FP:

- Drift $a(x,t)$ ↔ first-order term in FP
- Diffusion $b(x,t)$ ↔ second-order term
- Note this is the forward FP; the reverse FP is

$$
-\partial_t p(x,t) = \partial_x\big[a(x,t) p(x,t)\big] + \frac12 \partial_x^2 \big[b(x,t)^2 p(x,t)\big].
$$



## 3. Continuity Equation and Probability Flow ODE

Consider particles in $\mathbb{R}^d$ following deterministic ODE
$$
\frac{dX_t}{dt} = h(X_t,t),
$$
with smooth velocity field $h(x,t)$. The spatial distribution induces a time-varying density $p_t(x)$. Goal: derive the continuity equation for the density:

$$
\partial_t p_t(x) = - \nabla_x \cdot \bigl(h(x,t) p_t(x)\bigr),
$$
expressing conservation of “mass” (probability).

---

### 3.1 **Mass and Flux in a Region**

For any bounded region $B \subset \mathbb{R}^d$, mass is
$$
M_B(t) := \int_B p_t(x)dx.
$$

Particles only cross boundary $\partial B$, so the change in $M_B(t)$ is determined by boundary flux. With outward normal $n(x)$, flux out of $B$:
$$
\int_{\partial B} p_t(x) h(x,t)\cdot n(x) dS.
$$

“Time change of mass inside = negative outward flux”:
$$
\frac{d}{dt} \int_B p_t(x)dx
= - \int_{\partial B} p_t(x) h(x,t)\cdot n(x) dS.
$$

---

### 3.2 **Divergence Theorem**

Convert boundary integral to volume integral:
$$
\int_{\partial B} p_t h\cdot n dS
= \int_B \nabla_x \cdot (p_t h) dx.
$$

Substitute:
$$
\frac{d}{dt} \int_B p_t(x)dx
= - \int_B \nabla_x \cdot \bigl(p_t(x) h(x,t)\bigr) dx.
$$

---

### 3.3 **Move Time Derivative Inside Integral**

Under smoothness,
$$
\int_B \partial_t p_t(x) dx
= - \int_B \nabla_x \cdot \bigl(p_t(x) h(x,t)\bigr) dx.
$$

---

### 3.4 **$B$ Arbitrary ⇒ Integrands Equal**

Since true for any $B$, the integrands must match almost everywhere:
$$
\partial_t p_t(x)
= - \nabla_x \cdot \bigl(h(x,t) p_t(x)\bigr).
$$

This is the continuity (Liouville) equation, guaranteeing total mass conservation:
$$
\frac{d}{dt} \int_{\mathbb{R}^d} p_t(x) dx = 0.
$$

---

### 3.5 **Summary: Particle ODE ↔ Density PDE**

Overall,
$$
dx = h(x,t)dt
\Longleftrightarrow
\partial_t p_t(x) = - \nabla_x \cdot \bigl(h(x,t) p_t(x)\bigr).
$$

## 4. Probability Flow ODE of the Forward SDE (Murphy 25.47)

Forward SDE $dx_t = f(x_t,t) dt + g(t) d\omega$ has FP

$$
\partial_t p_t = -\nabla \cdot (f p_t) + \tfrac12 g(t)^2 \Delta_x p_t
$$

Using $\Delta_x p_t = \nabla \cdot (p_t \nabla \log p_t)$,

$$
\partial_t p_t = -\nabla \cdot \big[(f - \tfrac12 g^2 \nabla \log p_t) p_t\big]
$$

Thus the corresponding deterministic ODE is

$$
dx_t = \big[f(x_t,t) - \tfrac12 g(t)^2 \nabla_x \log p_t(x_t)\big] dt
$$

## 5. Time Reversal and the Reverse SDE (Murphy 25.49)

Starting from the forward SDE, a “time-reversed” SDE exists whose drift has an extra score correction.

### 5.1 Forward SDE and Marginal

Forward Itô SDE

$$
dx_t = f(x_t,t) dt + g(t) d\omega,
\qquad t \in [0,T].
$$

Let the marginal density be

$$
X_t \sim q_t(x), \qquad \text{i.e. } q_t(x) = \text{Prob}(X_t \in dx).
$$

### 5.2 Define the Reverse-Time Process

Intuition: reverse time is the same path seen backwards.

Define

$$
Y_\tau := X_{T - \tau}, \qquad \tau \in [0,T].
$$

- $\tau = 0$: $Y_0 = X_T$
- $\tau = T$: $Y_T = X_0$

So $Y_\tau$ is $X_t$ with time flipped. We want an SDE for $Y_\tau$ whose marginal at $\tau$ equals $q_{T-\tau}$.

Below we still write $t$ for “reverse-time” parameter; just remember forward runs $0\to T$, reverse runs $T\to 0$ on the same axis.

### 5.3 Discrete-Time Intuition

Think of a discrete chain with step $\Delta t$:

$$
X_{k+1} \sim p(x_{k+1} \mid x_k), \qquad k = 0,1,\dots,N-1,
$$

with marginals $q_k(x)$. Forward evolution:

$$
q_{k+1}(x') = \int p(x' \mid x) q_k(x) dx.
$$

Reversing (sample $X_{k+1}$ then generate $X_k$) via Bayes gives the reverse kernel:

$$
\tilde{p}(x \mid x') = \text{Prob}(X_k = x \mid X_{k+1} = x') = \frac{p(x' \mid x) q_k(x)}{q_{k+1}(x')}.
$$

The reverse kernel depends on forward kernel **and** marginals $q_k,q_{k+1}$ — this is why $\nabla_x \log q_t(x)$ (the score) appears in the continuous limit: marginals enter the reverse dynamics.

### 5.4 Anderson’s Continuous-Time Result

In the limit, Anderson’s theorem states:

If $X_t$ follows forward SDE
 
$$
  dx_t = f(x_t,t) dt + g(t) d\omega, \qquad t \in [0,T],
$$
 
 with marginals $q_t(x)$. Define the reverse process $Y_\tau = X_{T-\tau}$; then $Y_t$ satisfies an Itô SDE
 
 $$
  dx_t = \big[ f(x,t) - g(t)^2 \nabla_x \log q_t(x) \big] dt + g(t) d\omega,
 $$
 
where $W_t$ is the reverse-time Wiener process.

So the reverse SDE shares the diffusion $g(t)$ but its drift gains a score term $ -g(t)^2 \nabla_x \log q_t(x) $. This is Murphy Eq. (25.49).

### 5.5 Approximate the Score with a Network

In practice we don’t know $q_t(x)$, nor its score

$$
\nabla_x \log q_t(x).
$$

Introduce a neural net $s_\theta(x,t)$ to approximate the score:

$$
s_\theta(x,t) \approx \nabla_x \log q_t(x).
$$

Then the practical reverse SDE replaces the true score by $s_\theta$.

---

## 6. Apply to DDPM (25.50, 25.52, 25.53)

### 6.1 DDPM Forward SDE Form

Discrete DDPM noising converges to the continuous SDE

$$
dx_t = -\frac{1}{2} \beta(t) x_t dt + \sqrt{\beta(t)} d\omega.
$$

Here

- $f(x,t) = -\frac{1}{2} \beta(t) x$
- $g^2(t) = \beta(t)$

### 6.2 Reverse SDE (Murphy 25.50)

Plug $f,g$ into Anderson’s reverse SDE:

Forward:

$$
dx_t = -\frac{1}{2} \beta(t) x_t dt + \sqrt{\beta(t)} d\omega.
$$

With marginals $q_t(x)$, the reverse SDE is

$$
dx_t = \left[ -\frac{1}{2} \beta(t) x_t - \beta(t) \nabla_{x_t} \log q_t(x_t) \right] dt + \sqrt{\beta(t)} d\omega.
$$

Drift = original linear term $ -\tfrac12 \beta(t) x_t $ plus a score correction $ -\beta(t) \nabla_{x_t} \log q_t(x_t) $.

### 6.3 Approximate Score with $s_\theta$ (25.52)

Train $s_\theta(x,t)$ to approximate the score:

$$
s_\theta(x_t,t) \approx \nabla_{x_t} \log q_t(x_t).
$$

Replace the score to get the implementable reverse SDE:

$$
dx_t = \left[ -\frac{1}{2} \beta(t) x_t - \beta(t) s_\theta(x_t,t) \right] dt + \sqrt{\beta(t)} d\omega.
$$

Or compactly

$$
dx_t = -\frac{1}{2} \beta(t) \big[ x_t + 2 s_\theta(x_t,t) \big] dt + \sqrt{\beta(t)} d\omega.
$$

This is Murphy Eq. (25.52).

### 6.4 Euler–Maruyama Discretization (25.53)

For general SDE

$$
dx_t = \mu(x_t,t) dt + \sigma(t) d\omega,
$$

Euler–Maruyama with step $\Delta t$ gives

$$
x_{t - \Delta t} = x_t + \mu(x_t,t) \Delta t + \sigma(t) \sqrt{\Delta t} \varepsilon_t,
$$

with $\varepsilon_t \sim \mathcal{N}(0,I)$.

For DDPM reverse SDE, drift

$$
\mu(x_t,t) = -\frac{1}{2} \beta(t) [x_t + 2 s_\theta(x_t,t)],
$$

diffusion

$$
\sigma(t) = \sqrt{\beta(t)}.
$$

Plugging yields the discrete reverse update:

$$
x_{t - 1} = x_t + \frac{1}{2} \beta(t) [x_t + 2 s_\theta(x_t,t)] \Delta t +  \sqrt{\beta(t) \Delta t} \varepsilon_t.
$$

This is Eq. (25.53).

---

## 7. FP → ODE Again: Reverse Probability Flow ODE (25.54, 25.55)

Here the logic mirrors “forward SDE → FP → probability flow ODE,” but for the reverse SDE.

### 7.1 FP of the Reverse SDE

Start from the reverse SDE (with $s_\theta$):

$$
dx_t = \big[ f(x_t,t) - g(t)^2 s_\theta(x_t,t) \big] dt + g(t) d\omega.
$$

Write abstractly:

- Drift $\tilde{f}(x,t) = f(x,t) - g(t)^2 s_\theta(x,t)$
- Diffusion $g(t)$

Forward FP for this reverse SDE (note sign flip):

$$
- \partial_t p_t(x) =  \nabla_x \cdot \big( \tilde{f}(x,t) p_t(x) \big) + \frac{1}{2} g(t)^2 \Delta_x p_t(x).
$$

So

$$
\partial_t p_t(x) =  - \nabla_x \cdot \big( [f(x,t) - g(t)^2 s_\theta(x,t)] p_t(x) \big) - \frac{1}{2} g(t)^2 \Delta_x p_t(x).
$$

### 7.2 Write Diffusion Term as Divergence

Using

$$
\Delta_x p_t(x) = \nabla_x \cdot \big( \nabla_x p_t(x) \big),
$$

and

$$
\nabla_x p_t(x) = p_t(x) \nabla_x \log p_t(x),
$$

we get

$$
\Delta_x p_t(x) = \nabla_x \cdot \big( p_t(x) \nabla_x \log p_t(x) \big).
$$

Thus

$$
\frac{1}{2} g(t)^2 \Delta_x p_t(x) = \frac{1}{2} g(t)^2 \nabla_x \cdot \big( p_t(x) \nabla_x \log p_t(x) \big).
$$

### 7.3 Merge Drift and Diffusion

Substitute into FP:

$$
\partial_t p_t(x) = - \nabla_x \cdot \big( [f - g^2 s_\theta] p_t \big) - \frac{1}{2} g^2 \nabla_x \cdot \big( p_t \nabla_x \log p_t \big).
$$

Omitting explicit $(x,t)$ for brevity ($p_t=p_t(x)$, $f=f(x,t)$, $s_\theta=s_\theta(x,t)$, $g=g(t)$):

First term

$$
 {} - \nabla_x \cdot \big( [f - g^2 s_\theta] p_t \big),
$$

Second term

$$
\frac{1}{2} g^2 \nabla_x \cdot \big( p_t \nabla_x \log p_t \big).
$$

Combine as one divergence:

$$
\partial_t p_t = - \nabla_x \cdot \Big( [f - g^2 s_\theta] p_t + \frac{1}{2} g^2 p_t \nabla_x \log p_t \Big).
$$

Note the sign: writing the second term as

$$
+\frac{1}{2} g^2 \nabla_x \cdot(\cdots) = - \nabla_x \cdot\Big( \frac{1}{2} g^2 p_t \nabla_x \log p_t\Big),
$$

gives

$$
\partial_t p_t = - \nabla_x \cdot \Big( [f - g^2 s_\theta] p_t + \frac{1}{2} g^2 p_t \nabla_x \log p_t \Big).
$$

In the ideal case with the true score in drift,

$$
s_\theta(x,t) = \nabla_x \log p_t(x),
$$

the vector field simplifies:

$$
[f - g^2 s_\theta] + \frac{1}{2} g^2 \nabla_x \log p_t = f - g^2 \nabla_x \log p_t + \frac{1}{2} g^2 \nabla_x \log p_t = f - \frac{1}{2} g^2 \nabla_x \log p_t.
$$

Thus the reverse probability flow ODE uses drift

$$
f(x,t) - \frac{1}{2} g(t)^2 s_\theta(x,t),
$$

not $f - g^2 s_\theta$. The reverse probability flow ODE:

$$
dx_t = \big[ f(x_t,t) - \frac{1}{2} g(t)^2 s_\theta(x_t,t) \big] dt.
$$

### 7.4 Plug in DDPM’s $f,g$ to Get (25.55)

For DDPM,

$$
f(x,t) = -\frac{1}{2} \beta(t) x, \qquad g(t) = \sqrt{\beta(t)}.  $$

Then

$$
f(x_t,t) - \frac{1}{2} g(t)^2 s_\theta(x_t,t) = -\frac{1}{2} \beta(t) x_t -
\frac{1}{2} \beta(t) s_\theta(x_t,t).
$$

Many papers absorb $\beta(t)^2$ into a rescaled time; Murphy writes the simple form:

$$
dx_t = -\frac{1}{2} \beta(t) \big[ x_t + s_\theta(x_t,t) \big] dt,
$$

the reverse diffusion ODE (Eq. 25.55). Euler discretization:

$$
x_{t - \Delta t} = x_t + \frac{1}{2} \beta(t) \big[ x_t + s_\theta(x_t,t) \big] \Delta t.
$$

This is the deterministic ODE sampling rule (no stochastic noise).

---

## 8. Reverse SDE = ODE + Langevin Noise (25.57)

The reverse SDE can be split into a probability-flow ODE part plus a score-aligned Langevin part.

### 8.1 From Eq. 25.52

Recall reverse SDE with $s_\theta$:

$$
dx_t = -\frac{1}{2} \beta(t) \big[ x_t + 2 s_\theta(x_t,t) \big] dt + \beta(t) d\omega.
$$

Expand:

$$
dx_t = -\frac{1}{2} \beta(t) \big[ x_t + s_\theta(x_t,t) \big] dt - \frac{1}{2} \beta(t) s_\theta(x_t,t) dt + \beta(t) d\omega.
$$

### 8.2 Identify ODE vs Langevin Parts

Split as:

- First term

  $$
  -\frac{1}{2} \beta(t) \big[ x_t + s_\theta(x_t,t) \big] dt
  $$

  is exactly the reverse probability-flow ODE drift.

- Remaining terms

  $$
  -\frac{1}{2} \beta(t) s_\theta(x_t,t) dt + \beta(t) d\omega
  $$

  form a Langevin step along the score direction plus Gaussian noise.

So the full SDE can be viewed as:

- Take a small ODE step along the probability-flow field,
- Then apply a bit of Langevin noise correction.

### 8.3 Sampling Strategies

Common options:

1. **Pure SDE sampling**  
   Use the full reverse SDE (25.52) with Euler–Maruyama; add noise every step → stochastic trajectories.

2. **Pure ODE sampling (deterministic)**  
   Drop the noise/Langevin part, keep only the reverse probability-flow ODE (25.55). Given the same $x_T$, solving the ODE yields the same $x_0$ deterministically.

3. **Hybrid (ODE + light Langevin)**  
   Use the split in 25.57: mostly ODE trajectory with occasional Langevin steps to “correct” the distribution.

Core idea: **The same set of marginals $q_t$ can be generated by a reverse SDE or a reverse ODE; the SDE can be decomposed into ODE + Langevin noise.**

With the earlier chain CK → Kramers–Moyal → FP → SDE → probability-flow ODE, we now connect Murphy’s Eqs. 25.49, 25.50, 25.52, 25.53, 25.54, 25.55, 25.57 into a single unified picture.

---

## 9. Summary: SDE ↔ FP ↔ ODE ↔ Time Reversal at a Glance

- CK: $p(x,t+\Delta t) = \int T(x,t+\Delta t \mid y,t) p(y,t) dy$
- Change variable $\Delta = x-y$, Taylor expand, define $M_n,A_n$
- Kramers–Moyal: $\partial_t p = \sum_{n\ge1} (-1)^n/n! \partial_x^n[A_n p]$
- Diffusion scaling, truncate high order → FP: $\partial_t p = -\partial_x[A_1 p] + \tfrac12 \partial_x^2[A_2 p]$
- Itô link: $A_1=a, A_2=b^2$ → SDE $dx = a dt + b dW$
- Continuity equation: $\partial_t p = -\nabla \cdot (h p)$ ↔ $dx = h dt$
- Forward SDE FP: $-\nabla \cdot (f p) + \tfrac12 g^2 \Delta p$, rewrite as PF ODE $dx = [f - \tfrac12 g^2 \nabla \log p] dt$ (25.47)
- Anderson time reversal: reverse SDE $dx = [f - g^2 \nabla \log q_t] dt + g dW$ (25.49)
- DDPM forward $f=-\tfrac12 \beta x, g=\beta$ → reverse SDE (25.50); replace by $s_\theta$ (25.52)
- FP→ODE again: diffusion “eats half the score,” reverse PF ODE $dx = [f - \tfrac12 g^2 s_\theta] dt$ (25.54), plug DDPM → $dx = -\tfrac12 \beta(t)[x + s_\theta] dt$ (25.55)
- Reverse SDE can be seen as ODE + Langevin noise (25.57)

Whole chain fixed by the initial sign convention $\Delta = x - y$ and the Itô FP correspondence; all signs and coefficients follow.

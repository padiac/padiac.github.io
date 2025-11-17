## A Unified View with Detailed Balance

These notes organize how random walks, diffusion, DMC, and MCMC connect through branching, continuity, and detailed balance.

---

## 1. Random walks and the diffusion equation

Consider a 1D lattice with spacing $\Delta x$ and discrete time step $\Delta t$. A single walker moves according to

- at each step it either jumps left or right by $\Delta x$,
- with probabilities $T_+(x)$ (from $x-\Delta x$ to $x$) and $T_-(x)$ (from $x+\Delta x$ to $x$),
- and (optionally) may be killed or duplicated by a reaction term $\mu(x)$.

Let $p(x,t)$ be the probability that a single walker is at position $x$ at time $t$. The one-step evolution rule is

$$
p(x, t + \Delta t)
= p(x-\Delta x, t) T_+(x-\Delta x) + p(x+\Delta x, t) T_-(x+\Delta x).
$$

For the simplest symmetric case we take

$$
T_+(x) = T_-(x) = \frac{1 - \mu(x)\Delta t}{2}.
$$

- The factor $1 - \mu(x)\Delta t$ represents the probability that the walker survives the reaction in this step.
- The factor $1/2$ gives equal chance to go left or right conditional on survival.

This is already a discrete conservation law for a single walker: probability at $x$ at the next time step comes entirely from neighboring sites.

### 1.1 Taylor expansion and continuum limit

Expand the left-hand side of the random-walk update in $\Delta t$ and the right-hand side in $\Delta x$. To second order we obtain

$$
p(x, t + \Delta t)
= p(x,t) + \frac{\partial p}{\partial t} \Delta t + O(\Delta t^2).
$$

$$
p(x \pm \Delta x, t) 
= p(x,t) \pm \frac{\partial p}{\partial x}\Delta x + \frac{1}{2}\frac{\partial^2 p}{\partial x^2}(\Delta x)^2 + O(\Delta x^3).
$$

Insert these Taylor expansions into the update equation, enforce the symmetric choice $T_+=T_-=(1-\mu\Delta t)/2$, and keep terms up to first order in $\Delta t$ and second order in $\Delta x$:

$$
p(x,t) + \frac{\partial p}{\partial t}\Delta t
= \left[p(x,t) + \frac{1}{2} \frac{\partial^2 p}{\partial x^2} (\Delta x)^2\right]
   (1 - \mu(x)\Delta t) + O(\Delta t^2,\Delta x^3).
$$

Subtract $p(x,t)$ from both sides, divide by $\Delta t$, and take the limit $\Delta t \to 0$ while enforcing the Brownian scaling

$$
D = \lim_{\Delta t\to 0} \frac{(\Delta x)^2}{2\Delta t}
\quad (\text{finite}).
$$

Taking the limit with this scaling turns the discrete relation into the diffusion-reaction equation

$$
\frac{\partial p}{\partial t}
= D \frac{\partial^2 p}{\partial x^2} - \mu(x) p.
$$

- The $D \partial_x^2 p$ term comes from the symmetric random walk.
- The $-\mu(x)p$ term comes from the survival factor $1-\mu\Delta t$. Positive $\mu$ corresponds to annihilation (probability decays). Negative $\mu$ corresponds to branching (effective growth).

Without the reaction ($\mu \equiv 0$) this reduces to the standard diffusion equation.

### 1.2 Why $\Delta x^2 \propto \Delta t$ is necessary

The Brownian scaling $(\Delta x)^2 \sim 2D\Delta t$ is not decoration; it is the condition that the discrete walk has a well-defined continuum limit with finite diffusion constant.

- If $\Delta x$ is fixed while $\Delta t \to 0$, then $(\Delta x)^2/\Delta t \to \infty$: the walker can move arbitrarily far in arbitrarily short time. No reasonable diffusion PDE emerges.
- If $\Delta x$ shrinks faster than $\sqrt{\Delta t}$, diffusion vanishes.

So whenever a random walk is claimed to approximate diffusion, some version of this Brownian scaling is hidden in the construction.

---

## 2. From single-walker probability to density: DMC picture

Now imagine $N$ independent walkers. Let the initial density of walkers be $\rho(x_0, t_0)$. Each walker starting at $x_0$ has transition probability $p(x,t \mid x_0, t_0)$ obtained from the diffusion-reaction PDE above.

The ensemble density at time $t$ is

$$
\rho(x,t)
= \int \rho(x_0, t_0) p(x,t \mid x_0,t_0) dx_0.
$$

Formally, $p(x,t\mid x_0,t_0)$ plays the role of a Green's function of the PDE, and summing over all walkers' starting positions just convolves the Green's function with the initial density.

Crucially, because each $p(\cdot,\cdot\mid x_0,t_0)$ satisfies that PDE, the density $\rho(x,t)$ must also satisfy the same PDE:

$$
\frac{\partial \rho}{\partial t}
= D \frac{\partial^2 \rho}{\partial x^2} - \mu(x)\rho.
$$

This is precisely the continuum equation that Diffusion Monte Carlo (DMC) is discretizing and simulating, with walkers as particles whose population can branch or die according to the sign of $\mu(x)$. At large imaginary time (in DMC language) the density relaxes to the ground state eigenfunction of the corresponding operator.

---

## 3. Continuity equation vs reaction term

When $\mu(x)\equiv 0$, the diffusion-reaction PDE reduces to the standard continuity equation with flux definition below.

$$
\frac{\partial \rho}{\partial t} + \nabla\cdot J = 0.
$$

$$
J = -D\nabla \rho.
$$

This expresses local conservation: the only way to change density at $x$ is through net flux from neighboring points.

When $\mu(x)\neq 0$, the PDE becomes

$$
\frac{\partial \rho}{\partial t} + \nabla\cdot J = - \mu(x)\rho,
$$

which is a continuity equation with a source/sink term: $\mu(x)\rho$ locally creates or destroys walkers.

DMC implements this by

- random walk moves (diffusion part), and
- branching or annihilation moves controlled by $\mu(x)\Delta t$.

So, in the DMC picture:

- The random walk part enforces something like a local discretized continuity equation.
- The branching part is the discrete analog of the reaction term $-\mu\rho$.

---

## 4. Stationary distributions and global balance for Markov chains

Now switch point of view to a general Markov chain. Let $X_n$ be a discrete-time Markov chain with state space $\mathcal{X}$ (continuous or discrete), and transition kernel

$$
T(x\to y) \equiv \Pr[X_{n+1}=y \mid X_n=x].
$$

A probability density (or mass function) $\pi(x)$ is stationary if

$$
\pi(y) = \int_{\mathcal{X}} \pi(x) T(x\to y) dx
\quad (\text{discrete sum if }\mathcal{X}\text{ is countable}).
$$

This can be read as a global balance equation:

- $\pi(y)$ is the total mass sitting at $y$ in steady state.
- The right-hand side is total mass flowing into $y$ from all states $x$ via one transition.

No local geometry, PDE, or time scale is assumed here. $T(x\to y)$ can be wild: it may jump arbitrarily far, in any way we like, as long as the stationarity integral condition above is satisfied.

This is fundamentally different from the diffusion PDE:

- Diffusion encodes local conservation plus local flux laws.
- A Markov chain stationary equation encodes only a global balance of inflow and outflow per step.

In particular, there is no notion of $\Delta t$, $\Delta x$, or Brownian scaling; MCMC has no physical time unless we choose to interpret iteration count as "time" for convenience.

---

## 5. Detailed balance and the Metropolis-Hastings acceptance rule

### 5.1 From desired stationary distribution to constraints on $T$

Suppose we want to design a Markov chain with a given target distribution $\pi(x)$. A very convenient sufficient condition is the detailed balance condition

$$
\pi(x) T(x\to y) = \pi(y) T(y\to x)
\quad \forall x,y.
$$

- Summing both sides over $x$ immediately yields the global balance relation, so detailed balance $\Rightarrow$ global balance.
- Detailed balance is stronger than necessary, but makes life easy.

Now write the transition as

$$
T(x\to y) = Q(x\to y)\alpha(x\to y),
$$

where

- $Q(x\to y)$ is a proposal kernel (how we suggest moves),
- $\alpha(x\to y)\in[0,1]$ is an acceptance probability.

For $y\neq x$, we allow $Q(x\to y) > 0$ and define

$$
Q(x\to x) = 1 - \int_{y\neq x} Q(x\to y) dy
$$

to make $Q$ stochastic. In continuous spaces, the stay-put probability is absorbed into the diagonal mass of the kernel.

Insert the decomposition $T=Q\alpha$ into the detailed-balance condition:

$$
\pi(x) Q(x\to y)\alpha(x\to y)
= \pi(y) Q(y\to x)\alpha(y\to x).
$$

This is the constraint that $\alpha$ must satisfy.

Define for brevity

$$
R(x,y) = \frac{\pi(y)Q(y\to x)}{\pi(x)Q(x\to y)}.
$$

This is equivalent to

$$
\alpha(x\to y) = R(x,y)\alpha(y\to x).
$$

We know nothing else about $\alpha(x\to y)$ except that it must lie in $[0,1]$. So we need to construct a pair $(\alpha(x\to y),\alpha(y\to x))$ that satisfies this relation and respects $0 \le \alpha \le 1$.

### 5.2 Constructing Metropolis-Hastings from detailed balance

Metropolis-Hastings' trick is exactly this construction.

#### Case 1: $R(x,y) \le 1$

Choose

$$
\alpha(x\to y) = R(x,y), \quad
\alpha(y\to x) = 1.
$$

Then the relation holds trivially, and both acceptances are in $[0,1]$ because $0\le R\le 1$.

#### Case 2: $R(x,y) \ge 1$

Choose

$$
\alpha(x\to y) = 1, \quad
\alpha(y\to x) = \frac{1}{R(x,y)}.
$$

Now $0\le 1/R \le 1$ because $R\ge 1$, and the relation holds again.

Putting both cases together gives the familiar MH form

$$
\alpha(x\to y)
= \min\left(1, \frac{\pi(y)Q(y\to x)}{\pi(x)Q(x\to y)}\right).
$$

### 5.3 What really happened (and what did not happen)

Important logical point:

- We did not prove detailed balance from this $\alpha$.
- We started by demanding detailed balance for $\pi$.
- Then we constructed an $\alpha$ that makes detailed balance true.

So the correct causality is "Want detailed balance for pi" $\Rightarrow$ "choose this MH acceptance". Many expositions fall into a circular-looking argument like:

1. Assume $\alpha$ has the MH form.
2. Plug into the detailed-balance constraint.
3. Verify equality.
4. Conclude "therefore MH is correct".

This hides the fact that $\alpha$ was specifically designed to make that equality hold. The verification is just a sanity check, not a proof of necessity.

---

## 6. Diffusion dynamics vs MCMC dynamics

We can now cleanly compare DMC-type diffusion and MCMC.

### 6.1 Diffusion (DMC) dynamics

- There is an underlying PDE (diffusion-reaction) of the form $\partial_t \rho = D\nabla^2 \rho - \mu(x)\rho$.
- The random walk is constructed so that its continuum limit is exactly this PDE. This imposes Brownian scaling $\Delta x^2 \sim 2D\Delta t$.
- Trajectories are "continuous" in the sense that each step only reaches nearby sites; long-distance moves appear only as the cumulative effect of many small steps.
- The number of walkers in a region approximates $\rho(x,t)$, and the dynamics is a genuine time evolution.

In other words, DMC is literally simulating a physical (or imaginary-time) evolution equation.

### 6.2 MCMC dynamics

For MCMC we only require $\pi(y) = \int \pi(x) T(x\to y) dx$. There is no PDE in the background, and no intrinsic notion of time step.

- $T(x\to y)$ can jump arbitrarily far in one step.
- There is no requirement of local conservation or Brownian scaling.
- The "time index" $n$ is just iteration count; different implementations can mix states in totally different ways while having the same $\pi$.

MH acceptance $\alpha$ is simply a balance corrector that ensures $\pi$ is stationary for the chosen $Q$; it has no physical interpretation unless we choose to impose one.

### 6.3 When do they look similar?

If we choose in MCMC:

- a local proposal $Q(x\to y)$ that only moves to nearest neighbors,
- $Q$ symmetric ($Q(x\to y)=Q(y\to x)$),
- acceptance $\alpha\approx 1$ (for example by tuning proposal step size small),

then the resulting chain's large-scale behavior resembles a diffusion process. In an appropriate scaling limit,

- iteration count plays the role of $t/\Delta t$,
- proposal step size plays the role of $\Delta x$,
- the chain's empirical distribution evolves like a solution to a diffusion PDE.

But this is an optional limit:

- DMC must satisfy such a relation to approximate the PDE.
- MCMC may or may not; it is free to do wild jumps and still have the correct $\pi$.

---

## 7. Unified view and summary

1. **Random walk -> diffusion PDE**  
   A symmetric nearest-neighbor walk with Brownian scaling $\Delta x^2 = 2D\Delta t$ gives, in the continuum limit, the diffusion equation (with optional reaction term). DMC explicitly uses this to represent ground-state wavefunctions as long-time limits of a branching random walk.
2. **Single-walker probability vs density**  
   The single-walker probability $p(x,t\mid x_0,t_0)$ solves the diffusion PDE. An ensemble density $\rho(x,t) = \int \rho(x_0,t_0) p(x,t\mid x_0,t_0) dx_0$ solves the same PDE. DMC walkers are a Monte Carlo realization of $\rho$.
3. **Continuity equation**  
   When $\mu=0$, the PDE becomes $\partial_t\rho + \nabla\cdot J = 0$, $J=-D\nabla\rho$: a strict local conservation law. With $\mu\neq 0$, we get a continuity equation plus source/sink term.
4. **Markov chain stationary equation**  
   For a general chain with transition kernel $T$, stationarity of $\pi$ means $\pi(y) = \int \pi(x) T(x\to y) dx$. This is a global balance condition with no local structure and no time scale.
5. **Detailed balance and MH**  
   Imposing $\pi(x)T(x\to y)=\pi(y)T(y\to x)$ is a convenient sufficient condition for stationarity. Writing $T = Q\alpha$ and solving $\pi(x)Q(x\to y)\alpha(x\to y)=\pi(y)Q(y\to x)\alpha(y\to x)$ leads directly to the Metropolis-Hastings acceptance $\alpha(x\to y) = \min(1,\pi(y)Q(y\to x)/(\pi(x)Q(x\to y)))$.
6. **Diffusion vs MCMC**  
   - DMC = real (imaginary-time) dynamics approximating a PDE, forced to obey Brownian scaling and local conservation.  
   - MCMC = fake dynamics engineered so that a given $\pi$ is stationary; it need not correspond to any PDE or local conservation law.
7. **Connection**  
   In a specific scaling limit with local symmetric proposals and high acceptance, MCMC trajectories begin to look like diffusion trajectories, and the empirical distribution evolves approximately like a diffusion equation. But this is an emergent similarity, not a fundamental requirement of MCMC.

In this sense,

> **Diffusion is one special, physically motivated kind of Markov dynamics.  
> MCMC is a huge family of Markov dynamics, most of which have no reason at all to resemble diffusion, except when we deliberately force them to.**

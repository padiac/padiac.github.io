## A Unifying Markov-Process Perspective

---

## 0. Overview

This note collects and systematizes how random walks, diffusion equations, diffusion Monte Carlo (DMC), and Markov chain Monte Carlo (MCMC) interrelate through Markov generators.

- **Random walks** on a lattice
- The **diffusion equation** and diffusion-reaction equations
- **Diffusion Monte Carlo (DMC)** as a particle method for solving PDEs
- **Markov Chain Monte Carlo (MCMC)** as a method for sampling a target distribution
- Their shared link via **Markov generators** and the analogy between **Newtonian vs fluid** descriptions in physics

Key theme:

> At a high level, all of these are different ways to use Markov processes to approximate measures (probability distributions, densities, or PDE solutions). The apparent differences come from which layer we inspect: microscopic particles, macroscopic densities, or stationary distributions.

## 1. Three Monte Carlo Traditions

There are three classical "Monte Carlo" traditions that often get mixed up.

### 1.1 Classical Monte Carlo (MC)

Goal: estimate integrals, expectations, areas, or volumes by averaging independent samples drawn from a known measure $\mu$.

$$
I = \int f(x) d\mu(x) \approx \frac{1}{N} \sum_{i=1}^{N} f(x_i)
$$

Here $x_i \sim \mu$ i.i.d., and no Markov chain is required; samples can be independent. This is the basic "throw random points and average" picture.

### 1.2 Markov Chain Monte Carlo (MCMC)

Goal: sample from a target density $\pi(x)$ known up to normalization by constructing a Markov chain with transition kernel $T(x \to x')$ such that

$$
\pi(x) T(x \to x') = \pi(x') T(x' \to x)
$$

or at least

$$
\pi(x') = \int \pi(x) T(x \to x') dx.
$$

The stationary distribution of the chain is $\pi$, and time averages over the chain approximate expectations under $\pi$.

### 1.3 Diffusion Monte Carlo (DMC)

Goal: use a particle system (random walk plus branching) to solve a diffusion-reaction PDE.

$$
\frac{\partial \rho}{\partial t} = D \frac{\partial^2 \rho}{\partial x^2} - \mu(x) \rho(x, t)
$$

In quantum Monte Carlo this corresponds to the imaginary-time Schrodinger equation; in general it is a diffusion-reaction equation. A swarm of walkers performs diffusion steps and is duplicated or killed according to $\mu(x)$; their empirical density approximates $\rho(x, t)$.

Despite different stories, all three methods approximate properties of some measure (integral, target distribution, PDE solution).

## 2. Microscopic Random Walk vs Macroscopic Diffusion

### 2.1 Discrete-time, nearest-neighbor random walk with a stay term

Consider a 1D lattice with spacing $\Delta x$ and time step $\Delta t$. Let $p(x, t)$ be the probability for a single walker to be at position $x$ at time $t$.

Allow three possibilities per step:

- move left by $-\Delta x$ with probability $T_{+}$
- move right by $+\Delta x$ with probability $T_{-}$
- stay put with probability $T_{0}$

The master equation is

$$
p(x, t + \Delta t) = T_{+} p(x - \Delta x, t) + T_{-} p(x + \Delta x, t) + T_{0} p(x, t).
$$

To allow a sink or source term we impose

$$
T_{+} + T_{-} + T_{0} = 1 - \mu \Delta t,
$$

where $\mu(x)$ later corresponds to a reaction rate in the PDE.

### 2.2 Taylor expansion and the continuum limit

Expand $p(x \pm \Delta x, t)$ and $p(x, t + \Delta t)$ for small $\Delta x$ and $\Delta t$. Write $p$, $p_{x}$, $p_{xx}$, and $p_{t}$ for partial derivatives at $(x, t)$.

$$
p(x, t + \Delta t) = p + p_{t} \Delta t + O(\Delta t^2)
$$

$$
p(x - \Delta x, t) = p - p_{x} \Delta x + \frac{1}{2} p_{xx} (\Delta x)^2 + O(\Delta x^3)
$$

$$
p(x + \Delta x, t) = p + p_{x} \Delta x + \frac{1}{2} p_{xx} (\Delta x)^2 + O(\Delta x^3)
$$

Substituting into the master equation gives

$$
p + p_{t} \Delta t = T_{+} \left(p - p_{x} \Delta x + \frac{1}{2} p_{xx} (\Delta x)^2 \right) + T_{-} \left(p + p_{x} \Delta x + \frac{1}{2} p_{xx} (\Delta x)^2 \right) + T_{0} p + \ldots
$$

Collecting terms leads to

$$
p_{t} \Delta t = -\mu \Delta t \, p + (T_{-} - T_{+}) p_{x} \Delta x + \frac{1}{2} (T_{+} + T_{-}) p_{xx} (\Delta x)^2 + \ldots
$$

### 2.3 Symmetric random walk: no drift

For pure diffusion with no drift enforce $T_{+} = T_{-} = A$. Then $T_{-} - T_{+} = 0$ and $T_{+} + T_{-} = 2A$, yielding

$$
p_{t} \Delta t = -\mu \Delta t \, p + A (\Delta x)^2 p_{xx} + \ldots
$$

Divide by $\Delta t$ to obtain

$$
p_{t} = -\mu p + A \frac{(\Delta x)^2}{\Delta t} p_{xx} + O(\Delta x^2, \Delta t).
$$

Define the diffusion constant

$$
D = A \frac{(\Delta x)^2}{\Delta t}
$$

and take the continuum limit $\Delta x \to 0$, $\Delta t \to 0$ with $D$ fixed to arrive at

$$
p_{t} = D p_{xx} - \mu p.
$$

### 2.4 Where did the stay-put probability go?

$T_{0}$ only appears through the constraint

$$
T_{+} + T_{-} + T_{0} = 1 - \mu \Delta t,
$$

so once $D$ and $\mu$ are fixed, $A = T_{+} = T_{-}$ and $T_{0}$ are linked by

$$
2A + T_{0} = 1 - \mu \Delta t
$$

and

$$
D = A \frac{(\Delta x)^2}{\Delta t}.
$$

There are infinitely many microscopic choices of $T_{+}$, $T_{-}$, and $T_{0}$ that produce the same continuum PDE; the stay probability is absorbed into $D$ and $\mu$.

### 2.5 The scaling $(\Delta x)^2 \propto \Delta t$ and its meaning

From

$$
D = A \frac{(\Delta x)^2}{\Delta t}
$$

we see that to keep $D$ finite and nonzero while $\Delta x \to 0$ and $\Delta t \to 0$ we must have

$$
(\Delta x)^2 = 2 D \Delta t
$$

up to convention. If $(\Delta x)^2 \ll \Delta t$, then $D \to 0$ and the diffusion term vanishes (no spatial spreading). If $(\Delta x)^2 \gg \Delta t$, then $D \to \infty$ and the system flattens instantly. Choosing $\Delta t$ therefore fixes the natural scale for $\Delta x$ when approximating a specific diffusion constant.

## 3. Single-Walker vs Density: Newtonian vs Fluid View

There is a crucial conceptual separation.

1. **Microscopic view (Newton-like):** track a single walker trajectory $X(t)$. Each step follows left/right/stay probabilities, and no explicit conservation law is enforced at this level.
2. **Macroscopic view (fluid-like):** consider a very large number $N$ of walkers with empirical density

   $$
   \rho(x, t) = \frac{1}{N} \sum_{i=1}^{N} \delta(x - X_i(t)).
   $$

   In the limit $N \to \infty$, $\rho(x, t)$ obeys the continuity law

   $$
   \frac{\partial \rho}{\partial t} = - \frac{\partial J}{\partial x},
   $$

   with flux

   $$
   J = -D \frac{\partial \rho}{\partial x},
   $$

   leading to

   $$
   \frac{\partial \rho}{\partial t} = D \frac{\partial^2 \rho}{\partial x^2},
   $$

   and including reaction gives

   $$
   \frac{\partial \rho}{\partial t} = D \frac{\partial^2 \rho}{\partial x^2} - \mu \rho.
   $$

This mirrors classical analogies: Newtonian mechanics vs fluid mechanics, or kinetic theory vs hydrodynamics. Random walks describe particles; diffusion equations describe densities.

## 4. Diffusion Monte Carlo (DMC) as a PDE Solver

### 4.1 PDE viewpoint

At the PDE level DMC aims to solve

$$
\frac{\partial \rho}{\partial t} = D \frac{\partial^2 \rho}{\partial x^2} - \mu(x) \rho(x, t).
$$

Here $\rho(x, t)$ is a particle density or, after normalization, a probability density. $D$ is a diffusion constant, and $\mu(x)$ is a net death rate (positive for sinks, negative for sources). DMC approximates this PDE by letting walkers diffuse and branch:

- Diffusion steps implement $D \partial^2 \rho / \partial x^2$.
- Branching implements the reaction term $-\mu(x) \rho$.

The walkers' empirical density approximates $\rho(x, t)$. In the long-time limit, DMC can approach a steady state $\rho_{\ast}(x)$ satisfying

$$
0 = D \frac{\partial^2 \rho_{\ast}}{\partial x^2} - \mu(x) \rho_{\ast}(x).
$$

### 4.2 Imaginary-time Schrodinger equation as a special case

In quantum Monte Carlo one starts from the imaginary-time Schrodinger equation

$$
\frac{\partial \psi}{\partial \tau} = D \frac{\partial^2 \psi}{\partial x^2} - (V(x) - E_T) \psi,
$$

where $V(x)$ is a potential and $E_T$ is a reference energy. The TDSE solution decomposes as

$$
\psi(x, \tau) = \sum_{n} c_n \psi_n(x) e^{-E_n \tau} \to c_0 \psi_0(x)
$$

as $\tau \to \infty$. This is just a diffusion-reaction equation with

$$
\mu(x) = V(x) - E_T.
$$

Thus the Schrodinger flavor of DMC is a physical interpretation of a general diffusion-reaction Monte Carlo method; the same machinery solves classical diffusion-reaction PDEs.

## 5. MCMC vs DMC: Are They Really Different?

At a high level both MCMC and DMC evolve Markov processes (possibly with branching), approach a stationary or steady-state density $\rho_{\ast}$, and can be seen as realizations of

$$
\frac{\partial \rho}{\partial t} = L \rho
$$

for some generator $L$.

### 5.1 MCMC: stationary distribution is given

In MCMC you are given a target density $\pi(x) \propto \exp(-E(x))$. You construct a transition kernel $T(x \to x')$ so that $\pi$ is stationary:

$$
\pi(x') = \int \pi(x) T(x \to x') dx,
$$

often via detailed balance. The chain $X_0, X_1, \ldots$ samples from $\pi$ in the long-time limit. The stationary density is therefore part of the input, and the dynamics are engineered to preserve it.

### 5.2 DMC: stationary distribution is the solution of a PDE

In DMC you are given a PDE or operator $L_{\text{DMC}}$:

$$
\frac{\partial \rho}{\partial t} = L_{\text{DMC}} \rho = D \frac{\partial^2 \rho}{\partial x^2} - \mu(x) \rho.
$$

The stationary density satisfies $L_{\text{DMC}} \rho_{\ast} = 0$ and is unknown. The diffusion-plus-branching process simulates the action of $L_{\text{DMC}}$ and relaxes toward $\rho_{\ast}$. Here the generator is the starting point and the density is the output.

### 5.3 They can be made equivalent by construction

Given a DMC generator $L_{\text{DMC}}$, you can discretize time and space to form a Markov chain $T_{\text{DMC}}$ that approximates the same evolution—essentially a specialized MCMC scheme whose stationary distribution matches the PDE's steady state.

Given a target density $\pi(x)$, you can construct a diffusion-reaction PDE whose steady-state solution is $\rho_{\ast}(x) \propto \pi(x)$, for instance by choosing

$$
\mu(x) = -\log \pi(x)
$$

up to constants and carefully designing drift and branching so that $\rho_{\ast}$ equals $\pi$. A DMC-like particle system can then sample from $\pi$ via PDE relaxation.

Thus, in an abstract sense, MCMC and DMC can be embedded into the same Markov-process framework and even simulate each other when designed deliberately that way. Nevertheless, MCMC is naturally used when the target density is known, whereas DMC is naturally used to solve PDEs or eigenvalue problems where the density is unknown.

## 6. Detailed Balance, Global Balance, and Conservation

### 6.1 MCMC: detailed vs global balance

Detailed balance,

$$
\pi(x) T(x \to x') = \pi(x') T(x' \to x),
$$

is a strong condition ensuring $\pi$ is stationary and the chain is reversible. Global balance,

$$
\pi(x') = \int \pi(x) T(x \to x') dx,
$$

is weaker and only requires stationarity. These are algebraic constraints on $T(x \to x')$ and specific to MCMC's target-density story.

### 6.2 Diffusion PDE: conservation and reaction

For diffusion without reaction the conservation law reads

$$
\frac{\partial \rho}{\partial t} + \frac{\partial J}{\partial x} = 0,
$$

with flux $J = -D \partial \rho / \partial x$. Adding reaction yields

$$
\frac{\partial \rho}{\partial t} = D \frac{\partial^2 \rho}{\partial x^2} - \mu(x) \rho,
$$

where $\mu(x) \rho$ is not locally balanced; it adds or removes mass.

### 6.3 Single walker: no explicit conservation

For a single walker you do not enforce conservation; you just evolve its position according to random steps and possible death or branching. Conservation emerges statistically when inspecting the density of many walkers, not at the level of one trajectory.

## 7. The Unifying Markov-Generator View

All of the above can be unified by the language of Markov generators.

### 7.1 General form

A Markov process on a state space $X$ has a generator $L$ acting on densities $\rho(x, t)$ such that

$$
\frac{\partial \rho}{\partial t} = L \rho.
$$

Examples include diffusion-reaction,

$$
(L \rho)(x) = D \frac{\partial^2 \rho}{\partial x^2} - \mu(x) \rho(x),
$$

Fokker-Planck diffusion,

$$
(L \rho)(x) = - \frac{\partial}{\partial x} \left( b(x) \rho \right) + \frac{1}{2} \frac{\partial^2}{\partial x^2} \left( a(x) \rho \right),
$$

and jump processes of the MCMC type,

$$
(L \rho)(x') = \int \left[ T(x \to x') \rho(x) - T(x' \to x) \rho(x') \right] dx.
$$

A stationary distribution $\rho_{\ast}$ satisfies

$$
L \rho_{\ast} = 0.
$$

### 7.2 MCMC and DMC in this language

- **MCMC:** $L$ is chosen (through $T$) so that a prescribed $\pi$ satisfies $L \pi = 0$; the generator is engineered from $\pi$.
- **DMC:** $L$ is given by a PDE; $\rho_{\ast}$ solves $L \rho_{\ast} = 0$; the generator is the starting point and the stationary density is discovered.

Both are "Markov process + generator + evolution $\partial \rho / \partial t = L \rho$"; the difference is which part is input vs output.

### 7.3 Newton vs fluid analogy

The discussion mirrors the Newtonian vs fluid analogy: microscopic random walks correspond to Newton-like trajectories, whereas diffusion PDEs correspond to fluid-like densities. MCMC and DMC walkers are microscopic trajectories; their stationary or time-dependent densities are macroscopic fields.

## 8. Takeaways

1. **Random walk and diffusion:** A nearest-neighbor random walk with scaling $(\Delta x)^2 \propto \Delta t$ leads to diffusion or diffusion-reaction equations. The stay-put probability $T_{0}$ is absorbed into $D$ and $\mu$.
2. **Microscopic vs macroscopic:** Single walkers follow simple rules, while large ensembles obey PDEs, mirroring Newtonian vs fluid descriptions.
3. **DMC:** Random walk plus branching solves diffusion-reaction PDEs (including imaginary-time Schrodinger), and steady states emerge in the long-time limit.
4. **MCMC:** Markov chains sample from a given target distribution; detailed and global balance ensure stationarity.
5. **Unification:** Both MCMC and DMC are Markov processes governed by generators $L$; in MCMC the stationary distribution is input, whereas in DMC it is output. Each can emulate the other when deliberately constructed that way, confirming that they are projections of the same Markov-structure framework.
